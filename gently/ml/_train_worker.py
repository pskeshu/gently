"""
Training worker script — runs in a subprocess.

Reads configuration from a JSON file, trains a model, and reports
progress via JSON lines to progress.jsonl.

Usage: python _train_worker.py <config.json>
"""

import json
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python _train_worker.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text())

    model_config = config.get("model_config", {})
    training_config = config.get("training_config", {})
    data_root = Path(config.get("data_root", "."))
    labels_file = Path(config.get("labels_file", ""))
    output_dir = Path(config.get("output_dir", "."))
    progress_file = Path(config.get("progress_file", output_dir / "progress.jsonl"))
    weights_dir = Path(config.get("weights_dir", output_dir / "weights"))
    metrics_file = Path(config.get("metrics_file", output_dir / "metrics.json"))

    weights_dir.mkdir(parents=True, exist_ok=True)

    # Import torch (guarded)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
    except ImportError:
        _write_progress(progress_file, {"error": "PyTorch not installed"})
        sys.exit(1)

    try:
        import torchvision.models as models  # noqa: F401
    except ImportError:
        _write_progress(progress_file, {"error": "torchvision not installed"})
        sys.exit(1)

    # Load labels
    if labels_file.exists():
        labels_data = json.loads(labels_file.read_text())
    else:
        _write_progress(progress_file, {"error": f"Labels file not found: {labels_file}"})
        sys.exit(1)

    # Build datasets
    from gently.ml.data_loader import create_data_splits

    architecture = model_config.get("architecture", "resnet18")
    num_classes = model_config.get("num_classes", 8)
    input_size = model_config.get("input_size", 224)
    input_channels = model_config.get("input_channels", 1)
    pretrained = model_config.get("pretrained", True)
    dropout = model_config.get("dropout", 0.2)
    freeze_backbone_epochs = model_config.get("freeze_backbone_epochs", 5)

    epochs = training_config.get("epochs", 50)
    batch_size = training_config.get("batch_size", 32)
    lr = training_config.get("learning_rate", 1e-4)
    weight_decay = training_config.get("weight_decay", 1e-4)
    mixed_precision = training_config.get("mixed_precision", True)
    early_stopping_patience = training_config.get("early_stopping_patience", 10)

    # Create datasets from labels
    train_data, val_data, test_data = create_data_splits(
        labels_data,
        data_root,
        input_size,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(architecture, num_classes, pretrained, input_channels, dropout)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if mixed_precision and device.type == "cuda" else None

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        # Freeze/unfreeze backbone
        if epoch == freeze_backbone_epochs:
            for param in model.parameters():
                param.requires_grad = True

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                if scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), weights_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Write progress
        _write_progress(
            progress_file,
            {
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "best_val_accuracy": round(best_val_acc, 4),
                "lr": optimizer.param_groups[0]["lr"],
                "timestamp": time.time(),
            },
        )

        # Early stopping
        if patience_counter >= early_stopping_patience:
            break

    # Save final metrics
    metrics = {
        "final_train_loss": round(train_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "final_val_accuracy": round(val_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "epochs_trained": epoch + 1,
        "architecture": architecture,
    }
    metrics_file.write_text(json.dumps(metrics, indent=2))

    # Save final weights
    torch.save(model.state_dict(), weights_dir / "final_model.pt")


def _build_model(architecture, num_classes, pretrained, input_channels, dropout):
    """Build a torchvision model with modified first conv and classifier."""
    import torch.nn as nn
    import torchvision.models as models

    weights = "DEFAULT" if pretrained else None

    if architecture.startswith("resnet"):
        if architecture == "resnet18":
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet50(weights=weights)
        # Modify first conv for grayscale
        if input_channels != 3:
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            if pretrained and input_channels == 1:
                # Average RGB weights for grayscale
                model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        # Replace classifier
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )

    elif architecture.startswith("efficientnet"):
        if architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
        elif architecture == "efficientnet_b2":
            model = models.efficientnet_b2(weights=weights)
        else:
            model = models.efficientnet_b4(weights=weights)
        # Modify first conv
        if input_channels != 3:
            old_conv = model.features[0][0]
            out_channels = old_conv.out_channels
            model.features[0][0] = nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif architecture == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights=weights)
        if input_channels != 3:
            old_conv = model.features[0][0]
            out_channels = old_conv.out_channels
            model.features[0][0] = nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif architecture.startswith("convnext"):
        if architecture == "convnext_tiny":
            model = models.convnext_tiny(weights=weights)
        else:
            model = models.convnext_small(weights=weights)
        if input_channels != 3:
            old_conv = model.features[0][0]
            out_channels = old_conv.out_channels
            model.features[0][0] = nn.Conv2d(
                input_channels,
                out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model


def _write_progress(path: Path, data: dict):
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
