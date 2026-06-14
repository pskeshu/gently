"""
ML data models — pipelines, training runs, configs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TrainingStatus(str, Enum):
    """Status of an ML pipeline or training run."""

    PLANNED = "planned"
    DATA_PREP = "data_prep"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelArchitectureType(str, Enum):
    """Supported model architecture families."""

    RESNET_18 = "resnet18"
    RESNET_50 = "resnet50"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B4 = "efficientnet_b4"
    MOBILENET_V3 = "mobilenet_v3"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"


@dataclass
class ModelConfig:
    """Configuration for a model architecture."""

    architecture: str = "resnet18"
    num_classes: int = 8
    pretrained: bool = True
    input_channels: int = 1  # grayscale microscopy
    input_size: int = 224
    dropout: float = 0.2
    freeze_backbone_epochs: int = 5  # freeze backbone for N epochs, then unfreeze

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "dropout": self.dropout,
            "freeze_backbone_epochs": self.freeze_backbone_epochs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(
            architecture=d.get("architecture", "resnet18"),
            num_classes=d.get("num_classes", 8),
            pretrained=d.get("pretrained", True),
            input_channels=d.get("input_channels", 1),
            input_size=d.get("input_size", 224),
            dropout=d.get("dropout", 0.2),
            freeze_backbone_epochs=d.get("freeze_backbone_epochs", 5),
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    mixed_precision: bool = True  # AMP on A5000
    early_stopping_patience: int = 10
    augmentations: list[str] = field(
        default_factory=lambda: [
            "random_horizontal_flip",
            "random_rotation",
            "random_brightness",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
            "warmup_epochs": self.warmup_epochs,
            "mixed_precision": self.mixed_precision,
            "early_stopping_patience": self.early_stopping_patience,
            "augmentations": self.augmentations,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return cls(
            batch_size=d.get("batch_size", 32),
            epochs=d.get("epochs", 50),
            learning_rate=d.get("learning_rate", 1e-4),
            weight_decay=d.get("weight_decay", 1e-4),
            lr_scheduler=d.get("lr_scheduler", "cosine"),
            warmup_epochs=d.get("warmup_epochs", 5),
            mixed_precision=d.get("mixed_precision", True),
            early_stopping_patience=d.get("early_stopping_patience", 10),
            augmentations=d.get("augmentations", []),
        )


@dataclass
class DataSplit:
    """Defines how data is split for training."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_by: str = "stage"  # stratify splits by stage label
    session_ids: list[str] = field(default_factory=list)
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "stratify_by": self.stratify_by,
            "session_ids": self.session_ids,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataSplit":
        return cls(
            train_ratio=d.get("train_ratio", 0.7),
            val_ratio=d.get("val_ratio", 0.15),
            test_ratio=d.get("test_ratio", 0.15),
            stratify_by=d.get("stratify_by", "stage"),
            session_ids=d.get("session_ids", []),
            random_seed=d.get("random_seed", 42),
        )


@dataclass
class TrainingRun:
    """State of a single training run."""

    id: str = ""
    pipeline_id: str = ""
    status: str = TrainingStatus.PLANNED.value
    model_config: ModelConfig | None = None
    training_config: TrainingConfig | None = None
    data_split: DataSplit | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    best_val_accuracy: float = 0.0
    model_weights_path: str = ""
    metrics_path: str = ""
    peer_instance_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pipeline_id": self.pipeline_id,
            "status": self.status,
            "model_config": self.model_config.to_dict() if self.model_config else None,
            "training_config": self.training_config.to_dict() if self.training_config else None,
            "data_split": self.data_split.to_dict() if self.data_split else None,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "best_val_accuracy": self.best_val_accuracy,
            "model_weights_path": self.model_weights_path,
            "metrics_path": self.metrics_path,
            "peer_instance_id": self.peer_instance_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingRun":
        mc = d.get("model_config")
        tc = d.get("training_config")
        ds = d.get("data_split")
        return cls(
            id=d.get("id", ""),
            pipeline_id=d.get("pipeline_id", ""),
            status=d.get("status", TrainingStatus.PLANNED.value),
            model_config=ModelConfig.from_dict(mc) if mc else None,
            training_config=TrainingConfig.from_dict(tc) if tc else None,
            data_split=DataSplit.from_dict(ds) if ds else None,
            current_epoch=d.get("current_epoch", 0),
            total_epochs=d.get("total_epochs", 0),
            train_loss=d.get("train_loss", 0.0),
            val_loss=d.get("val_loss", 0.0),
            val_accuracy=d.get("val_accuracy", 0.0),
            best_val_accuracy=d.get("best_val_accuracy", 0.0),
            model_weights_path=d.get("model_weights_path", ""),
            metrics_path=d.get("metrics_path", ""),
            peer_instance_id=d.get("peer_instance_id", ""),
            started_at=d.get("started_at", ""),
            completed_at=d.get("completed_at", ""),
            error_message=d.get("error_message", ""),
        )


@dataclass
class MLPipeline:
    """Top-level pipeline that coordinates one ML task."""

    id: str = ""
    campaign_id: str = ""
    name: str = ""
    task: str = "embryo_stage_classification"
    status: str = TrainingStatus.PLANNED.value
    model_config: ModelConfig | None = None
    data_split: DataSplit | None = None
    training_config: TrainingConfig | None = None
    best_run_id: str = ""
    best_accuracy: float = 0.0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "campaign_id": self.campaign_id,
            "name": self.name,
            "task": self.task,
            "status": self.status,
            "model_config": self.model_config.to_dict() if self.model_config else None,
            "data_split": self.data_split.to_dict() if self.data_split else None,
            "training_config": self.training_config.to_dict() if self.training_config else None,
            "best_run_id": self.best_run_id,
            "best_accuracy": self.best_accuracy,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MLPipeline":
        mc = d.get("model_config")
        ds = d.get("data_split")
        tc = d.get("training_config")
        return cls(
            id=d.get("id", ""),
            campaign_id=d.get("campaign_id", ""),
            name=d.get("name", ""),
            task=d.get("task", "embryo_stage_classification"),
            status=d.get("status", TrainingStatus.PLANNED.value),
            model_config=ModelConfig.from_dict(mc) if mc else None,
            data_split=DataSplit.from_dict(ds) if ds else None,
            training_config=TrainingConfig.from_dict(tc) if tc else None,
            best_run_id=d.get("best_run_id", ""),
            best_accuracy=d.get("best_accuracy", 0.0),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
