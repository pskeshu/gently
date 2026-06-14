"""
Model evaluation — metrics, confusion matrix, per-stage reporting.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Complete evaluation report for a trained model."""

    run_id: str = ""
    accuracy: float = 0.0
    per_stage_precision: dict[str, float] = field(default_factory=dict)
    per_stage_recall: dict[str, float] = field(default_factory=dict)
    per_stage_f1: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    total_samples: int = 0
    correct: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "accuracy": self.accuracy,
            "per_stage_precision": self.per_stage_precision,
            "per_stage_recall": self.per_stage_recall,
            "per_stage_f1": self.per_stage_f1,
            "confusion_matrix": self.confusion_matrix,
            "class_names": self.class_names,
            "total_samples": self.total_samples,
            "correct": self.correct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationReport":
        return cls(
            run_id=d.get("run_id", ""),
            accuracy=d.get("accuracy", 0.0),
            per_stage_precision=d.get("per_stage_precision", {}),
            per_stage_recall=d.get("per_stage_recall", {}),
            per_stage_f1=d.get("per_stage_f1", {}),
            confusion_matrix=d.get("confusion_matrix", []),
            class_names=d.get("class_names", []),
            total_samples=d.get("total_samples", 0),
            correct=d.get("correct", 0),
        )

    def summary(self) -> str:
        """Human-readable summary for the agent."""
        lines = [f"Accuracy: {self.accuracy:.1%} ({self.correct}/{self.total_samples})"]
        if self.per_stage_f1:
            lines.append("Per-stage F1:")
            for stage, f1 in sorted(self.per_stage_f1.items(), key=lambda x: -x[1]):
                prec = self.per_stage_precision.get(stage, 0)
                rec = self.per_stage_recall.get(stage, 0)
                lines.append(f"  {stage}: F1={f1:.3f} (P={prec:.3f}, R={rec:.3f})")
        return "\n".join(lines)


def evaluate_model(
    model,
    data_loader,
    class_names: list[str],
    device=None,
    run_id: str = "",
) -> EvaluationReport:
    """Evaluate a model on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    data_loader : DataLoader
        Test dataset loader.
    class_names : list of str
        Names for each class index.
    device : torch.device, optional
        Device to use.
    run_id : str
        Training run identifier.

    Returns
    -------
    EvaluationReport
    """
    try:
        import torch
    except ImportError:
        return EvaluationReport(run_id=run_id)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    num_classes = len(class_names)

    # Confusion matrix
    cm = [[0] * num_classes for _ in range(num_classes)]
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)

            for true, pred in zip(batch_y.cpu().tolist(), predicted.cpu().tolist(), strict=False):
                cm[true][pred] += 1
                total += 1
                if true == pred:
                    correct += 1

    # Compute per-class metrics
    precision = {}
    recall = {}
    f1 = {}

    for i, name in enumerate(class_names):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(num_classes)) - tp
        fn = sum(cm[i][j] for j in range(num_classes)) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision[name] = round(p, 4)
        recall[name] = round(r, 4)
        f1[name] = round(f, 4)

    accuracy = correct / total if total > 0 else 0.0

    return EvaluationReport(
        run_id=run_id,
        accuracy=round(accuracy, 4),
        per_stage_precision=precision,
        per_stage_recall=recall,
        per_stage_f1=f1,
        confusion_matrix=cm,
        class_names=class_names,
        total_samples=total,
        correct=correct,
    )


def save_evaluation(report: EvaluationReport, path: Path):
    """Save evaluation report to JSON."""
    path.write_text(json.dumps(report.to_dict(), indent=2))


def load_evaluation(path: Path) -> EvaluationReport:
    """Load evaluation report from JSON."""
    data = json.loads(path.read_text())
    return EvaluationReport.from_dict(data)
