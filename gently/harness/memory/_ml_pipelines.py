"""
MlPipelinesMixin — ContextStore extension for ML pipeline state.

Provides tables and methods for:
- ML pipelines (top-level training tasks)
- Training runs (individual experiments within a pipeline)
- Data assessments (pre-training data analysis snapshots)
"""

import json
import logging
from typing import Any

from ._protocols import StoreProtocol

logger = logging.getLogger(__name__)

ML_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS ml_pipelines (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    name TEXT NOT NULL,
    task TEXT NOT NULL DEFAULT 'embryo_stage_classification',
    status TEXT DEFAULT 'planned',
    model_config TEXT,
    data_split TEXT,
    training_config TEXT,
    best_run_id TEXT,
    best_accuracy REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
);

CREATE TABLE IF NOT EXISTS ml_training_runs (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT NOT NULL,
    status TEXT DEFAULT 'planned',
    model_config TEXT,
    training_config TEXT,
    data_split TEXT,
    current_epoch INTEGER DEFAULT 0,
    total_epochs INTEGER DEFAULT 0,
    train_loss REAL DEFAULT 0.0,
    val_loss REAL DEFAULT 0.0,
    val_accuracy REAL DEFAULT 0.0,
    best_val_accuracy REAL DEFAULT 0.0,
    model_weights_path TEXT DEFAULT '',
    metrics_path TEXT DEFAULT '',
    peer_instance_id TEXT DEFAULT '',
    started_at TEXT DEFAULT '',
    completed_at TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    FOREIGN KEY (pipeline_id) REFERENCES ml_pipelines(id)
);

CREATE TABLE IF NOT EXISTS ml_data_assessments (
    id TEXT PRIMARY KEY,
    pipeline_id TEXT,
    total_sessions INTEGER DEFAULT 0,
    total_embryos INTEGER DEFAULT 0,
    total_volumes INTEGER DEFAULT 0,
    annotated_embryos INTEGER DEFAULT 0,
    stage_distribution TEXT,
    coverage_gaps TEXT,
    quality_notes TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (pipeline_id) REFERENCES ml_pipelines(id)
);

CREATE INDEX IF NOT EXISTS idx_ml_pipelines_campaign ON ml_pipelines(campaign_id);
CREATE INDEX IF NOT EXISTS idx_ml_pipelines_status ON ml_pipelines(status);
CREATE INDEX IF NOT EXISTS idx_ml_training_runs_pipeline ON ml_training_runs(pipeline_id);
CREATE INDEX IF NOT EXISTS idx_ml_training_runs_status ON ml_training_runs(status);
CREATE INDEX IF NOT EXISTS idx_ml_data_assessments_pipeline ON ml_data_assessments(pipeline_id);
"""


class MlPipelinesMixin(StoreProtocol):
    """ContextStore mixin for ML pipeline management."""

    def _ensure_ml_tables(self):
        """Create ML tables if they don't exist (idempotent)."""
        self._conn.executescript(ML_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # ML Pipelines
    # ------------------------------------------------------------------

    def create_ml_pipeline(
        self,
        campaign_id: str,
        name: str,
        task: str = "embryo_stage_classification",
        model_config: dict | None = None,
        data_split: dict | None = None,
        training_config: dict | None = None,
    ) -> dict[str, Any]:
        """Create a new ML pipeline."""
        self._ensure_ml_tables()
        pipeline_id = self._gen_id()
        now = self._now()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO ml_pipelines
                   (id, campaign_id, name, task, status, model_config, data_split,
                    training_config, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'planned', ?, ?, ?, ?, ?)""",
                (
                    pipeline_id,
                    campaign_id,
                    name,
                    task,
                    json.dumps(model_config) if model_config else None,
                    json.dumps(data_split) if data_split else None,
                    json.dumps(training_config) if training_config else None,
                    now,
                    now,
                ),
            )
        pipeline = self.get_ml_pipeline(pipeline_id)
        assert pipeline is not None
        return pipeline

    def get_ml_pipeline(self, pipeline_id: str) -> dict[str, Any] | None:
        """Get a pipeline by ID."""
        self._ensure_ml_tables()
        row = self._conn.execute(
            "SELECT * FROM ml_pipelines WHERE id = ?", (pipeline_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_pipeline(row)

    def list_ml_pipelines(self, campaign_id: str | None = None) -> list[dict[str, Any]]:
        """List pipelines, optionally filtered by campaign."""
        self._ensure_ml_tables()
        if campaign_id:
            rows = self._conn.execute(
                "SELECT * FROM ml_pipelines WHERE campaign_id = ? ORDER BY created_at DESC",
                (campaign_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM ml_pipelines ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_pipeline(r) for r in rows]

    def update_ml_pipeline(self, pipeline_id: str, **kwargs) -> dict[str, Any] | None:
        """Update pipeline fields."""
        self._ensure_ml_tables()
        allowed = {
            "status",
            "model_config",
            "data_split",
            "training_config",
            "best_run_id",
            "best_accuracy",
            "name",
        }
        updates = []
        values = []
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k in ("model_config", "data_split", "training_config") and isinstance(v, dict):
                v = json.dumps(v)
            updates.append(f"{k} = ?")
            values.append(v)

        if not updates:
            return self.get_ml_pipeline(pipeline_id)

        updates.append("updated_at = ?")
        values.append(self._now())
        values.append(pipeline_id)

        with self._tx() as conn:
            conn.execute(
                f"UPDATE ml_pipelines SET {', '.join(updates)} WHERE id = ?",
                values,
            )
        return self.get_ml_pipeline(pipeline_id)

    def _row_to_pipeline(self, row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "campaign_id": row["campaign_id"],
            "name": row["name"],
            "task": row["task"],
            "status": row["status"],
            "model_config": json.loads(row["model_config"]) if row["model_config"] else None,
            "data_split": json.loads(row["data_split"]) if row["data_split"] else None,
            "training_config": json.loads(row["training_config"])
            if row["training_config"]
            else None,
            "best_run_id": row["best_run_id"],
            "best_accuracy": row["best_accuracy"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------

    def create_training_run(
        self,
        pipeline_id: str,
        model_config: dict | None = None,
        training_config: dict | None = None,
        data_split: dict | None = None,
        peer_instance_id: str = "",
    ) -> dict[str, Any]:
        """Create a new training run."""
        self._ensure_ml_tables()
        run_id = self._gen_id()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO ml_training_runs
                   (id, pipeline_id, status, model_config, training_config,
                    data_split, peer_instance_id)
                   VALUES (?, ?, 'planned', ?, ?, ?, ?)""",
                (
                    run_id,
                    pipeline_id,
                    json.dumps(model_config) if model_config else None,
                    json.dumps(training_config) if training_config else None,
                    json.dumps(data_split) if data_split else None,
                    peer_instance_id,
                ),
            )
        run = self.get_training_run(run_id)
        assert run is not None
        return run

    def get_training_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a training run by ID."""
        self._ensure_ml_tables()
        row = self._conn.execute(
            "SELECT * FROM ml_training_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_run(row)

    def list_training_runs(self, pipeline_id: str) -> list[dict[str, Any]]:
        """List runs for a pipeline."""
        self._ensure_ml_tables()
        rows = self._conn.execute(
            "SELECT * FROM ml_training_runs WHERE pipeline_id = ? ORDER BY rowid",
            (pipeline_id,),
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def update_training_run(self, run_id: str, **kwargs) -> dict[str, Any] | None:
        """Update training run fields."""
        self._ensure_ml_tables()
        allowed = {
            "status",
            "current_epoch",
            "total_epochs",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "best_val_accuracy",
            "model_weights_path",
            "metrics_path",
            "started_at",
            "completed_at",
            "error_message",
        }
        updates = []
        values = []
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            updates.append(f"{k} = ?")
            values.append(v)

        if not updates:
            return self.get_training_run(run_id)

        values.append(run_id)
        with self._tx() as conn:
            conn.execute(
                f"UPDATE ml_training_runs SET {', '.join(updates)} WHERE id = ?",
                values,
            )
        return self.get_training_run(run_id)

    def _row_to_run(self, row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "pipeline_id": row["pipeline_id"],
            "status": row["status"],
            "model_config": json.loads(row["model_config"]) if row["model_config"] else None,
            "training_config": json.loads(row["training_config"])
            if row["training_config"]
            else None,
            "data_split": json.loads(row["data_split"]) if row["data_split"] else None,
            "current_epoch": row["current_epoch"],
            "total_epochs": row["total_epochs"],
            "train_loss": row["train_loss"],
            "val_loss": row["val_loss"],
            "val_accuracy": row["val_accuracy"],
            "best_val_accuracy": row["best_val_accuracy"],
            "model_weights_path": row["model_weights_path"],
            "metrics_path": row["metrics_path"],
            "peer_instance_id": row["peer_instance_id"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
        }

    # ------------------------------------------------------------------
    # Data Assessments
    # ------------------------------------------------------------------

    def save_data_assessment(
        self,
        pipeline_id: str | None = None,
        total_sessions: int = 0,
        total_embryos: int = 0,
        total_volumes: int = 0,
        annotated_embryos: int = 0,
        stage_distribution: dict | None = None,
        coverage_gaps: list | None = None,
        quality_notes: str = "",
    ) -> dict[str, Any]:
        """Save a data assessment snapshot."""
        self._ensure_ml_tables()
        assessment_id = self._gen_id()
        now = self._now()
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO ml_data_assessments
                   (id, pipeline_id, total_sessions, total_embryos, total_volumes,
                    annotated_embryos, stage_distribution, coverage_gaps,
                    quality_notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    assessment_id,
                    pipeline_id,
                    total_sessions,
                    total_embryos,
                    total_volumes,
                    annotated_embryos,
                    json.dumps(stage_distribution) if stage_distribution else None,
                    json.dumps(coverage_gaps) if coverage_gaps else None,
                    quality_notes,
                    now,
                ),
            )
        assessment = self.get_data_assessment(assessment_id)
        assert assessment is not None
        return assessment

    def get_data_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        """Get a data assessment by ID."""
        self._ensure_ml_tables()
        row = self._conn.execute(
            "SELECT * FROM ml_data_assessments WHERE id = ?", (assessment_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "pipeline_id": row["pipeline_id"],
            "total_sessions": row["total_sessions"],
            "total_embryos": row["total_embryos"],
            "total_volumes": row["total_volumes"],
            "annotated_embryos": row["annotated_embryos"],
            "stage_distribution": json.loads(row["stage_distribution"])
            if row["stage_distribution"]
            else None,
            "coverage_gaps": json.loads(row["coverage_gaps"]) if row["coverage_gaps"] else None,
            "quality_notes": row["quality_notes"],
            "created_at": row["created_at"],
        }
