"""
EmbryoDataset - Iterator-based access to embryo imaging data.

Provides streaming access to images per embryo for running perception algorithms.

Example usage:
    dataset = EmbryoDataset()

    # Iterate through embryos in a session
    for embryo in dataset.iter_embryos(session_id="59799c78"):
        print(f"Processing {embryo.embryo_id}: {embryo.num_images} images")

        # Stream images for this embryo
        for img in embryo.iter_images():
            # Run your perception
            result = my_perception(img.image_b64)

            # Store result
            dataset.store_prediction(
                image_uid=img.uid,
                embryo_id=embryo.embryo_id,
                timepoint=img.timepoint,
                predicted_stage=result.stage,
                confidence=result.confidence,
            )
"""

import json
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from .schema import DEFAULT_DB_PATH, get_connection

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """Data for a single image in the dataset."""

    uid: str
    embryo_id: str
    timepoint: int
    timestamp: str

    # Image data (loaded on demand)
    _image_b64: str | None = field(default=None, repr=False)
    _volume_path: str | None = None
    _image_path: str | None = None

    # Ground truth (if available)
    ground_truth_stage: str | None = None

    # Metadata
    shape: tuple[int, int] | None = None
    projection_type: str = "max_z"
    session_id: str | None = None

    # Internal reference to dataset for lazy loading
    _dataset: Optional["EmbryoDataset"] = field(default=None, repr=False)

    @property
    def image_b64(self) -> str | None:
        """Load and return base64 image data (lazy loading)."""
        if self._image_b64 is None and self._dataset:
            self._image_b64 = self._dataset._load_image_b64(self)
        return self._image_b64

    @property
    def volume_path(self) -> str | None:
        """Path to the source volume TIFF."""
        return self._volume_path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without image data)."""
        return {
            "uid": self.uid,
            "embryo_id": self.embryo_id,
            "timepoint": self.timepoint,
            "timestamp": self.timestamp,
            "ground_truth_stage": self.ground_truth_stage,
            "shape": self.shape,
            "projection_type": self.projection_type,
            "session_id": self.session_id,
            "volume_path": self._volume_path,
        }


@dataclass
class DatasetEmbryoEntry:
    """Information about an embryo in the dataset."""

    embryo_id: str
    session_id: str | None
    num_images: int
    num_volumes: int
    timepoint_range: tuple[int, int]  # (min, max)
    has_ground_truth: bool
    ground_truth_stages: list[str] = field(default_factory=list)

    # Internal reference to dataset
    _dataset: Optional["EmbryoDataset"] = field(default=None, repr=False)

    def iter_images(
        self,
        timepoint_range: tuple[int, int] | None = None,
        load_image_data: bool = True,
    ) -> Iterator[ImageData]:
        """
        Iterate through images for this embryo.

        Parameters
        ----------
        timepoint_range : tuple, optional
            (start, end) timepoint range to filter
        load_image_data : bool
            If True, preload base64 image data

        Yields
        ------
        ImageData
            Image data for each timepoint
        """
        if self._dataset:
            yield from self._dataset.iter_images(
                embryo_id=self.embryo_id,
                session_id=self.session_id,
                timepoint_range=timepoint_range,
                load_image_data=load_image_data,
            )


class EmbryoDataset:
    """
    Dataset interface for embryo imaging data.

    Provides iterator-based access to images organized by embryo,
    with support for storing predictions and annotations.

    Supports both the legacy schema (``dataset.db``) and the
    GentlyStore schema (``gently.db``).  Schema is auto-detected
    on first database access.

    Parameters
    ----------
    db_path : Path, optional
        Path to SQLite database
    data_dir : Path, optional
        Root data directory for loading images
    gently_store : GentlyStore, optional
        If provided, queries use the GentlyStore DB and schema.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        data_dir: Path = Path("D:/gently/data"),
        gently_store=None,
    ):
        if gently_store is not None:
            self.db_path = gently_store.db_path
            self.data_dir = gently_store.root
            self._gently_store = gently_store
        else:
            self.db_path = db_path or DEFAULT_DB_PATH
            self.data_dir = data_dir
            self._gently_store = None
        self._conn: sqlite3.Connection | None = None
        self._is_gently_schema: bool | None = None

    @classmethod
    def from_store(cls, store) -> "EmbryoDataset":
        """Create an EmbryoDataset backed by a GentlyStore."""
        return cls(gently_store=store)

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            self._conn = get_connection(self.db_path)
        return self._conn

    @property
    def is_gently_schema(self) -> bool:
        """Detect whether the DB uses the GentlyStore schema."""
        if self._is_gently_schema is None:
            cursor = self.conn.execute("PRAGMA table_info(volumes)")
            columns = {row[1] for row in cursor.fetchall()}
            # GentlyStore schema uses acquired_at; legacy uses uid + timestamp
            self._is_gently_schema = "uid" not in columns
        return self._is_gently_schema

    def _resolve_db_path(self, db_path: str | None) -> str | None:
        """Convert DB-stored file path to absolute path string."""
        if db_path is None:
            return None
        if self.is_gently_schema and not Path(db_path).is_absolute():
            return str(self.data_dir / db_path)
        return db_path

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # =========================================================================
    # Iteration Methods
    # =========================================================================

    def iter_embryos(
        self,
        session_id: str | None = None,
        has_ground_truth: bool | None = None,
        min_images: int = 1,
    ) -> Iterator[DatasetEmbryoEntry]:
        """
        Iterate through embryos in the dataset.

        Parameters
        ----------
        session_id : str, optional
            Filter to specific session
        has_ground_truth : bool, optional
            If True, only embryos with ground truth labels
            If False, only embryos without ground truth
        min_images : int
            Minimum number of images required

        Yields
        ------
        DatasetEmbryoEntry
            Information about each embryo
        """
        # Build query — schema-dependent
        if self.is_gently_schema:
            query = """
                SELECT
                    v.embryo_id,
                    v.session_id,
                    COUNT(*) as num_volumes,
                    (SELECT COUNT(*) FROM projections p
                     WHERE p.embryo_id = v.embryo_id
                       AND p.session_id = v.session_id) as num_images,
                    MIN(v.timepoint) as min_tp,
                    MAX(v.timepoint) as max_tp
                FROM volumes v
                WHERE v.embryo_id IS NOT NULL
            """
        else:
            query = """
                SELECT
                    v.embryo_id,
                    v.session_id,
                    COUNT(DISTINCT v.uid) as num_volumes,
                    COUNT(DISTINCT i.uid) as num_images,
                    MIN(COALESCE(v.timepoint, 0)) as min_tp,
                    MAX(COALESCE(v.timepoint, 0)) as max_tp
                FROM volumes v
                LEFT JOIN images i ON i.embryo_id = v.embryo_id
                    AND i.timepoint = v.timepoint
                WHERE v.embryo_id IS NOT NULL
            """
        params = []

        if session_id:
            query += " AND v.session_id = ?"
            params.append(session_id)

        query += " GROUP BY v.embryo_id, v.session_id"
        query += f" HAVING num_volumes >= {min_images}"
        query += " ORDER BY v.session_id, v.embryo_id"

        for row in self.conn.execute(query, params):
            embryo_id = row[0]
            sess_id = row[1]

            # Check ground truth
            gt_rows = self.conn.execute(
                """
                SELECT stage FROM ground_truth
                WHERE embryo_id = ? AND (session_id = ? OR ? IS NULL)
                ORDER BY start_timepoint
            """,
                (embryo_id, sess_id, sess_id),
            ).fetchall()

            gt_stages = [r[0] for r in gt_rows]
            has_gt = len(gt_stages) > 0

            if has_ground_truth is not None:
                if has_ground_truth and not has_gt:
                    continue
                if not has_ground_truth and has_gt:
                    continue

            yield DatasetEmbryoEntry(
                embryo_id=embryo_id,
                session_id=sess_id,
                num_volumes=row[2],
                num_images=row[3],
                timepoint_range=(row[4], row[5]),
                has_ground_truth=has_gt,
                ground_truth_stages=gt_stages,
                _dataset=self,
            )

    def iter_images(
        self,
        embryo_id: str,
        session_id: str | None = None,
        timepoint_range: tuple[int, int] | None = None,
        load_image_data: bool = True,
    ) -> Iterator[ImageData]:
        """
        Iterate through images for an embryo.

        Parameters
        ----------
        embryo_id : str
            Embryo ID to get images for
        session_id : str, optional
            Session ID filter
        timepoint_range : tuple, optional
            (start, end) timepoint range
        load_image_data : bool
            If True, load base64 image data

        Yields
        ------
        ImageData
            Image data for each timepoint
        """
        # Query volumes (they have the main data) — schema-dependent
        if self.is_gently_schema:
            query = """
                SELECT
                    v.session_id || '/' || v.embryo_id || '/t'
                        || printf('%04d', v.timepoint) as volume_uid,
                    v.embryo_id,
                    v.timepoint,
                    v.acquired_at as timestamp,
                    v.file_path as volume_path,
                    v.session_id,
                    v.shape as shape_json,
                    p.file_path as image_uid,
                    NULL as image_shape
                FROM volumes v
                LEFT JOIN projections p ON p.embryo_id = v.embryo_id
                    AND p.timepoint = v.timepoint
                    AND p.session_id = v.session_id
                WHERE v.embryo_id = ?
            """
        else:
            query = """
                SELECT
                    v.uid as volume_uid,
                    v.embryo_id,
                    v.timepoint,
                    v.timestamp,
                    v.file_path as volume_path,
                    v.session_id,
                    v.shape_json,
                    i.uid as image_uid,
                    i.shape_json as image_shape
                FROM volumes v
                LEFT JOIN images i ON i.embryo_id = v.embryo_id
                    AND i.timepoint = v.timepoint
                WHERE v.embryo_id = ?
            """
        params: list[Any] = [embryo_id]

        if session_id:
            query += " AND v.session_id = ?"
            params.append(session_id)

        if timepoint_range:
            query += " AND v.timepoint >= ? AND v.timepoint <= ?"
            params.extend(timepoint_range)

        if self.is_gently_schema:
            query += " ORDER BY v.acquired_at ASC"
        else:
            query += " ORDER BY v.timestamp ASC"

        # Get ground truth for this embryo
        gt_map = self._get_ground_truth_map(embryo_id, session_id)

        for idx, row in enumerate(self.conn.execute(query, params)):
            timepoint = row[2] or 0

            # Determine ground truth stage at this INDEX (not timepoint)
            # Ground truth start_timepoint is actually the index
            gt_stage = None
            for stage, (start_idx, end_idx) in gt_map.items():
                if start_idx <= idx <= end_idx:
                    gt_stage = stage
                    break

            # Parse shape
            shape = None
            if row[8]:  # image_shape
                try:
                    shape = tuple(json.loads(row[8]))
                except Exception:
                    pass

            img_data = ImageData(
                uid=row[7] or row[0],  # image_uid or volume_uid
                embryo_id=row[1],
                timepoint=timepoint,
                timestamp=row[3],
                _volume_path=self._resolve_db_path(row[4]),
                session_id=row[5],
                shape=shape,
                ground_truth_stage=gt_stage,
                _dataset=self if load_image_data else None,
            )

            yield img_data

    def get_image(
        self,
        embryo_id: str,
        timepoint: int,
        session_id: str | None = None,
    ) -> ImageData | None:
        """Get a single image by embryo and timepoint."""
        for img in self.iter_images(
            embryo_id=embryo_id,
            session_id=session_id,
            timepoint_range=(timepoint, timepoint),
        ):
            return img
        return None

    def get_image_by_index(
        self,
        embryo_id: str,
        index: int,
        session_id: str | None = None,
    ) -> ImageData | None:
        """Get a single image by sequential index (for volumes without timepoints)."""
        for i, img in enumerate(
            self.iter_images(
                embryo_id=embryo_id,
                session_id=session_id,
            )
        ):
            if i == index:
                return img
        return None

    def get_image_by_uid(self, uid: str) -> ImageData | None:
        """Get a single image by its UID.

        For GentlyStore schema (no UIDs), returns None.
        Use ``get_image(embryo_id, timepoint, session_id)`` instead.
        """
        if self.is_gently_schema:
            return None

        # Legacy schema — query the volume directly
        row = self.conn.execute(
            """
            SELECT
                v.uid as volume_uid,
                v.embryo_id,
                v.timepoint,
                v.timestamp,
                v.file_path as volume_path,
                v.session_id,
                v.shape_json,
                i.uid as image_uid,
                i.shape_json as image_shape
            FROM volumes v
            LEFT JOIN images i ON i.embryo_id = v.embryo_id
                AND i.timepoint = v.timepoint
            WHERE v.uid = ? OR i.uid = ?
            LIMIT 1
        """,
            (uid, uid),
        ).fetchone()

        if not row:
            return None

        # Get ground truth
        gt_map = self._get_ground_truth_map(row[1], row[5])
        timepoint = row[2] or 0

        gt_stage = None
        for stage, (start_tp, end_tp) in gt_map.items():
            if start_tp <= timepoint <= end_tp:
                gt_stage = stage
                break

        shape = None
        if row[8]:
            try:
                shape = tuple(json.loads(row[8]))
            except Exception:
                pass

        return ImageData(
            uid=row[7] or row[0],
            embryo_id=row[1],
            timepoint=timepoint,
            timestamp=row[3],
            _volume_path=row[4],
            session_id=row[5],
            shape=shape,
            ground_truth_stage=gt_stage,
            _dataset=self,
        )

    # =========================================================================
    # Ground Truth / Annotation Methods
    # =========================================================================

    def _get_ground_truth_map(
        self,
        embryo_id: str,
        session_id: str | None = None,
    ) -> dict[str, tuple[int, int]]:
        """
        Get ground truth stage → (start_tp, end_tp) mapping.

        Returns dict like: {"early": (0, 42), "bean": (43, 48), ...}

        Uses stored end_timepoint if available, otherwise calculates from
        the start of the next stage.
        """
        query = """
            SELECT stage, start_timepoint, end_timepoint FROM ground_truth
            WHERE embryo_id = ?
        """
        params = [embryo_id]

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY start_timepoint ASC"

        rows = self.conn.execute(query, params).fetchall()

        if not rows:
            return {}

        gt_map = {}
        for i, (stage, start_tp, stored_end_tp) in enumerate(rows):
            # Use stored end_timepoint if available (exclusive, so subtract 1)
            # Otherwise calculate from next stage's start
            if stored_end_tp is not None:
                end_tp = stored_end_tp - 1  # Convert exclusive to inclusive
            elif i + 1 < len(rows):
                end_tp = rows[i + 1][1] - 1
            else:
                end_tp = 999999
            gt_map[stage] = (start_tp, end_tp)

        return gt_map

    def set_ground_truth(
        self,
        session_id: str,
        embryo_id: str,
        stage: str,
        start_timepoint: int,
        end_timepoint: int | None = None,
        annotator: str | None = None,
        notes: str | None = None,
    ):
        """
        Set or update a ground truth annotation.

        Parameters
        ----------
        session_id : str
            Session ID
        embryo_id : str
            Embryo ID
        stage : str
            Stage name (early, bean, comma, etc.)
        start_timepoint : int
            Timepoint when this stage starts
        end_timepoint : int, optional
            Timepoint when this stage ends (exclusive)
        annotator : str, optional
            Who made the annotation
        notes : str, optional
            Additional notes
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO ground_truth
            (session_id, embryo_id, stage, start_timepoint, end_timepoint, annotator, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                embryo_id,
                stage,
                start_timepoint,
                end_timepoint,
                annotator,
                notes,
            ),
        )
        self.conn.commit()
        end_str = f"-{end_timepoint}" if end_timepoint else ""
        logger.info(f"Set ground truth: {embryo_id} {stage} @ t={start_timepoint}{end_str}")

    def delete_ground_truth(
        self,
        session_id: str,
        embryo_id: str,
        stage: str | None = None,
    ):
        """
        Delete ground truth annotation(s).

        Parameters
        ----------
        session_id : str
            Session ID
        embryo_id : str
            Embryo ID
        stage : str, optional
            If provided, delete only this stage. Otherwise delete all.
        """
        if stage:
            self.conn.execute(
                """
                DELETE FROM ground_truth
                WHERE session_id = ? AND embryo_id = ? AND stage = ?
            """,
                (session_id, embryo_id, stage),
            )
        else:
            self.conn.execute(
                """
                DELETE FROM ground_truth
                WHERE session_id = ? AND embryo_id = ?
            """,
                (session_id, embryo_id),
            )
        self.conn.commit()

    def get_ground_truth(
        self,
        session_id: str,
        embryo_id: str,
    ) -> list[dict[str, Any]]:
        """Get all ground truth entries for an embryo."""
        rows = self.conn.execute(
            """
            SELECT stage, start_timepoint, end_timepoint, annotator, notes, created_at
            FROM ground_truth
            WHERE session_id = ? AND embryo_id = ?
            ORDER BY start_timepoint
        """,
            (session_id, embryo_id),
        ).fetchall()

        return [
            {
                "stage": r[0],
                "start_timepoint": r[1],
                "end_timepoint": r[2],
                "annotator": r[3],
                "notes": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    # =========================================================================
    # Prediction Storage Methods
    # =========================================================================

    def create_perception_run(
        self,
        name: str,
        perception_method: str,
        model_name: str | None = None,
        config: dict | None = None,
        description: str | None = None,
        trace_type: str = "perception",
        source: str = "benchmark",
        session_id: str | None = None,
    ) -> int:
        """
        Create a new perception run record.

        Parameters
        ----------
        name : str
            Name for the run
        perception_method : str
            Method identifier (e.g., 'vlm_v1', 'vlm_v3')
        model_name : str, optional
            Model identifier
        config : dict, optional
            Full configuration as dict
        description : str, optional
            Human-readable description
        trace_type : str
            Type of traces ('perception', 'hatching_detector', etc.)
        source : str
            Origin of the run ('live', 'benchmark', 'replay')
        session_id : str, optional
            Link to live experiment session

        Returns
        -------
        int
            Run ID for storing predictions
        """
        config_json = json.dumps(config) if config else None

        if self.is_gently_schema:
            cursor = self.conn.execute(
                """
                INSERT INTO perception_runs
                (session_id, name, perception_method, model_name, config,
                 status, trace_type, source, created_at)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)
            """,
                (
                    session_id,
                    name,
                    perception_method,
                    model_name,
                    config_json,
                    trace_type,
                    source,
                    datetime.now().isoformat(),
                ),
            )
        else:
            cursor = self.conn.execute(
                """
                INSERT INTO perception_runs
                (name, perception_method, model_name, config_json, description, status,
                 trace_type, source, session_id)
                VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?)
            """,
                (
                    name,
                    perception_method,
                    model_name,
                    config_json,
                    description,
                    trace_type,
                    source,
                    session_id,
                ),
            )
        self.conn.commit()
        return cast("int", cursor.lastrowid)

    def store_prediction(
        self,
        run_id: int,
        embryo_id: str,
        timepoint: int,
        predicted_stage: str,
        confidence: float | None = None,
        reasoning: str | None = None,
        image_uid: str | None = None,
        session_id: str | None = None,
        is_transitional: bool = False,
        observed_features: dict | None = None,
        reasoning_trace: dict | None = None,
        execution_time_ms: float | None = None,
        trace_file_path: str | None = None,
    ) -> int:
        """
        Store a perception prediction.

        Returns
        -------
        int
            Prediction ID
        """
        # Look up ground truth
        gt_stage = None
        is_correct = None

        if session_id:
            gt_map = self._get_ground_truth_map(embryo_id, session_id)
            for stage, (start_tp, end_tp) in gt_map.items():
                if start_tp <= timepoint <= end_tp:
                    gt_stage = stage
                    is_correct = 1 if predicted_stage == gt_stage else 0
                    break

        if self.is_gently_schema:
            # GentlyStore schema: single predictions table with JSON blobs
            cursor = self.conn.execute(
                """
                INSERT INTO predictions
                (run_id, session_id, embryo_id, timepoint,
                 predicted_stage, confidence, reasoning, is_transitional,
                 ground_truth_stage, is_correct, execution_time_ms,
                 trace_file, observed_features, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    session_id,
                    embryo_id,
                    timepoint,
                    predicted_stage,
                    confidence,
                    reasoning,
                    1 if is_transitional else 0,
                    gt_stage,
                    is_correct,
                    execution_time_ms,
                    trace_file_path,
                    json.dumps(observed_features) if observed_features else None,
                    datetime.now().isoformat(),
                ),
            )
            prediction_id = cast("int", cursor.lastrowid)
        else:
            # Legacy schema: separate observed_features + reasoning_traces tables
            confidence_level = None
            if confidence is not None:
                if confidence >= 0.8:
                    confidence_level = "HIGH"
                elif confidence >= 0.5:
                    confidence_level = "MEDIUM"
                else:
                    confidence_level = "LOW"

            cursor = self.conn.execute(
                """
                INSERT INTO predictions
                (perception_run_id, image_uid, session_id, embryo_id, timepoint,
                 predicted_stage, confidence, confidence_level, is_transitional,
                 reasoning, ground_truth_stage, is_correct, execution_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    run_id,
                    image_uid,
                    session_id,
                    embryo_id,
                    timepoint,
                    predicted_stage,
                    confidence,
                    confidence_level,
                    1 if is_transitional else 0,
                    reasoning,
                    gt_stage,
                    is_correct,
                    execution_time_ms,
                ),
            )

            prediction_id = cast("int", cursor.lastrowid)

            # Store observed features if provided
            if observed_features:
                self.conn.execute(
                    """
                    INSERT INTO observed_features
                    (prediction_id, shape, curvature, shell_status, body_segments,
                     emergence, movement, texture, features_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        prediction_id,
                        observed_features.get("shape"),
                        observed_features.get("curvature"),
                        observed_features.get("shell_status"),
                        observed_features.get("body_segments"),
                        observed_features.get("emergence"),
                        observed_features.get("movement"),
                        observed_features.get("texture"),
                        json.dumps(observed_features),
                    ),
                )

            # Store reasoning trace if provided
            if reasoning_trace or trace_file_path:
                self.conn.execute(
                    """
                    INSERT INTO reasoning_traces
                    (prediction_id, contrastive_reasoning, steps_json,
                     tool_calls_json, tools_used_json, total_tool_calls, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        prediction_id,
                        reasoning_trace.get("contrastive_reasoning") if reasoning_trace else None,
                        json.dumps(reasoning_trace.get("steps", [])) if reasoning_trace else None,
                        json.dumps(reasoning_trace.get("tool_calls", []))
                        if reasoning_trace
                        else None,
                        json.dumps(reasoning_trace.get("tools_used", []))
                        if reasoning_trace
                        else None,
                        reasoning_trace.get("total_tool_calls", 0) if reasoning_trace else 0,
                        trace_file_path,
                    ),
                )

        self.conn.commit()
        return prediction_id

    def complete_perception_run(
        self,
        run_id: int,
        status: str = "completed",
        error_message: str | None = None,
    ):
        """Mark a perception run as completed."""
        now = datetime.now().isoformat()

        if self.is_gently_schema:
            self.conn.execute(
                """
                UPDATE perception_runs SET
                    status = ?, completed_at = ?, error_message = ?
                WHERE run_id = ?
            """,
                (status, now, error_message, run_id),
            )
        else:
            self.conn.execute(
                """
                UPDATE perception_runs SET
                    status = ?,
                    completed_at = ?,
                    error_message = ?,
                    total_samples = (
                        SELECT COUNT(*) FROM predictions WHERE perception_run_id = ?
                    )
                WHERE id = ?
            """,
                (status, now, error_message, run_id, run_id),
            )
        self.conn.commit()

    # =========================================================================
    # Metrics Methods
    # =========================================================================

    def compute_run_metrics(self, run_id: int) -> dict[str, Any]:
        """
        Compute accuracy metrics for a perception run.

        Returns
        -------
        dict
            Metrics including accuracy, confusion matrix, per-stage stats
        """
        # Get predictions with ground truth
        run_col = "run_id" if self.is_gently_schema else "perception_run_id"
        rows = self.conn.execute(
            f"""
            SELECT predicted_stage, ground_truth_stage, is_correct, confidence
            FROM predictions
            WHERE {run_col} = ? AND ground_truth_stage IS NOT NULL
        """,
            (run_id,),
        ).fetchall()

        if not rows:
            return {"error": "No predictions with ground truth"}

        total = len(rows)
        correct = sum(1 for r in rows if r[2] == 1)
        accuracy = correct / total if total > 0 else 0

        # Build confusion matrix
        confusion: dict[str, Any] = {}
        for pred_stage, gt_stage, _, _ in rows:
            if gt_stage not in confusion:
                confusion[gt_stage] = {}
            if pred_stage not in confusion[gt_stage]:
                confusion[gt_stage][pred_stage] = 0
            confusion[gt_stage][pred_stage] += 1

        # Per-stage metrics
        stages = set(r[0] for r in rows) | set(r[1] for r in rows)
        per_stage = {}

        for stage in stages:
            tp = sum(1 for r in rows if r[0] == stage and r[1] == stage)
            fp = sum(1 for r in rows if r[0] == stage and r[1] != stage)
            fn = sum(1 for r in rows if r[0] != stage and r[1] == stage)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            per_stage[stage] = {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "support": sum(1 for r in rows if r[1] == stage),
            }

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 3),
            "confusion_matrix": confusion,
            "per_stage": per_stage,
        }

    # =========================================================================
    # Image Loading
    # =========================================================================

    def _load_image_b64(self, img: ImageData) -> str | None:
        """
        Load base64 image data for an ImageData object.

        Tries multiple sources:
        1. Generate projection from volume TIFF
        2. Load from stored image file
        """
        # Try loading from volume
        if img._volume_path and Path(img._volume_path).exists():
            try:
                return self._load_projection_from_volume(img._volume_path)
            except Exception as e:
                logger.warning(f"Failed to load volume {img._volume_path}: {e}")

        return None

    def _load_projection_from_volume(self, volume_path: str) -> str:
        """Load volume and create max projection as base64 JPEG."""

        import tifffile

        # Load volume
        volume = tifffile.imread(volume_path)
        volume = np.squeeze(volume)

        # Handle different shapes
        if volume.ndim == 4:
            # [views, z, y, x] - take first view
            volume = volume[0]

        if volume.ndim == 3:
            z_depth, height, width = volume.shape
            # Extract View A (left half) if dual-view format
            if width > height * 1.5:
                volume = volume[:, :, : width // 2]
            # Max projection
            projection = np.max(volume, axis=0)
        else:
            projection = volume
            # Extract View A if 2D dual-view
            height, width = projection.shape
            if width > height * 1.5:
                projection = projection[:, : width // 2]

        # Normalize and encode
        from gently.core.imaging import image_to_base64, normalize_to_uint8

        projection = normalize_to_uint8(projection, method="percentile", p_low=1, p_high=99.5)
        return image_to_base64(projection, format="JPEG", quality=85, max_dimension=1024)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_sessions(self) -> list[dict[str, Any]]:
        """Get list of sessions with summary stats."""
        rows = self.conn.execute("""
            SELECT
                s.session_id,
                s.name,
                s.created_at,
                COUNT(DISTINCT e.embryo_id) as embryo_count,
                (SELECT COUNT(*) FROM ground_truth g
                 WHERE g.session_id = s.session_id) as gt_count,
                (SELECT COUNT(*) FROM volumes v
                 WHERE v.session_id = s.session_id) as volume_count
            FROM sessions s
            LEFT JOIN embryos e ON s.session_id = e.session_id
            GROUP BY s.session_id
            ORDER BY s.created_at DESC
        """).fetchall()

        return [
            {
                "session_id": r[0],
                "name": r[1],
                "created_at": r[2],
                "embryo_count": r[3],
                "has_ground_truth": r[4] > 0,
                "ground_truth_count": r[4],
                "volume_count": r[5],
            }
            for r in rows
        ]

    def get_perception_runs(self) -> list[dict[str, Any]]:
        """Get list of perception runs with metrics."""
        if self.is_gently_schema:
            # GentlyStore schema — inline the accuracy view
            rows = self.conn.execute("""
                SELECT
                    pr.run_id,
                    pr.name,
                    pr.perception_method,
                    pr.model_name,
                    COUNT(p.prediction_id) as total_predictions,
                    SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as correct,
                    SUM(CASE WHEN p.is_correct = 0 THEN 1 ELSE 0 END) as incorrect,
                    SUM(CASE WHEN p.is_correct IS NULL THEN 1 ELSE 0 END) as no_ground_truth,
                    ROUND(100.0 * SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) /
                          NULLIF(SUM(CASE WHEN p.is_correct IS NOT NULL THEN 1 ELSE 0 END), 0), 2)
                        as accuracy_pct
                FROM perception_runs pr
                LEFT JOIN predictions p ON pr.run_id = p.run_id
                GROUP BY pr.run_id
                ORDER BY pr.run_id DESC
            """).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT * FROM v_run_accuracy
                ORDER BY run_id DESC
            """).fetchall()

        return [dict(r) for r in rows]

    # =========================================================================
    # Trace Query Methods (for trace persistence system)
    # =========================================================================

    def get_traces_for_image(
        self,
        embryo_id: str,
        timepoint: int,
        session_id: str | None = None,
        trace_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get all traces for a specific image (embryo + timepoint).

        Supports multiple trace versions per image through different runs.

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Timepoint number
        session_id : str, optional
            Filter to specific session
        trace_type : str, optional
            Filter to specific trace type ('perception', 'hatching_detector', etc.)

        Returns
        -------
        list of dict
            Traces with prediction and run metadata
        """
        if self.is_gently_schema:
            query = """
                SELECT
                    p.prediction_id,
                    p.predicted_stage,
                    p.confidence,
                    p.reasoning,
                    p.created_at,
                    pr.run_id,
                    pr.name as run_name,
                    pr.trace_type,
                    pr.source,
                    pr.session_id as run_session_id,
                    pr.perception_method,
                    p.trace_file,
                    NULL as contrastive_reasoning,
                    p.observed_features as steps_json,
                    NULL as total_tool_calls
                FROM predictions p
                JOIN perception_runs pr ON p.run_id = pr.run_id
                WHERE p.embryo_id = ? AND p.timepoint = ?
            """
        else:
            query = """
                SELECT
                    p.id as prediction_id,
                    p.predicted_stage,
                    p.confidence,
                    p.reasoning,
                    p.timestamp,
                    pr.id as run_id,
                    pr.name as run_name,
                    pr.trace_type,
                    pr.source,
                    pr.session_id as run_session_id,
                    pr.perception_method,
                    rt.file_path,
                    rt.contrastive_reasoning,
                    rt.steps_json,
                    rt.total_tool_calls
                FROM predictions p
                JOIN perception_runs pr ON p.perception_run_id = pr.id
                LEFT JOIN reasoning_traces rt ON p.id = rt.prediction_id
                WHERE p.embryo_id = ? AND p.timepoint = ?
            """
        params = [embryo_id, timepoint]

        if session_id:
            query += " AND pr.session_id = ?"
            params.append(session_id)

        if trace_type:
            query += " AND pr.trace_type = ?"
            params.append(trace_type)

        if self.is_gently_schema:
            query += " ORDER BY p.created_at DESC"
        else:
            query += " ORDER BY p.timestamp DESC"

        rows = self.conn.execute(query, params).fetchall()

        return [
            {
                "prediction_id": r[0],
                "predicted_stage": r[1],
                "confidence": r[2],
                "reasoning": r[3],
                "timestamp": r[4],
                "run_id": r[5],
                "run_name": r[6],
                "trace_type": r[7],
                "source": r[8],
                "run_session_id": r[9],
                "perception_method": r[10],
                "file_path": r[11],
                "contrastive_reasoning": r[12],
                "steps_json": r[13],
                "total_tool_calls": r[14],
            }
            for r in rows
        ]

    def get_runs_for_session(
        self,
        session_id: str,
        trace_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get all perception runs for a session.

        Parameters
        ----------
        session_id : str
            Session identifier
        trace_type : str, optional
            Filter to specific trace type

        Returns
        -------
        list of dict
            Runs with metadata and prediction counts
        """
        if self.is_gently_schema:
            query = """
                SELECT
                    pr.run_id,
                    pr.name,
                    NULL as description,
                    pr.perception_method,
                    pr.model_name,
                    pr.trace_type,
                    pr.source,
                    pr.status,
                    pr.created_at,
                    pr.completed_at,
                    COUNT(p.prediction_id) as prediction_count
                FROM perception_runs pr
                LEFT JOIN predictions p ON pr.run_id = p.run_id
                WHERE pr.session_id = ?
            """
        else:
            query = """
                SELECT
                    pr.id,
                    pr.name,
                    pr.description,
                    pr.perception_method,
                    pr.model_name,
                    pr.trace_type,
                    pr.source,
                    pr.status,
                    pr.created_at,
                    pr.completed_at,
                    COUNT(p.id) as prediction_count
                FROM perception_runs pr
                LEFT JOIN predictions p ON pr.id = p.perception_run_id
                WHERE pr.session_id = ?
            """
        params = [session_id]

        if trace_type:
            query += " AND pr.trace_type = ?"
            params.append(trace_type)

        if self.is_gently_schema:
            query += " GROUP BY pr.run_id ORDER BY pr.created_at DESC"
        else:
            query += " GROUP BY pr.id ORDER BY pr.created_at DESC"

        rows = self.conn.execute(query, params).fetchall()

        return [
            {
                "id": r[0],
                "name": r[1],
                "description": r[2],
                "perception_method": r[3],
                "model_name": r[4],
                "trace_type": r[5],
                "source": r[6],
                "status": r[7],
                "created_at": r[8],
                "completed_at": r[9],
                "prediction_count": r[10],
            }
            for r in rows
        ]

    def get_run_predictions(
        self,
        run_id: int,
        embryo_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get all predictions for a run, optionally filtered by embryo.

        Parameters
        ----------
        run_id : int
            Perception run ID
        embryo_id : str, optional
            Filter to specific embryo

        Returns
        -------
        list of dict
            Predictions with trace info
        """
        if self.is_gently_schema:
            query = """
                SELECT
                    p.prediction_id,
                    p.embryo_id,
                    p.timepoint,
                    p.predicted_stage,
                    p.confidence,
                    p.reasoning,
                    p.ground_truth_stage,
                    p.is_correct,
                    p.created_at,
                    p.trace_file
                FROM predictions p
                WHERE p.run_id = ?
            """
        else:
            query = """
                SELECT
                    p.id,
                    p.embryo_id,
                    p.timepoint,
                    p.predicted_stage,
                    p.confidence,
                    p.reasoning,
                    p.ground_truth_stage,
                    p.is_correct,
                    p.timestamp,
                    rt.file_path
                FROM predictions p
                LEFT JOIN reasoning_traces rt ON p.id = rt.prediction_id
                WHERE p.perception_run_id = ?
            """
        params: list[Any] = [run_id]

        if embryo_id:
            query += " AND p.embryo_id = ?"
            params.append(embryo_id)

        query += " ORDER BY p.embryo_id, p.timepoint"

        rows = self.conn.execute(query, params).fetchall()

        return [
            {
                "id": r[0],
                "embryo_id": r[1],
                "timepoint": r[2],
                "predicted_stage": r[3],
                "confidence": r[4],
                "reasoning": r[5],
                "ground_truth_stage": r[6],
                "is_correct": r[7],
                "timestamp": r[8],
                "file_path": r[9],
            }
            for r in rows
        ]

    # =========================================================================
    # Cross-Session UID Methods
    # =========================================================================

    def get_embryo_by_uid(self, uid: str) -> list[dict[str, Any]]:
        """
        Get all instances of an embryo across sessions by its UID.

        Parameters
        ----------
        uid : str
            Global unique identifier for the embryo

        Returns
        -------
        list of dict
            All embryo instances matching this UID across different sessions
        """
        if self.is_gently_schema:
            rows = self.conn.execute(
                """
                SELECT
                    e.embryo_id,
                    e.session_id,
                    e.embryo_uid,
                    e.nickname,
                    NULL as user_label,
                    e.position_x as stage_position_x,
                    e.position_y as stage_position_y,
                    e.calibration as calibration_json,
                    e.created_at,
                    s.name as session_name,
                    s.created_at as session_created_at,
                    (SELECT COUNT(*) FROM volumes v
                     WHERE v.embryo_id = e.embryo_id
                       AND v.session_id = e.session_id) as volume_count,
                    (SELECT COUNT(*) FROM projections p
                     WHERE p.embryo_id = e.embryo_id
                       AND p.session_id = e.session_id) as image_count
                FROM embryos e
                LEFT JOIN sessions s ON e.session_id = s.session_id
                WHERE e.embryo_uid = ?
                ORDER BY e.created_at ASC
            """,
                (uid,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT
                    e.embryo_id,
                    e.session_id,
                    e.embryo_uid,
                    e.nickname,
                    e.user_label,
                    e.stage_position_x,
                    e.stage_position_y,
                    e.calibration_json,
                    e.created_at,
                    s.name as session_name,
                    s.created_at as session_created_at,
                    (SELECT COUNT(*) FROM volumes v
                     WHERE v.embryo_uid = e.embryo_uid
                     AND v.session_id = e.session_id) as volume_count,
                    (SELECT COUNT(*) FROM images i
                     WHERE i.embryo_uid = e.embryo_uid
                     AND i.session_id = e.session_id) as image_count
                FROM embryos e
                LEFT JOIN sessions s ON e.session_id = s.session_id
                WHERE e.embryo_uid = ?
                ORDER BY e.created_at ASC
            """,
                (uid,),
            ).fetchall()

        return [
            {
                "embryo_id": r[0],
                "session_id": r[1],
                "embryo_uid": r[2],
                "nickname": r[3],
                "user_label": r[4],
                "stage_position_x": r[5],
                "stage_position_y": r[6],
                "calibration_json": r[7],
                "created_at": r[8],
                "session_name": r[9],
                "session_created_at": r[10],
                "volume_count": r[11],
                "image_count": r[12],
            }
            for r in rows
        ]

    def iter_images_by_uid(
        self,
        uid: str,
        load_image_data: bool = True,
    ) -> Iterator[ImageData]:
        """
        Iterate through all images for an embryo across all sessions.

        Parameters
        ----------
        uid : str
            Global unique identifier for the embryo
        load_image_data : bool
            If True, load base64 image data

        Yields
        ------
        ImageData
            Image data for each timepoint across all sessions
        """
        # Query volumes with this embryo UID across all sessions
        if self.is_gently_schema:
            query = """
                SELECT
                    v.session_id || '/' || v.embryo_id || '/t'
                        || printf('%04d', v.timepoint) as volume_uid,
                    v.embryo_id,
                    v.timepoint,
                    v.acquired_at as timestamp,
                    v.file_path as volume_path,
                    v.session_id,
                    v.shape as shape_json,
                    p.file_path as image_uid,
                    NULL as image_shape
                FROM volumes v
                JOIN embryos e ON v.embryo_id = e.embryo_id
                    AND v.session_id = e.session_id
                LEFT JOIN projections p ON p.embryo_id = v.embryo_id
                    AND p.timepoint = v.timepoint
                    AND p.session_id = v.session_id
                WHERE e.embryo_uid = ?
                ORDER BY v.session_id, v.acquired_at ASC
            """
        else:
            query = """
                SELECT
                    v.uid as volume_uid,
                    v.embryo_id,
                    v.timepoint,
                    v.timestamp,
                    v.file_path as volume_path,
                    v.session_id,
                    v.shape_json,
                    i.uid as image_uid,
                    i.shape_json as image_shape
                FROM volumes v
                LEFT JOIN images i ON i.embryo_uid = v.embryo_uid
                    AND i.timepoint = v.timepoint
                    AND i.session_id = v.session_id
                WHERE v.embryo_uid = ?
                ORDER BY v.session_id, v.timestamp ASC
            """

        for row in self.conn.execute(query, (uid,)):
            timepoint = row[2] or 0
            embryo_id = row[1]
            session_id = row[5]

            # Get ground truth for this embryo instance
            gt_map = self._get_ground_truth_map(embryo_id, session_id)

            gt_stage = None
            for stage, (start_tp, end_tp) in gt_map.items():
                if start_tp <= timepoint <= end_tp:
                    gt_stage = stage
                    break

            shape = None
            if row[8]:
                try:
                    shape = tuple(json.loads(row[8]))
                except Exception:
                    pass

            img_data = ImageData(
                uid=row[7] or row[0],
                embryo_id=embryo_id,
                timepoint=timepoint,
                timestamp=row[3],
                _volume_path=self._resolve_db_path(row[4]),
                session_id=session_id,
                shape=shape,
                ground_truth_stage=gt_stage,
                _dataset=self if load_image_data else None,
            )

            yield img_data

    def get_embryos_with_multiple_sessions(self) -> list[dict[str, Any]]:
        """
        Get embryos that appear in multiple sessions (imported).

        Returns
        -------
        list of dict
            Embryo UIDs with session counts and details
        """
        if self.is_gently_schema:
            rows = self.conn.execute("""
                SELECT
                    embryo_uid,
                    COUNT(DISTINCT session_id) as session_count,
                    GROUP_CONCAT(DISTINCT session_id) as session_ids,
                    MIN(created_at) as first_seen,
                    MAX(created_at) as last_seen,
                    (SELECT COUNT(*) FROM volumes v
                     JOIN embryos e2 ON v.embryo_id = e2.embryo_id
                         AND v.session_id = e2.session_id
                     WHERE e2.embryo_uid = e.embryo_uid) as total_volumes,
                    (SELECT COUNT(*) FROM ground_truth g
                     JOIN embryos e3 ON g.session_id = e3.session_id
                         AND g.embryo_id = e3.embryo_id
                     WHERE e3.embryo_uid = e.embryo_uid) as gt_count
                FROM embryos e
                WHERE embryo_uid IS NOT NULL
                GROUP BY embryo_uid
                HAVING COUNT(DISTINCT session_id) > 1
                ORDER BY session_count DESC, first_seen ASC
            """).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT
                    embryo_uid,
                    COUNT(DISTINCT session_id) as session_count,
                    GROUP_CONCAT(DISTINCT session_id) as session_ids,
                    MIN(created_at) as first_seen,
                    MAX(created_at) as last_seen,
                    (SELECT SUM(cnt) FROM (
                        SELECT COUNT(*) as cnt FROM volumes v
                        WHERE v.embryo_uid = e.embryo_uid
                    )) as total_volumes,
                    (SELECT COUNT(*) FROM ground_truth g
                     JOIN embryos e2 ON g.session_id = e2.session_id AND g.embryo_id = e2.embryo_id
                     WHERE e2.embryo_uid = e.embryo_uid) as gt_count
                FROM embryos e
                WHERE embryo_uid IS NOT NULL
                GROUP BY embryo_uid
                HAVING COUNT(DISTINCT session_id) > 1
                ORDER BY session_count DESC, first_seen ASC
            """).fetchall()

        return [
            {
                "embryo_uid": r[0],
                "session_count": r[1],
                "session_ids": r[2].split(",") if r[2] else [],
                "first_seen": r[3],
                "last_seen": r[4],
                "total_volumes": r[5],
                "has_ground_truth": r[6] > 0 if r[6] else False,
            }
            for r in rows
        ]

    def get_embryo_timeline_by_uid(self, uid: str) -> dict[str, Any]:
        """
        Get complete cross-session timeline for an embryo.

        Parameters
        ----------
        uid : str
            Global unique identifier for the embryo

        Returns
        -------
        dict
            Timeline with sessions and image counts per session
        """
        # Get all instances
        instances = self.get_embryo_by_uid(uid)

        if not instances:
            return {"error": "Embryo UID not found", "embryo_uid": uid}

        # Build timeline per session
        timeline = []
        for instance in instances:
            session_id = instance["session_id"]

            # Get timepoint range and image count for this session
            if self.is_gently_schema:
                stats = self.conn.execute(
                    """
                    SELECT
                        MIN(v.acquired_at) as first_timestamp,
                        MAX(v.acquired_at) as last_timestamp,
                        COUNT(*) as volume_count,
                        MIN(v.timepoint) as min_timepoint,
                        MAX(v.timepoint) as max_timepoint
                    FROM volumes v
                    JOIN embryos e ON v.embryo_id = e.embryo_id
                        AND v.session_id = e.session_id
                    WHERE e.embryo_uid = ? AND v.session_id = ?
                """,
                    (uid, session_id),
                ).fetchone()
            else:
                stats = self.conn.execute(
                    """
                    SELECT
                        MIN(v.timestamp) as first_timestamp,
                        MAX(v.timestamp) as last_timestamp,
                        COUNT(DISTINCT v.uid) as volume_count,
                        MIN(v.timepoint) as min_timepoint,
                        MAX(v.timepoint) as max_timepoint
                    FROM volumes v
                    WHERE v.embryo_uid = ? AND v.session_id = ?
                """,
                    (uid, session_id),
                ).fetchone()

            # Get ground truth stages for this session
            gt_rows = self.conn.execute(
                """
                SELECT stage, start_timepoint FROM ground_truth
                WHERE embryo_id = ? AND session_id = ?
                ORDER BY start_timepoint
            """,
                (instance["embryo_id"], session_id),
            ).fetchall()

            timeline.append(
                {
                    "session_id": session_id,
                    "session_name": instance["session_name"],
                    "embryo_id": instance["embryo_id"],
                    "first_timestamp": stats[0] if stats else None,
                    "last_timestamp": stats[1] if stats else None,
                    "volume_count": stats[2] if stats else 0,
                    "timepoint_range": (stats[3], stats[4]) if stats else (None, None),
                    "ground_truth_stages": [
                        {"stage": gt[0], "start_timepoint": gt[1]} for gt in gt_rows
                    ],
                }
            )

        return {
            "embryo_uid": uid,
            "total_sessions": len(instances),
            "total_volumes": sum(t["volume_count"] for t in timeline),
            "timeline": timeline,
        }
