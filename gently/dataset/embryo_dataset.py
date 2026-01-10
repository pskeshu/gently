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

import base64
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any, Tuple

import numpy as np

from .schema import get_connection, DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """Data for a single image in the dataset."""
    uid: str
    embryo_id: str
    timepoint: int
    timestamp: str

    # Image data (loaded on demand)
    _image_b64: Optional[str] = field(default=None, repr=False)
    _volume_path: Optional[str] = None
    _image_path: Optional[str] = None

    # Ground truth (if available)
    ground_truth_stage: Optional[str] = None

    # Metadata
    shape: Optional[Tuple[int, int]] = None
    projection_type: str = "max_z"
    session_id: Optional[str] = None

    # Internal reference to dataset for lazy loading
    _dataset: Optional["EmbryoDataset"] = field(default=None, repr=False)

    @property
    def image_b64(self) -> Optional[str]:
        """Load and return base64 image data (lazy loading)."""
        if self._image_b64 is None and self._dataset:
            self._image_b64 = self._dataset._load_image_b64(self)
        return self._image_b64

    @property
    def volume_path(self) -> Optional[str]:
        """Path to the source volume TIFF."""
        return self._volume_path

    def to_dict(self) -> Dict[str, Any]:
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
class EmbryoInfo:
    """Information about an embryo in the dataset."""
    embryo_id: str
    session_id: Optional[str]
    num_images: int
    num_volumes: int
    timepoint_range: Tuple[int, int]  # (min, max)
    has_ground_truth: bool
    ground_truth_stages: List[str] = field(default_factory=list)

    # Internal reference to dataset
    _dataset: Optional["EmbryoDataset"] = field(default=None, repr=False)

    def iter_images(
        self,
        timepoint_range: Optional[Tuple[int, int]] = None,
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

    Parameters
    ----------
    db_path : Path, optional
        Path to SQLite database
    data_dir : Path, optional
        Root data directory for loading images
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        data_dir: Path = Path("D:/gently/data"),
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.data_dir = data_dir
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            self._conn = get_connection(self.db_path)
        return self._conn

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
        session_id: Optional[str] = None,
        has_ground_truth: Optional[bool] = None,
        min_images: int = 1,
    ) -> Iterator[EmbryoInfo]:
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
        EmbryoInfo
            Information about each embryo
        """
        # Build query
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
            gt_rows = self.conn.execute("""
                SELECT stage FROM ground_truth
                WHERE embryo_id = ? AND (session_id = ? OR ? IS NULL)
                ORDER BY start_timepoint
            """, (embryo_id, sess_id, sess_id)).fetchall()

            gt_stages = [r[0] for r in gt_rows]
            has_gt = len(gt_stages) > 0

            if has_ground_truth is not None:
                if has_ground_truth and not has_gt:
                    continue
                if not has_ground_truth and has_gt:
                    continue

            yield EmbryoInfo(
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
        session_id: Optional[str] = None,
        timepoint_range: Optional[Tuple[int, int]] = None,
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
        # Query volumes (they have the main data)
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
        params = [embryo_id]

        if session_id:
            query += " AND v.session_id = ?"
            params.append(session_id)

        if timepoint_range:
            query += " AND v.timepoint >= ? AND v.timepoint <= ?"
            params.extend(timepoint_range)

        query += " ORDER BY v.timestamp ASC"  # Order by timestamp for consistent indexing

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
                except:
                    pass

            img_data = ImageData(
                uid=row[7] or row[0],  # image_uid or volume_uid
                embryo_id=row[1],
                timepoint=timepoint,
                timestamp=row[3],
                _volume_path=row[4],
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
        session_id: Optional[str] = None,
    ) -> Optional[ImageData]:
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
        session_id: Optional[str] = None,
    ) -> Optional[ImageData]:
        """Get a single image by sequential index (for volumes without timepoints)."""
        for i, img in enumerate(self.iter_images(
            embryo_id=embryo_id,
            session_id=session_id,
        )):
            if i == index:
                return img
        return None

    def get_image_by_uid(self, uid: str) -> Optional[ImageData]:
        """Get a single image by its UID."""
        # Query the volume directly
        row = self.conn.execute("""
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
        """, (uid, uid)).fetchone()

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
            except:
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
        session_id: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Get ground truth stage → (start_tp, end_tp) mapping.

        Returns dict like: {"early": (0, 42), "bean": (43, 48), ...}
        """
        query = """
            SELECT stage, start_timepoint FROM ground_truth
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
        for i, (stage, start_tp) in enumerate(rows):
            # End timepoint is start of next stage - 1, or infinity
            if i + 1 < len(rows):
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
        annotator: Optional[str] = None,
        notes: Optional[str] = None,
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
        annotator : str, optional
            Who made the annotation
        notes : str, optional
            Additional notes
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO ground_truth
            (session_id, embryo_id, stage, start_timepoint, annotator, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, embryo_id, stage, start_timepoint, annotator, notes))
        self.conn.commit()
        logger.info(f"Set ground truth: {embryo_id} {stage} @ t={start_timepoint}")

    def delete_ground_truth(
        self,
        session_id: str,
        embryo_id: str,
        stage: Optional[str] = None,
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
            self.conn.execute("""
                DELETE FROM ground_truth
                WHERE session_id = ? AND embryo_id = ? AND stage = ?
            """, (session_id, embryo_id, stage))
        else:
            self.conn.execute("""
                DELETE FROM ground_truth
                WHERE session_id = ? AND embryo_id = ?
            """, (session_id, embryo_id))
        self.conn.commit()

    def get_ground_truth(
        self,
        session_id: str,
        embryo_id: str,
    ) -> List[Dict[str, Any]]:
        """Get all ground truth entries for an embryo."""
        rows = self.conn.execute("""
            SELECT stage, start_timepoint, annotator, notes, created_at
            FROM ground_truth
            WHERE session_id = ? AND embryo_id = ?
            ORDER BY start_timepoint
        """, (session_id, embryo_id)).fetchall()

        return [
            {
                "stage": r[0],
                "start_timepoint": r[1],
                "annotator": r[2],
                "notes": r[3],
                "created_at": r[4],
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
        model_name: Optional[str] = None,
        config: Optional[Dict] = None,
        description: Optional[str] = None,
    ) -> int:
        """
        Create a new perception run record.

        Returns
        -------
        int
            Run ID for storing predictions
        """
        cursor = self.conn.execute("""
            INSERT INTO perception_runs
            (name, perception_method, model_name, config_json, description, status)
            VALUES (?, ?, ?, ?, ?, 'running')
        """, (
            name,
            perception_method,
            model_name,
            json.dumps(config) if config else None,
            description,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def store_prediction(
        self,
        run_id: int,
        embryo_id: str,
        timepoint: int,
        predicted_stage: str,
        confidence: Optional[float] = None,
        reasoning: Optional[str] = None,
        image_uid: Optional[str] = None,
        session_id: Optional[str] = None,
        is_transitional: bool = False,
        observed_features: Optional[Dict] = None,
        reasoning_trace: Optional[Dict] = None,
        execution_time_ms: Optional[float] = None,
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

        # Determine confidence level
        confidence_level = None
        if confidence is not None:
            if confidence >= 0.8:
                confidence_level = "HIGH"
            elif confidence >= 0.5:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"

        cursor = self.conn.execute("""
            INSERT INTO predictions
            (perception_run_id, image_uid, session_id, embryo_id, timepoint,
             predicted_stage, confidence, confidence_level, is_transitional,
             reasoning, ground_truth_stage, is_correct, execution_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
        ))

        prediction_id = cursor.lastrowid

        # Store observed features if provided
        if observed_features:
            self.conn.execute("""
                INSERT INTO observed_features
                (prediction_id, shape, curvature, shell_status, body_segments,
                 emergence, movement, texture, features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                observed_features.get("shape"),
                observed_features.get("curvature"),
                observed_features.get("shell_status"),
                observed_features.get("body_segments"),
                observed_features.get("emergence"),
                observed_features.get("movement"),
                observed_features.get("texture"),
                json.dumps(observed_features),
            ))

        # Store reasoning trace if provided
        if reasoning_trace:
            self.conn.execute("""
                INSERT INTO reasoning_traces
                (prediction_id, contrastive_reasoning, steps_json,
                 tool_calls_json, tools_used_json, total_tool_calls)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                reasoning_trace.get("contrastive_reasoning"),
                json.dumps(reasoning_trace.get("steps", [])),
                json.dumps(reasoning_trace.get("tool_calls", [])),
                json.dumps(reasoning_trace.get("tools_used", [])),
                reasoning_trace.get("total_tool_calls", 0),
            ))

        self.conn.commit()
        return prediction_id

    def complete_perception_run(
        self,
        run_id: int,
        status: str = "completed",
        error_message: Optional[str] = None,
    ):
        """Mark a perception run as completed."""
        self.conn.execute("""
            UPDATE perception_runs SET
                status = ?,
                completed_at = ?,
                error_message = ?,
                total_samples = (
                    SELECT COUNT(*) FROM predictions WHERE perception_run_id = ?
                )
            WHERE id = ?
        """, (status, datetime.now().isoformat(), error_message, run_id, run_id))
        self.conn.commit()

    # =========================================================================
    # Metrics Methods
    # =========================================================================

    def compute_run_metrics(self, run_id: int) -> Dict[str, Any]:
        """
        Compute accuracy metrics for a perception run.

        Returns
        -------
        dict
            Metrics including accuracy, confusion matrix, per-stage stats
        """
        # Get predictions with ground truth
        rows = self.conn.execute("""
            SELECT predicted_stage, ground_truth_stage, is_correct, confidence
            FROM predictions
            WHERE perception_run_id = ? AND ground_truth_stage IS NOT NULL
        """, (run_id,)).fetchall()

        if not rows:
            return {"error": "No predictions with ground truth"}

        total = len(rows)
        correct = sum(1 for r in rows if r[2] == 1)
        accuracy = correct / total if total > 0 else 0

        # Build confusion matrix
        confusion = {}
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

    def _load_image_b64(self, img: ImageData) -> Optional[str]:
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
        from PIL import Image
        import io

        # Load volume
        volume = tifffile.imread(volume_path)

        # Handle different shapes
        if volume.ndim == 4:
            # [views, z, y, x] - take first view
            volume = volume[0]

        if volume.ndim == 3:
            # [z, y, x] - max projection
            projection = np.max(volume, axis=0)
        else:
            projection = volume

        # Normalize to 8-bit
        projection = projection.astype(np.float32)
        p_min, p_max = np.percentile(projection, [1, 99.5])
        projection = np.clip((projection - p_min) / (p_max - p_min + 1e-6) * 255, 0, 255)
        projection = projection.astype(np.uint8)

        # Convert to JPEG base64
        img = Image.fromarray(projection)

        # Resize if too large
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)

        return base64.b64encode(buffer.getvalue()).decode()

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Get list of sessions with summary stats."""
        rows = self.conn.execute("""
            SELECT
                s.session_id,
                s.name,
                s.created_at,
                COUNT(DISTINCT e.embryo_id) as embryo_count,
                (SELECT COUNT(*) FROM ground_truth g
                 WHERE g.session_id = s.session_id) as gt_count
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
            }
            for r in rows
        ]

    def get_perception_runs(self) -> List[Dict[str, Any]]:
        """Get list of perception runs with metrics."""
        rows = self.conn.execute("""
            SELECT * FROM v_run_accuracy
            ORDER BY run_id DESC
        """).fetchall()

        return [dict(r) for r in rows]
