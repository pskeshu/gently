"""
Data aggregator for populating the embryo dataset database.

Scans existing data directories and populates the SQLite database:
- Sessions from D:/gently/sessions/*.json
- Volumes from D:/gently/data/volume/ and D:/gently/images/
- Images from D:/gently/data/image/
- Ground truth from benchmarks/data/ground_truth/
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .schema import init_database

logger = logging.getLogger(__name__)


class DatasetAggregator:
    """
    Aggregates data from various sources into the SQLite database.

    Parameters
    ----------
    db_path : Path, optional
        Path to database file. Defaults to D:/gently/dataset.db
    sessions_dir : Path, optional
        Directory containing session JSON files
    data_dir : Path, optional
        Root data directory containing volume/, image/, analysis/
    images_dir : Path, optional
        Additional images directory (D:/gently/images/)
    ground_truth_dir : Path, optional
        Directory containing ground truth JSON files
    """

    def __init__(
        self,
        db_path: Path | None = None,
        sessions_dir: Path = Path("D:/gently/sessions"),
        data_dir: Path = Path("D:/gently/data"),
        images_dir: Path = Path("D:/gently/images"),
        ground_truth_dir: Path | None = None,
    ):
        self.db_path = db_path or Path("D:/gently/dataset.db")
        self.sessions_dir = sessions_dir
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.ground_truth_dir = (
            ground_truth_dir
            or Path(__file__).parent.parent.parent / "benchmarks" / "data" / "ground_truth"
        )
        self.conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Initialize and connect to the database."""
        self.conn = init_database(self.db_path)
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def aggregate_all(self, incremental: bool = True) -> dict[str, int]:
        """
        Run full aggregation from all sources.

        Parameters
        ----------
        incremental : bool
            If True, only process new/modified files since last run

        Returns
        -------
        dict
            Statistics about items processed
        """
        if not self.conn:
            self.connect()

        stats = {
            "sessions": 0,
            "embryos": 0,
            "volumes": 0,
            "images": 0,
            "ground_truth": 0,
        }

        # Get last aggregation time for incremental updates
        last_run = self._get_last_run_time() if incremental else None
        if last_run:
            logger.info(f"Incremental aggregation since {last_run}")
        else:
            logger.info("Full aggregation")

        # 1. Aggregate sessions
        session_stats = self.aggregate_sessions(since=last_run)
        stats["sessions"] = session_stats["added"] + session_stats["updated"]
        stats["embryos"] = session_stats.get("embryos", 0)

        # 2. Aggregate volumes from data directory
        volume_stats = self.aggregate_volumes(since=last_run)
        stats["volumes"] = volume_stats["added"]

        # 3. Aggregate images
        image_stats = self.aggregate_images(since=last_run)
        stats["images"] = image_stats["added"]

        # 4. Import ground truth
        gt_stats = self.aggregate_ground_truth()
        stats["ground_truth"] = gt_stats["added"]

        # Update last run time
        self._set_last_run_time(datetime.now())

        logger.info(f"Aggregation complete: {stats}")
        return stats

    def aggregate_sessions(self, since: datetime | None = None) -> dict[str, int]:
        """
        Aggregate session data from JSON files.

        Returns
        -------
        dict
            Statistics: added, updated, embryos
        """
        stats = {"added": 0, "updated": 0, "embryos": 0}

        if not self.sessions_dir.exists():
            logger.warning(f"Sessions directory not found: {self.sessions_dir}")
            return stats

        log_id = self._start_aggregation_log("sessions", str(self.sessions_dir))

        try:
            for session_file in self.sessions_dir.glob("*.json"):
                # Skip non-session files
                if session_file.name == "timeline.jsonl":
                    continue

                # Check modification time for incremental
                if since:
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if mtime < since:
                        continue

                try:
                    with open(session_file, encoding="utf-8") as f:
                        data = json.load(f)

                    result = self._process_session(data)
                    if result == "added":
                        stats["added"] += 1
                    elif result == "updated":
                        stats["updated"] += 1

                    # Count embryos
                    embryos = data.get("embryo_states", {}) or data.get("experiment_data", {}).get(
                        "embryos", {}
                    )
                    stats["embryos"] += len(embryos)

                except Exception as e:
                    logger.error(f"Error processing session {session_file}: {e}")

            self._complete_aggregation_log(
                log_id,
                stats["added"] + stats["updated"],
                stats["added"],
                stats["updated"],
            )

        except Exception as e:
            self._fail_aggregation_log(log_id, str(e))
            raise

        return stats

    def _process_session(self, data: dict[str, Any]) -> str:
        """
        Insert or update a session record.

        Returns
        -------
        str
            'added' or 'updated'
        """
        session_id = data.get("session_id")
        if not session_id:
            return "skipped"

        # Check if exists
        assert self.conn is not None
        existing = self.conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()

        # Extract metadata
        metadata = {
            k: v
            for k, v in data.items()
            if k
            not in (
                "session_id",
                "name",
                "description",
                "created_at",
                "last_active",
                "conversation",
                "system_prompt",
                "experiment_data",
                "embryo_states",
            )
        }

        if existing:
            assert self.conn is not None
            self.conn.execute(
                """
                UPDATE sessions SET
                    name = ?, description = ?, last_active = ?, metadata_json = ?
                WHERE session_id = ?
            """,
                (
                    data.get("name"),
                    data.get("description"),
                    data.get("last_active"),
                    json.dumps(metadata) if metadata else None,
                    session_id,
                ),
            )
            result = "updated"
        else:
            assert self.conn is not None
            self.conn.execute(
                """
                INSERT INTO sessions
                (session_id, name, description, created_at, last_active, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    data.get("name"),
                    data.get("description"),
                    data.get("created_at"),
                    data.get("last_active"),
                    json.dumps(metadata) if metadata else None,
                ),
            )
            result = "added"

        # Process embryos
        embryos = data.get("embryo_states", {}) or data.get("experiment_data", {}).get(
            "embryos", {}
        )
        for embryo_id, embryo_data in embryos.items():
            self._process_embryo(session_id, embryo_id, embryo_data)

        assert self.conn is not None
        self.conn.commit()
        return result

    def _process_embryo(self, session_id: str, embryo_id: str, data: dict[str, Any]):
        """Insert or update an embryo record."""
        stage_pos = data.get("stage_position", {})
        calibration = data.get("calibration", {})

        # Get UID from data, or generate backward-compatible UID
        embryo_uid = data.get("uid") or f"{session_id}_{embryo_id}"

        assert self.conn is not None
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embryos
            (embryo_id, session_id, nickname, user_label,
             stage_position_x, stage_position_y, calibration_json, embryo_uid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                embryo_id,
                session_id,
                data.get("nickname"),
                data.get("user_label"),
                stage_pos.get("x"),
                stage_pos.get("y"),
                json.dumps(calibration) if calibration else None,
                embryo_uid,
            ),
        )

    def aggregate_volumes(self, since: datetime | None = None) -> dict[str, int]:
        """
        Aggregate volume data from data directory.

        Scans both D:/gently/data/volume/ and D:/gently/images/

        Returns
        -------
        dict
            Statistics: added, skipped
        """
        stats = {"added": 0, "skipped": 0}

        # Scan D:/gently/data/volume/{YYYYMMDD}/*.json
        volume_dir = self.data_dir / "volume"
        if volume_dir.exists():
            log_id = self._start_aggregation_log("volumes", str(volume_dir))
            try:
                for date_dir in sorted(volume_dir.iterdir()):
                    if not date_dir.is_dir():
                        continue

                    for meta_file in date_dir.glob("*.json"):
                        if since:
                            mtime = datetime.fromtimestamp(meta_file.stat().st_mtime)
                            if mtime < since:
                                continue

                        try:
                            with open(meta_file, encoding="utf-8") as f:
                                data = json.load(f)

                            if self._process_volume(data, meta_file):
                                stats["added"] += 1
                            else:
                                stats["skipped"] += 1

                        except Exception as e:
                            logger.error(f"Error processing volume {meta_file}: {e}")
                            stats["skipped"] += 1

                assert self.conn is not None
                self.conn.commit()
                self._complete_aggregation_log(
                    log_id, stats["added"] + stats["skipped"], stats["added"], 0
                )

            except Exception as e:
                self._fail_aggregation_log(log_id, str(e))
                raise

        # Also scan D:/gently/images/{session_id}/*.tif
        if self.images_dir.exists():
            for session_dir in self.images_dir.iterdir():
                if not session_dir.is_dir():
                    continue

                session_id = session_dir.name

                for tif_file in session_dir.glob("*.tif"):
                    if since:
                        mtime = datetime.fromtimestamp(tif_file.stat().st_mtime)
                        if mtime < since:
                            continue

                    try:
                        if self._process_volume_from_tiff(tif_file, session_id):
                            stats["added"] += 1
                        else:
                            stats["skipped"] += 1
                    except Exception as e:
                        logger.error(f"Error processing TIFF {tif_file}: {e}")
                        stats["skipped"] += 1

            assert self.conn is not None
            self.conn.commit()

        return stats

    def _process_volume(self, data: dict[str, Any], meta_file: Path) -> bool:
        """
        Process a volume from its metadata JSON.

        Returns True if added, False if skipped (already exists).
        """
        uid = data.get("uid")
        if not uid:
            return False

        # Check if exists
        assert self.conn is not None
        existing = self.conn.execute("SELECT uid FROM volumes WHERE uid = ?", (uid,)).fetchone()
        if existing:
            return False

        # Find TIFF file
        tiff_path = meta_file.with_suffix(".tif")
        if not tiff_path.exists():
            tiff_path = meta_file.with_suffix(".tiff")

        metadata = data.get("metadata", {})
        session_id = metadata.get("session_id")
        embryo_id = metadata.get("embryo_id")

        # Look up embryo_uid from embryos table, or generate backward-compatible UID
        embryo_uid = None
        if session_id and embryo_id:
            assert self.conn is not None
            result = self.conn.execute(
                "SELECT embryo_uid FROM embryos WHERE session_id = ? AND embryo_id = ?",
                (session_id, embryo_id),
            ).fetchone()
            embryo_uid = result[0] if result else f"{session_id}_{embryo_id}"

        assert self.conn is not None
        self.conn.execute(
            """
            INSERT INTO volumes
            (uid, session_id, embryo_id, timepoint, file_path, shape_json, dtype,
             timestamp, metadata_json, embryo_uid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                uid,
                session_id,
                embryo_id,
                metadata.get("timepoint"),
                str(tiff_path) if tiff_path.exists() else None,
                json.dumps(data.get("shape")),
                data.get("dtype"),
                data.get("timestamp"),
                json.dumps(metadata) if metadata else None,
                embryo_uid,
            ),
        )

        return True

    def _process_volume_from_tiff(self, tif_path: Path, session_id: str) -> bool:
        """
        Process a volume from a TIFF file in the images directory.

        Parses filename like: embryo_1_20251222_173448.tif
        """
        # Generate UID from path
        uid = f"tiff_{tif_path.stem}"

        # Check if exists
        assert self.conn is not None
        existing = self.conn.execute("SELECT uid FROM volumes WHERE uid = ?", (uid,)).fetchone()
        if existing:
            return False

        # Parse filename
        parts = tif_path.stem.split("_")
        embryo_id = None
        timestamp_str = None

        if len(parts) >= 2 and parts[0] == "embryo":
            embryo_id = f"embryo_{parts[1]}"

        if len(parts) >= 4:
            # Try to parse timestamp from filename
            try:
                date_str = parts[2]
                time_str = parts[3]
                timestamp_str = (
                    f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    f"T{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
                )
            except Exception:
                timestamp_str = datetime.fromtimestamp(tif_path.stat().st_mtime).isoformat()

        if not timestamp_str:
            timestamp_str = datetime.fromtimestamp(tif_path.stat().st_mtime).isoformat()

        # Look up embryo_uid from embryos table, or generate backward-compatible UID
        embryo_uid = None
        if session_id and embryo_id:
            assert self.conn is not None
            result = self.conn.execute(
                "SELECT embryo_uid FROM embryos WHERE session_id = ? AND embryo_id = ?",
                (session_id, embryo_id),
            ).fetchone()
            embryo_uid = result[0] if result else f"{session_id}_{embryo_id}"

        assert self.conn is not None
        self.conn.execute(
            """
            INSERT INTO volumes
            (uid, session_id, embryo_id, file_path, timestamp, metadata_json, embryo_uid)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                uid,
                session_id,
                embryo_id,
                str(tif_path),
                timestamp_str,
                json.dumps({"source": "images_dir"}),
                embryo_uid,
            ),
        )

        return True

    def aggregate_images(self, since: datetime | None = None) -> dict[str, int]:
        """
        Aggregate image projection data.

        Returns
        -------
        dict
            Statistics: added, skipped
        """
        stats = {"added": 0, "skipped": 0}

        image_dir = self.data_dir / "image"
        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return stats

        log_id = self._start_aggregation_log("images", str(image_dir))

        try:
            for date_dir in sorted(image_dir.iterdir()):
                if not date_dir.is_dir():
                    continue

                for meta_file in date_dir.glob("*.json"):
                    if since:
                        mtime = datetime.fromtimestamp(meta_file.stat().st_mtime)
                        if mtime < since:
                            continue

                    try:
                        with open(meta_file, encoding="utf-8") as f:
                            data = json.load(f)

                        if self._process_image(data):
                            stats["added"] += 1
                        else:
                            stats["skipped"] += 1

                    except Exception as e:
                        logger.error(f"Error processing image {meta_file}: {e}")
                        stats["skipped"] += 1

            assert self.conn is not None
            self.conn.commit()
            self._complete_aggregation_log(
                log_id, stats["added"] + stats["skipped"], stats["added"], 0
            )

        except Exception as e:
            self._fail_aggregation_log(log_id, str(e))
            raise

        return stats

    def _process_image(self, data: dict[str, Any]) -> bool:
        """
        Process an image from its metadata JSON.

        Returns True if added, False if skipped.
        """
        uid = data.get("uid")
        if not uid:
            return False

        # Check if exists
        assert self.conn is not None
        existing = self.conn.execute("SELECT uid FROM images WHERE uid = ?", (uid,)).fetchone()
        if existing:
            return False

        metadata = data.get("metadata", {})
        session_id = metadata.get("session_id")
        embryo_id = metadata.get("embryo_id")

        # Look up embryo_uid from embryos table, or generate backward-compatible UID
        embryo_uid = None
        if session_id and embryo_id:
            assert self.conn is not None
            result = self.conn.execute(
                "SELECT embryo_uid FROM embryos WHERE session_id = ? AND embryo_id = ?",
                (session_id, embryo_id),
            ).fetchone()
            embryo_uid = result[0] if result else f"{session_id}_{embryo_id}"

        assert self.conn is not None
        self.conn.execute(
            """
            INSERT INTO images
            (uid, parent_uid, session_id, embryo_id, timepoint, projection_type,
             shape_json, dtype, b64_size_kb, timestamp, metadata_json, embryo_uid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                uid,
                metadata.get("parent_uid"),
                session_id,
                embryo_id,
                metadata.get("timepoint"),
                metadata.get("projection_type"),
                json.dumps(data.get("shape")),
                data.get("dtype"),
                metadata.get("b64_size_kb"),
                data.get("timestamp"),
                json.dumps(metadata) if metadata else None,
                embryo_uid,
            ),
        )

        return True

    def aggregate_ground_truth(self) -> dict[str, int]:
        """
        Import ground truth annotations from benchmark data.

        Parses JSON files with format:
        {
            "session_id": "...",
            "transitions": {
                "embryo_1": {"early": 0, "bean": 43, ...},
                ...
            }
        }

        Returns
        -------
        dict
            Statistics: added, updated
        """
        stats = {"added": 0, "updated": 0}

        if not self.ground_truth_dir.exists():
            logger.warning(f"Ground truth directory not found: {self.ground_truth_dir}")
            return stats

        log_id = self._start_aggregation_log("ground_truth", str(self.ground_truth_dir))

        try:
            for gt_file in self.ground_truth_dir.glob("*.json"):
                try:
                    with open(gt_file, encoding="utf-8") as f:
                        data = json.load(f)

                    session_id = data.get("session_id")
                    transitions = data.get("transitions", {})
                    annotator = data.get("annotator")
                    notes = data.get("notes")

                    for embryo_id, stages in transitions.items():
                        for stage, start_timepoint in stages.items():
                            # Check if exists
                            assert self.conn is not None
                            existing = self.conn.execute(
                                """
                                SELECT id FROM ground_truth
                                WHERE session_id = ? AND embryo_id = ? AND stage = ?
                            """,
                                (session_id, embryo_id, stage),
                            ).fetchone()

                            if existing:
                                assert self.conn is not None
                                self.conn.execute(
                                    """
                                    UPDATE ground_truth SET
                                        start_timepoint = ?, annotator = ?, notes = ?
                                    WHERE id = ?
                                """,
                                    (start_timepoint, annotator, notes, existing[0]),
                                )
                                stats["updated"] += 1
                            else:
                                assert self.conn is not None
                                self.conn.execute(
                                    """
                                    INSERT INTO ground_truth
                                    (session_id, embryo_id, stage, start_timepoint,
                                     annotator, notes)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                    (
                                        session_id,
                                        embryo_id,
                                        stage,
                                        start_timepoint,
                                        annotator,
                                        notes,
                                    ),
                                )
                                stats["added"] += 1

                except Exception as e:
                    logger.error(f"Error processing ground truth {gt_file}: {e}")

            assert self.conn is not None
            self.conn.commit()
            self._complete_aggregation_log(
                log_id,
                stats["added"] + stats["updated"],
                stats["added"],
                stats["updated"],
            )

        except Exception as e:
            self._fail_aggregation_log(log_id, str(e))
            raise

        return stats

    def _get_last_run_time(self) -> datetime | None:
        """Get the timestamp of the last successful aggregation."""
        assert self.conn is not None
        result = self.conn.execute("""
            SELECT MAX(completed_at) FROM aggregation_log
            WHERE status = 'completed'
        """).fetchone()

        if result and result[0]:
            return datetime.fromisoformat(result[0])
        return None

    def _set_last_run_time(self, timestamp: datetime):
        """Record the current aggregation time."""
        assert self.conn is not None
        self.conn.execute(
            """
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES ('last_aggregation', ?, ?)
        """,
            (timestamp.isoformat(), timestamp.isoformat()),
        )
        assert self.conn is not None
        self.conn.commit()

    def _start_aggregation_log(self, source_type: str, source_path: str) -> int:
        """Start a new aggregation log entry."""
        assert self.conn is not None
        cursor = self.conn.execute(
            """
            INSERT INTO aggregation_log (source_type, source_path, started_at, status)
            VALUES (?, ?, ?, 'running')
        """,
            (source_type, source_path, datetime.now().isoformat()),
        )
        assert self.conn is not None
        self.conn.commit()
        return cast("int", cursor.lastrowid)

    def _complete_aggregation_log(self, log_id: int, processed: int, added: int, updated: int):
        """Mark an aggregation log as completed."""
        assert self.conn is not None
        self.conn.execute(
            """
            UPDATE aggregation_log SET
                items_processed = ?, items_added = ?, items_updated = ?,
                completed_at = ?, status = 'completed'
            WHERE id = ?
        """,
            (processed, added, updated, datetime.now().isoformat(), log_id),
        )
        assert self.conn is not None
        self.conn.commit()

    def _fail_aggregation_log(self, log_id: int, error_message: str):
        """Mark an aggregation log as failed."""
        assert self.conn is not None
        self.conn.execute(
            """
            UPDATE aggregation_log SET
                completed_at = ?, status = 'failed', error_message = ?
            WHERE id = ?
        """,
            (datetime.now().isoformat(), error_message, log_id),
        )
        assert self.conn is not None
        self.conn.commit()


def get_stage_at_timepoint(
    conn: sqlite3.Connection, session_id: str, embryo_id: str, timepoint: int
) -> str | None:
    """
    Get the ground truth stage at a specific timepoint.

    Uses the ground_truth table to determine which stage the embryo
    was in at the given timepoint.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection
    session_id : str
        Session ID
    embryo_id : str
        Embryo ID
    timepoint : int
        Timepoint to query

    Returns
    -------
    str or None
        Stage name or None if not found
    """
    result = conn.execute(
        """
        SELECT stage FROM ground_truth
        WHERE session_id = ? AND embryo_id = ? AND start_timepoint <= ?
        ORDER BY start_timepoint DESC
        LIMIT 1
    """,
        (session_id, embryo_id, timepoint),
    ).fetchone()

    return result[0] if result else None
