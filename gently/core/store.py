"""
GentlyStore — Unified data storage for Gently.

One class, one SQLite database, one directory tree under a single root.

Replaces:
    - TiledStore / DatabrokerStore  (UID-based file store + index.json)
    - ImageManager                  (volume TIFF writing + in-memory index)
    - SessionManager                (session JSON persistence)
    - TracePersister                (dual JSON + SQLite trace persistence)
    - VizServer ImageStore          (in-memory image cache)

Directory layout under *root*::

    gently.db
    incoming/                          # staging for device-written TIFFs
    volumes/{session_id}/{embryo_id}_t{tp:04d}.tif
    projections/{session_id}/{embryo_id}_t{tp:04d}.jpg
    snapshots/{session_id}/{source}_{timestamp}.tif
    traces/{session_id}/{embryo_id}_t{tp:04d}.json
    sessions/{session_id}.json

Usage::

    store = GentlyStore(Path("D:/Gently2"))
    store.create_session("s1", name="Overnight run")
    store.register_embryo("s1", "embryo_1", position_x=100.0, position_y=200.0)
    path = store.put_volume("s1", "embryo_1", 0, volume_array)
    proj = store.get_projection_path("s1", "embryo_1", 0)
"""

import json
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from .store_types import (
    EmbryoInfo,
    GroundTruthEntry,
    PredictionInfo,
    ProjectionInfo,
    SessionInfo,
    StoreStats,
    VolumeInfo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """\
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    name         TEXT,
    description  TEXT,
    created_at   TEXT NOT NULL,
    last_active  TEXT NOT NULL,
    metadata     TEXT
);

CREATE TABLE IF NOT EXISTS embryos (
    embryo_id    TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    embryo_uid   TEXT,
    nickname     TEXT,
    position_x   REAL,
    position_y   REAL,
    calibration  TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (embryo_id, session_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS volumes (
    session_id   TEXT NOT NULL,
    embryo_id    TEXT NOT NULL,
    timepoint    INTEGER NOT NULL,
    file_path    TEXT NOT NULL,
    shape        TEXT,
    dtype        TEXT,
    acquired_at  TEXT NOT NULL,
    metadata     TEXT,
    PRIMARY KEY (session_id, embryo_id, timepoint),
    FOREIGN KEY (embryo_id, session_id) REFERENCES embryos(embryo_id, session_id)
);

CREATE TABLE IF NOT EXISTS projections (
    session_id   TEXT NOT NULL,
    embryo_id    TEXT NOT NULL,
    timepoint    INTEGER NOT NULL,
    file_path    TEXT NOT NULL,
    width        INTEGER,
    height       INTEGER,
    size_kb      REAL,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (session_id, embryo_id, timepoint),
    FOREIGN KEY (session_id, embryo_id, timepoint)
        REFERENCES volumes(session_id, embryo_id, timepoint)
);

CREATE TABLE IF NOT EXISTS perception_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT,
    name                TEXT NOT NULL,
    perception_method   TEXT NOT NULL,
    model_name          TEXT,
    trace_type          TEXT DEFAULT 'perception',
    source              TEXT DEFAULT 'live',
    config              TEXT,
    status              TEXT DEFAULT 'running',
    created_at          TEXT NOT NULL,
    completed_at        TEXT,
    error_message       TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              INTEGER NOT NULL,
    session_id          TEXT NOT NULL,
    embryo_id           TEXT NOT NULL,
    timepoint           INTEGER NOT NULL,
    predicted_stage     TEXT NOT NULL,
    confidence          REAL,
    reasoning           TEXT,
    is_transitional     INTEGER DEFAULT 0,
    ground_truth_stage  TEXT,
    is_correct          INTEGER,
    execution_time_ms   REAL,
    trace_file          TEXT,
    observed_features   TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES perception_runs(run_id),
    FOREIGN KEY (session_id, embryo_id, timepoint)
        REFERENCES volumes(session_id, embryo_id, timepoint)
);

CREATE TABLE IF NOT EXISTS ground_truth (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    embryo_id       TEXT NOT NULL,
    stage           TEXT NOT NULL,
    start_timepoint INTEGER NOT NULL,
    end_timepoint   INTEGER,
    annotator       TEXT,
    notes           TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE (session_id, embryo_id, stage),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    source       TEXT NOT NULL,   -- 'bottom_camera', 'lightsheet', etc.
    file_path    TEXT NOT NULL,
    width        INTEGER,
    height       INTEGER,
    metadata     TEXT,
    captured_at  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id, source);
CREATE INDEX IF NOT EXISTS idx_volumes_acquired ON volumes(acquired_at);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_embryo
    ON predictions(session_id, embryo_id, timepoint);
CREATE INDEX IF NOT EXISTS idx_predictions_stage ON predictions(predicted_stage);
CREATE INDEX IF NOT EXISTS idx_runs_session ON perception_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_embryos_session ON embryos(session_id, created_at);

-- Views
CREATE VIEW IF NOT EXISTS v_latest_prediction AS
SELECT p.* FROM predictions p
INNER JOIN (
    SELECT session_id, embryo_id, timepoint, MAX(prediction_id) AS max_id
    FROM predictions GROUP BY session_id, embryo_id, timepoint
) latest ON p.prediction_id = latest.max_id;
"""


# ---------------------------------------------------------------------------
# GentlyStore
# ---------------------------------------------------------------------------


class GentlyStore:
    """One class for all Gently storage. Owns one SQLite DB + one directory tree."""

    def __init__(self, root: Path):
        """
        Parameters
        ----------
        root : Path
            Root directory for all data (e.g. ``Path("D:/Gently2")``).
            Created if it does not exist.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ("incoming", "volumes", "projections", "traces", "sessions"):
            (self.root / subdir).mkdir(exist_ok=True)

        # Open database
        self._db_path = self.root / "gently.db"
        self._conn = self._open_db()

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        return conn

    @contextmanager
    def _tx(self):
        """Context manager that commits on success, rolls back on error."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _rel_path(self, absolute_path: Path) -> str:
        """Convert absolute path to root-relative string for DB storage."""
        try:
            return str(Path(absolute_path).relative_to(self.root))
        except ValueError:
            # Outside root — store as-is
            return str(absolute_path)

    def _ensure_subdir(self, *parts: str) -> Path:
        """Create and return a subdirectory under root."""
        d = self.root.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _parse_json_field(d: dict, field: str):
        """Decode a JSON string field in-place, if present."""
        if d.get(field):
            d[field] = json.loads(d[field])

    def _abs_path(self, rel_path: str) -> Path:
        """Convert root-relative string to absolute path."""
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return self.root / p

    # ==================================================================
    # Sessions
    # ==================================================================

    def create_session(
        self,
        session_id: str,
        name: str | None = None,
        description: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new session. Returns session_id."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(session_id, name, description, created_at, last_active, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    name,
                    description,
                    now,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )
        logger.info(f"Created session {session_id}")
        return session_id

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Return session row as dict, or None."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        self._parse_json_field(d, "metadata")
        return d  # type: ignore[return-value]  # dict matches TypedDict shape

    def list_sessions(self) -> list[SessionInfo]:
        """Return all sessions ordered by last_active descending."""
        rows = self._conn.execute("SELECT * FROM sessions ORDER BY last_active DESC").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            self._parse_json_field(d, "metadata")
            result.append(d)
        return result  # type: ignore[return-value]  # dict matches TypedDict shape

    def touch_session(self, session_id: str):
        """Update last_active timestamp."""
        with self._tx():
            self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE session_id = ?",
                (self._now(), session_id),
            )

    def save_session_snapshot(self, session_id: str, snapshot: dict):
        """Write a JSON snapshot (conversation + state) to sessions/{id}.json."""
        path = self.root / "sessions" / f"{session_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)
        self.touch_session(session_id)

    def load_session_snapshot(self, session_id: str) -> dict | None:
        """Load session snapshot JSON. Returns None if missing."""
        path = self.root / "sessions" / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    # ==================================================================
    # Embryos
    # ==================================================================

    def register_embryo(
        self,
        session_id: str,
        embryo_id: str,
        embryo_uid: str | None = None,
        nickname: str | None = None,
        position_x: float | None = None,
        position_y: float | None = None,
        calibration: dict | None = None,
    ):
        """Register or update an embryo in a session."""
        now = self._now()
        cal_json = json.dumps(calibration) if calibration else None
        with self._tx():
            self._conn.execute(
                "INSERT INTO embryos "
                "(embryo_id, session_id, embryo_uid, nickname, "
                " position_x, position_y, calibration, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(embryo_id, session_id) DO UPDATE SET "
                "  embryo_uid = COALESCE(excluded.embryo_uid, embryos.embryo_uid), "
                "  nickname = COALESCE(excluded.nickname, embryos.nickname), "
                "  position_x = COALESCE(excluded.position_x, embryos.position_x), "
                "  position_y = COALESCE(excluded.position_y, embryos.position_y), "
                "  calibration = COALESCE(excluded.calibration, embryos.calibration)",
                (
                    embryo_id,
                    session_id,
                    embryo_uid,
                    nickname,
                    position_x,
                    position_y,
                    cal_json,
                    now,
                ),
            )

    def get_embryo(self, session_id: str, embryo_id: str) -> EmbryoInfo | None:
        row = self._conn.execute(
            "SELECT * FROM embryos WHERE session_id = ? AND embryo_id = ?",
            (session_id, embryo_id),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        self._parse_json_field(d, "calibration")
        return d  # type: ignore[return-value]  # dict matches TypedDict shape

    def list_embryos(self, session_id: str) -> list[EmbryoInfo]:
        rows = self._conn.execute(
            "SELECT * FROM embryos WHERE session_id = ? ORDER BY embryo_id",
            (session_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            self._parse_json_field(d, "calibration")
            result.append(d)
        return result  # type: ignore[return-value]  # dict matches TypedDict shape

    # ==================================================================
    # Volumes
    # ==================================================================

    def _volume_dir(self, session_id: str) -> Path:
        return self._ensure_subdir("volumes", session_id)

    def _volume_filename(self, embryo_id: str, timepoint: int) -> str:
        return f"{embryo_id}_t{timepoint:04d}.tif"

    def _projection_dir(self, session_id: str) -> Path:
        return self._ensure_subdir("projections", session_id)

    def _projection_filename(self, embryo_id: str, timepoint: int) -> str:
        return f"{embryo_id}_t{timepoint:04d}.jpg"

    def put_volume(
        self,
        session_id: str,
        embryo_id: str,
        timepoint: int,
        volume: np.ndarray,
        metadata: dict | None = None,
    ) -> Path:
        """
        Write a volume to disk, generate a JPEG projection, insert DB rows.

        Use this in offline mode or when the device process doesn't write files.

        Parameters
        ----------
        session_id, embryo_id, timepoint
            Natural key for the volume.
        volume : np.ndarray
            Raw volume data (3D or 4D).
        metadata : dict, optional
            Extra metadata stored as JSON in the volumes row.

        Returns
        -------
        Path
            Absolute path to the written TIFF file.
        """
        import tifffile

        vol_dir = self._volume_dir(session_id)
        vol_path = vol_dir / self._volume_filename(embryo_id, timepoint)

        tifffile.imwrite(str(vol_path), volume, compression="zlib")

        # Generate projection
        proj_path = self._generate_projection(session_id, embryo_id, timepoint, volume)

        # Insert DB rows
        self._insert_volume_row(
            session_id,
            embryo_id,
            timepoint,
            vol_path,
            volume.shape,
            str(volume.dtype),
            metadata,
        )
        if proj_path is not None:
            self._insert_projection_row(
                session_id,
                embryo_id,
                timepoint,
                proj_path,
            )

        logger.debug(f"put_volume: {session_id}/{embryo_id} t={timepoint} -> {vol_path}")
        return vol_path

    def register_volume(
        self,
        session_id: str,
        embryo_id: str,
        timepoint: int,
        incoming_path: Path,
        metadata: dict | None = None,
        volume_data: np.ndarray | None = None,
    ) -> Path:
        """
        Zero-copy path: rename an existing TIFF to canonical location.

        Used when the device process already wrote the file (e.g. to
        ``incoming/{uuid}.tif``). The file is **moved** (renamed) to
        ``volumes/{session_id}/{embryo_id}_t{tp:04d}.tif``.

        Generates a JPEG projection and inserts DB rows.

        Parameters
        ----------
        incoming_path : Path
            Path to the already-written TIFF file.
        volume_data : np.ndarray, optional
            Already-loaded volume array.  When provided the moved file
            is **not** re-read from disk, saving one full TIFF decode.

        Returns
        -------
        Path
            Canonical path after rename.
        """
        incoming_path = Path(incoming_path)
        if not incoming_path.exists():
            raise FileNotFoundError(f"Incoming file not found: {incoming_path}")

        vol_dir = self._volume_dir(session_id)
        canonical = vol_dir / self._volume_filename(embryo_id, timepoint)

        # Move (rename if same drive, copy+delete otherwise)
        if canonical.exists():
            canonical.unlink()
        try:
            incoming_path.rename(canonical)
        except OSError:
            # Cross-device rename: copy then delete
            shutil.copy2(str(incoming_path), str(canonical))
            incoming_path.unlink()

        # Use caller-provided array or read from disk
        if volume_data is not None:
            volume = volume_data
        else:
            from gently.core.imaging import load_volume

            volume = load_volume(canonical)

        proj_path = self._generate_projection(session_id, embryo_id, timepoint, volume)

        self._insert_volume_row(
            session_id,
            embryo_id,
            timepoint,
            canonical,
            volume.shape,
            str(volume.dtype),
            metadata,
        )
        if proj_path is not None:
            self._insert_projection_row(
                session_id,
                embryo_id,
                timepoint,
                proj_path,
            )

        logger.debug(f"register_volume: {incoming_path.name} -> {canonical}")
        return canonical

    # ------------------------------------------------------------------
    # Snapshots (bottom camera, etc.)
    # ------------------------------------------------------------------

    def _snapshot_dir(self, session_id: str) -> Path:
        return self._ensure_subdir("snapshots", session_id)

    def register_snapshot(
        self,
        session_id: str,
        source: str,
        incoming_path: Path,
        metadata: dict | None = None,
    ) -> Path:
        """Move a transient TIFF from *incoming/* to ``snapshots/{session}/``.

        Parameters
        ----------
        session_id : str
            Current session identifier.
        source : str
            Camera/source name, e.g. ``"bottom_camera"``.
        incoming_path : Path
            Path to the file in the incoming staging directory.
        metadata : dict, optional
            Extra metadata to store in the DB row.

        Returns
        -------
        Path
            Canonical path after the move.
        """
        incoming_path = Path(incoming_path)
        if not incoming_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {incoming_path}")

        snap_dir = self._snapshot_dir(session_id)
        now = self._now()
        # Use original stem (UUID) to avoid collisions
        canonical = snap_dir / f"{source}_{incoming_path.stem}.tif"

        try:
            incoming_path.rename(canonical)
        except OSError:
            shutil.copy2(str(incoming_path), str(canonical))
            incoming_path.unlink()

        # Read shape for DB record
        import tifffile

        arr = tifffile.imread(str(canonical))

        with self._tx():
            self._conn.execute(
                "INSERT INTO snapshots "
                "(session_id, source, file_path, width, height, metadata, captured_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    source,
                    self._rel_path(canonical),
                    arr.shape[-1] if arr.ndim >= 2 else None,
                    arr.shape[-2] if arr.ndim >= 2 else None,
                    json.dumps(metadata) if metadata else None,
                    now,
                ),
            )
        logger.debug("register_snapshot: %s -> %s", incoming_path.name, canonical)
        return canonical

    def list_snapshots(self, session_id: str, source: str | None = None) -> list[dict[str, Any]]:
        """List snapshot records for a session, optionally filtered by source."""
        if source:
            rows = self._conn.execute(
                "SELECT * FROM snapshots WHERE session_id = ? AND source = ? ORDER BY captured_at",
                (session_id, source),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM snapshots WHERE session_id = ? ORDER BY captured_at",
                (session_id,),
            ).fetchall()
        cols = [
            d[0] for d in self._conn.execute("SELECT * FROM snapshots LIMIT 0").description or []
        ]
        return [dict(zip(cols, row, strict=False)) for row in rows]

    # ------------------------------------------------------------------
    # Incoming cleanup
    # ------------------------------------------------------------------

    def cleanup_incoming(self, max_age_seconds: float = 300) -> int:
        """Delete stale files from the incoming staging directory.

        Files older than *max_age_seconds* (default 5 min) are assumed
        orphaned — read into memory but never registered or moved.

        Returns the number of files deleted.
        """
        import time

        incoming = self.incoming_dir
        if not incoming.exists():
            return 0

        cutoff = time.time() - max_age_seconds
        deleted = 0
        for f in incoming.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                try:
                    f.unlink()
                    deleted += 1
                    logger.debug("cleanup_incoming: deleted %s", f.name)
                except OSError as e:
                    logger.warning("cleanup_incoming: could not delete %s: %s", f.name, e)
        if deleted:
            logger.info("cleanup_incoming: removed %d stale file(s)", deleted)
        return deleted

    # ------------------------------------------------------------------
    # Volume retrieval
    # ------------------------------------------------------------------

    def get_volume(self, session_id: str, embryo_id: str, timepoint: int) -> np.ndarray | None:
        """Load a volume from disk. Returns None if not found."""
        path = self.get_volume_path(session_id, embryo_id, timepoint)
        if path is None or not path.exists():
            return None
        import tifffile

        return tifffile.imread(str(path))

    def get_volume_path(self, session_id: str, embryo_id: str, timepoint: int) -> Path | None:
        """Return the absolute path to a volume, or None."""
        row = self._conn.execute(
            "SELECT file_path FROM volumes "
            "WHERE session_id = ? AND embryo_id = ? AND timepoint = ?",
            (session_id, embryo_id, timepoint),
        ).fetchone()
        if row is None:
            return None
        return self._abs_path(row["file_path"])

    def list_volumes(self, session_id: str, embryo_id: str | None = None) -> list[VolumeInfo]:
        """List volume metadata rows for a session (optionally filtered)."""
        if embryo_id:
            rows = self._conn.execute(
                "SELECT * FROM volumes WHERE session_id = ? AND embryo_id = ? ORDER BY timepoint",
                (session_id, embryo_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM volumes WHERE session_id = ? ORDER BY embryo_id, timepoint",
                (session_id,),
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("shape"):
                d["shape"] = json.loads(d["shape"])
            if d.get("metadata"):
                d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result  # type: ignore[return-value]  # dict matches TypedDict shape

    def get_acquisition_params(self, session_id: str, embryo_id: str | None = None) -> dict | None:
        """
        Get the acquisition parameters used in a session.

        Returns the metadata from the most recent volume, which contains
        num_slices, exposure_ms, interval_seconds, calibration, etc.

        Parameters
        ----------
        session_id : str
            Session to query.
        embryo_id : str, optional
            If provided, get params for this specific embryo.

        Returns
        -------
        dict or None
            Acquisition metadata dict, or None if no volumes found.
        """
        if embryo_id:
            row = self._conn.execute(
                "SELECT metadata FROM volumes "
                "WHERE session_id = ? AND embryo_id = ? AND metadata IS NOT NULL "
                "ORDER BY timepoint DESC LIMIT 1",
                (session_id, embryo_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT metadata FROM volumes "
                "WHERE session_id = ? AND metadata IS NOT NULL "
                "ORDER BY timepoint DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row and row["metadata"]:
            return json.loads(row["metadata"])
        return None

    # -- internal helpers --

    def _generate_projection(
        self,
        session_id: str,
        embryo_id: str,
        timepoint: int,
        volume: np.ndarray,
    ) -> Path | None:
        """Generate JPEG projection file from volume data."""
        from .imaging import generate_jpeg_projection

        proj_dir = self._projection_dir(session_id)
        proj_path = proj_dir / self._projection_filename(embryo_id, timepoint)
        return generate_jpeg_projection(volume, proj_path)

    def _insert_volume_row(
        self,
        session_id,
        embryo_id,
        timepoint,
        vol_path,
        shape,
        dtype,
        metadata,
    ):
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO volumes "
                "(session_id, embryo_id, timepoint, file_path, shape, dtype, "
                " acquired_at, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    embryo_id,
                    timepoint,
                    self._rel_path(vol_path),
                    json.dumps(list(shape)),
                    dtype,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def _insert_projection_row(
        self,
        session_id,
        embryo_id,
        timepoint,
        proj_path,
    ):
        from PIL import Image as PILImage

        # Read dimensions and size
        try:
            img = PILImage.open(str(proj_path))
            w, h = img.size
            size_kb = proj_path.stat().st_size / 1024
        except Exception:
            w, h, size_kb = None, None, None

        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO projections "
                "(session_id, embryo_id, timepoint, file_path, width, height, "
                " size_kb, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    embryo_id,
                    timepoint,
                    self._rel_path(proj_path),
                    w,
                    h,
                    size_kb,
                    now,
                ),
            )

    # ==================================================================
    # Projections
    # ==================================================================

    def get_projection_path(self, session_id: str, embryo_id: str, timepoint: int) -> Path | None:
        """Return absolute path to the JPEG projection, or None."""
        row = self._conn.execute(
            "SELECT file_path FROM projections "
            "WHERE session_id = ? AND embryo_id = ? AND timepoint = ?",
            (session_id, embryo_id, timepoint),
        ).fetchone()
        if row is None:
            return None
        return self._abs_path(row["file_path"])

    def get_projection_b64(self, session_id: str, embryo_id: str, timepoint: int) -> str | None:
        """Return base64-encoded JPEG projection, or None."""
        import base64

        path = self.get_projection_path(session_id, embryo_id, timepoint)
        if path is None or not path.exists():
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def list_projections(self, session_id: str, embryo_id: str) -> list[ProjectionInfo]:
        rows = self._conn.execute(
            "SELECT * FROM projections WHERE session_id = ? AND embryo_id = ? ORDER BY timepoint",
            (session_id, embryo_id),
        ).fetchall()
        return [dict(r) for r in rows]  # type: ignore[misc]  # dict matches TypedDict shape

    # ==================================================================
    # Perception Runs & Predictions
    # ==================================================================

    def create_perception_run(
        self,
        session_id: str,
        name: str,
        method: str,
        model_name: str | None = None,
        trace_type: str = "perception",
        source: str = "live",
        config: dict | None = None,
    ) -> int:
        """Create a perception run. Returns run_id."""
        now = self._now()
        with self._tx():
            cursor = self._conn.execute(
                "INSERT INTO perception_runs "
                "(session_id, name, perception_method, model_name, trace_type, "
                " source, config, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)",
                (
                    session_id,
                    name,
                    method,
                    model_name,
                    trace_type,
                    source,
                    json.dumps(config) if config else None,
                    now,
                ),
            )
            return cast("int", cursor.lastrowid)

    def store_prediction(
        self,
        run_id: int,
        session_id: str,
        embryo_id: str,
        timepoint: int,
        predicted_stage: str,
        confidence: float | None = None,
        reasoning: str | None = None,
        is_transitional: bool = False,
        execution_time_ms: float | None = None,
        trace_data: dict | None = None,
        observed_features: dict | None = None,
        ground_truth_stage: str | None = None,
        is_correct: int | None = None,
    ) -> int:
        """
        Insert a prediction row. Optionally writes trace JSON file.

        Parameters
        ----------
        trace_data : dict, optional
            If provided, written to ``traces/{session_id}/{embryo_id}_t{tp:04d}.json``.

        Returns
        -------
        int
            prediction_id
        """
        # Write trace file if provided
        trace_file_rel = None
        if trace_data is not None:
            trace_dir = self.root / "traces" / session_id
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{embryo_id}_t{timepoint:04d}.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)
            trace_file_rel = self._rel_path(trace_path)

        now = self._now()
        with self._tx():
            cursor = self._conn.execute(
                "INSERT INTO predictions "
                "(run_id, session_id, embryo_id, timepoint, predicted_stage, "
                " confidence, reasoning, is_transitional, ground_truth_stage, "
                " is_correct, execution_time_ms, trace_file, observed_features, "
                " created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    session_id,
                    embryo_id,
                    timepoint,
                    predicted_stage,
                    confidence,
                    reasoning,
                    1 if is_transitional else 0,
                    ground_truth_stage,
                    is_correct,
                    execution_time_ms,
                    trace_file_rel,
                    json.dumps(observed_features) if observed_features else None,
                    now,
                ),
            )
            return cast("int", cursor.lastrowid)

    def complete_perception_run(
        self, run_id: int, status: str = "completed", error_message: str | None = None
    ):
        """Mark a perception run as completed or failed."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "UPDATE perception_runs SET status = ?, completed_at = ?, "
                "error_message = ? WHERE run_id = ?",
                (status, now, error_message, run_id),
            )

    def get_predictions(
        self,
        session_id: str,
        embryo_id: str | None = None,
        run_id: int | None = None,
    ) -> list[PredictionInfo]:
        """Query predictions with optional filters."""
        clauses = ["session_id = ?"]
        params: list = [session_id]

        if embryo_id:
            clauses.append("embryo_id = ?")
            params.append(embryo_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)

        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM predictions WHERE {where} ORDER BY timepoint, prediction_id",
            params,
        ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            if d.get("observed_features"):
                d["observed_features"] = json.loads(d["observed_features"])
            result.append(d)
        return result  # type: ignore[return-value]  # dict matches TypedDict shape

    # ==================================================================
    # Ground Truth
    # ==================================================================

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
        """Insert or update a ground-truth annotation."""
        with self._tx():
            self._conn.execute(
                "INSERT INTO ground_truth "
                "(session_id, embryo_id, stage, start_timepoint, end_timepoint, "
                " annotator, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, embryo_id, stage) DO UPDATE SET "
                "  start_timepoint = excluded.start_timepoint, "
                "  end_timepoint = excluded.end_timepoint, "
                "  annotator = excluded.annotator, "
                "  notes = excluded.notes",
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

    def get_ground_truth(self, session_id: str, embryo_id: str) -> list[GroundTruthEntry]:
        rows = self._conn.execute(
            "SELECT * FROM ground_truth "
            "WHERE session_id = ? AND embryo_id = ? ORDER BY start_timepoint",
            (session_id, embryo_id),
        ).fetchall()
        return [dict(r) for r in rows]  # type: ignore[misc]  # dict matches TypedDict shape

    # ==================================================================
    # Utility
    # ==================================================================

    def stats(self) -> StoreStats:
        """Return counts and disk-usage summary."""
        tables = [
            "sessions",
            "embryos",
            "volumes",
            "projections",
            "perception_runs",
            "predictions",
            "ground_truth",
        ]
        counts = {}
        for t in tables:
            counts[t] = self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]

        # Disk usage (approximate)
        total_bytes = 0
        for subdir in ("volumes", "projections", "traces", "sessions"):
            d = self.root / subdir
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size

        counts["disk_usage_mb"] = round(total_bytes / (1024 * 1024), 1)
        counts["db_size_mb"] = round(self._db_path.stat().st_size / (1024 * 1024), 2)
        return counts  # type: ignore[return-value]  # dict matches TypedDict shape

    @property
    def db_path(self) -> Path:
        """Path to the SQLite database file."""
        return self._db_path

    @property
    def incoming_dir(self) -> Path:
        """Path to the staging directory for device-written TIFFs."""
        return self.root / "incoming"

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("GentlyStore closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"GentlyStore(root={self.root})"
