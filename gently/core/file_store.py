"""
FileStore -- Pure file-based storage for Gently.

Drop-in replacement for GentlyStore that uses YAML/JSONL files instead
of SQLite.  All state lives under a single root directory (e.g.
``D:/Gently3``) with the following layout::

    sessions/
      _index.yaml                           # session_id -> folder_name
      {YYYYMMDD}_{HHMM}_{slug}_{id8}/
        session.yaml
        session.lock                        # PID + hostname while active
        intent.yaml
        timelapse.yaml
        timeline.jsonl
        interaction_log.jsonl
        conversation.json
        summary.yaml
        perception_runs.yaml                # run_id -> run metadata
        snapshots/
          {source}_{stem}.tif
        embryos/
          {embryo_id}/
            embryo.yaml
            predictions.jsonl
            ground_truth.yaml
            timelapse.mp4
            volumes/
              t{NNNN}.tif
              t{NNNN}.meta.yaml
            projections/
              t{NNNN}.jpg
            traces/
              t{NNNN}.json
    incoming/
      {uuid}.tif
    logs/
      gently_{timestamp}.log
      device_layer_{timestamp}.log

Usage::

    store = FileStore(Path("D:/Gently3"))
    store.create_session("s1", name="Overnight run")
    store.register_embryo("s1", "embryo_1", position_x=100.0, position_y=200.0)
    path = store.put_volume("s1", "embryo_1", 0, volume_array)
    proj = store.get_projection_path("s1", "embryo_1", 0)
"""

import base64
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

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
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str, max_len: int = 30) -> str:
    """Lowercase, replace non-alphanum with hyphens, truncate."""
    if not text:
        return "unnamed"
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        return "unnamed"
    return slug[:max_len]


def _sanitize_for_yaml(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _coarse_from_legacy(record: dict) -> dict | None:
    """Extract coarse XY from an embryo.yaml record, accepting either the new
    `position_coarse` dict or the legacy flat `position_x` / `position_y` keys.
    Returns None if neither shape carries usable values.
    """
    coarse = record.get("position_coarse")
    if isinstance(coarse, dict) and coarse:
        return coarse
    px, py = record.get("position_x"), record.get("position_y")
    if px is None and py is None:
        return None
    out = {}
    if px is not None:
        out["x"] = px
    if py is not None:
        out["y"] = py
    return out or None


def _normalize_embryo_record(record: dict | None) -> EmbryoInfo | None:
    """Backfill an embryo.yaml dict so callers always see the new schema.

    Adds `position_coarse` derived from legacy `position_x` / `position_y` if
    only the legacy fields are present, and ensures `position_fine` exists
    (as None) for forward-compat. The original record is not mutated.
    """
    if record is None:
        return None
    out = dict(record)
    if out.get("position_coarse") is None:
        backfill = _coarse_from_legacy(out)
        if backfill is not None:
            out["position_coarse"] = backfill
    out.setdefault("position_fine", None)
    return cast("EmbryoInfo", out)


def _write_yaml(path: Path, data: Any) -> None:
    """Write YAML atomically: write to a temp file, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _sanitize_for_yaml(data)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", prefix=path.stem, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            f.flush()
            os.fsync(f.fileno())
        # os.replace is atomic and overwrites on Windows — no unlink gap that
        # a crash/power-loss could leave the target missing.
        os.replace(tmp, path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_yaml(path: Path) -> Any:
    """Read a YAML file.  Returns None if missing or empty.

    Tolerant of legacy session files written before _sanitize_for_yaml
    existed — those embed numpy scalars as
    ``!!python/object/apply:numpy.core.multiarray.scalar`` tags that
    safe_load refuses to construct. When we hit such a file we fall
    back to unsafe_load (the only writer of these files is our own
    code on local disk, same trust boundary as the code itself) and
    immediately sanitize the result so the caller always receives
    native Python types. New writes go through _write_yaml + safe_dump
    so legacy form does not propagate.
    """
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return yaml.safe_load(text)
    except yaml.constructor.ConstructorError as err:
        marker = str(err)
        if "python/object" not in marker and "numpy" not in marker:
            raise
        data = yaml.unsafe_load(text)
        return _sanitize_for_yaml(data)


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append a single JSON line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    """Read all lines from a JSONL file."""
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _last_jsonl_record(path: Path) -> dict | None:
    """Return the last parseable JSON record in a JSONL file, reading only the tail.

    Keeps appends O(1): instead of loading + parsing the whole file (which made
    per-prediction writes O(n) and quadratic over a long timelapse), we read a
    bounded window from the end and walk backwards to the last complete line,
    skipping a possible trailing partial line from an interrupted write.
    """
    if not path.exists():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return None
    window = min(size, 65536)
    with open(path, "rb") as f:
        f.seek(size - window)
        data = f.read(window)
    for line in reversed(data.split(b"\n")):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def _now() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# FileStore
# ---------------------------------------------------------------------------


class FileStore:
    """Pure file-based storage for Gently.  Drop-in replacement for GentlyStore."""

    def __init__(self, root: Path):
        """
        Parameters
        ----------
        root : Path
            Root directory for all data (e.g. ``Path("D:/Gently3")``).
            Created if it does not exist.
        """
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

        # Create top-level subdirectories
        for subdir in ("sessions", "incoming", "logs"):
            (self._root / subdir).mkdir(exist_ok=True)

        # Load session index (session_id -> folder_name)
        self._index_path = self._root / "sessions" / "_index.yaml"
        self._index: dict[str, str] = _read_yaml(self._index_path) or {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def incoming_dir(self) -> Path:
        return self._root / "incoming"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_index(self) -> None:
        """Persist the session index mapping to disk."""
        _write_yaml(self._index_path, self._index)

    def _session_dir(self, session_id: str) -> Path | None:
        """Return the session folder path, or None if unknown."""
        folder = self._index.get(session_id)
        if folder is None:
            return None
        return self._root / "sessions" / folder

    def _require_session_dir(self, session_id: str) -> Path:
        """Return session folder path; raise if session does not exist."""
        d = self._session_dir(session_id)
        if d is None or not d.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        return d

    def _embryo_dir(self, session_id: str, embryo_id: str) -> Path:
        sd = self._require_session_dir(session_id)
        return sd / "embryos" / embryo_id

    def _volume_dir(self, session_id: str, embryo_id: str) -> Path:
        d = self._embryo_dir(session_id, embryo_id) / "volumes"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _projection_dir(self, session_id: str, embryo_id: str) -> Path:
        d = self._embryo_dir(session_id, embryo_id) / "projections"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _trace_dir(self, session_id: str, embryo_id: str) -> Path:
        d = self._embryo_dir(session_id, embryo_id) / "traces"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _volume_filename(self, timepoint: int) -> str:
        return f"t{timepoint:04d}.tif"

    def _volume_meta_filename(self, timepoint: int) -> str:
        return f"t{timepoint:04d}.meta.yaml"

    def _projection_filename(self, timepoint: int) -> str:
        return f"t{timepoint:04d}.jpg"

    def _generate_projection(
        self,
        session_id: str,
        embryo_id: str,
        timepoint: int,
        volume: np.ndarray,
    ) -> Path | None:
        """Generate JPEG projection file from volume data."""
        from .imaging import generate_jpeg_projection

        proj_dir = self._projection_dir(session_id, embryo_id)
        proj_path = proj_dir / self._projection_filename(timepoint)
        return generate_jpeg_projection(volume, proj_path)

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
        """Create a new session.  Returns session_id."""
        # If session already exists, return silently (matches INSERT OR IGNORE)
        if session_id in self._index:
            logger.debug("Session %s already exists, skipping create", session_id)
            return session_id

        now_dt = datetime.now()
        now = now_dt.isoformat()
        slug = _slugify(name) if name else "unnamed"
        id8 = session_id[:8] if len(session_id) >= 8 else session_id
        folder_name = f"{now_dt.strftime('%Y%m%d')}_{now_dt.strftime('%H%M')}_{slug}_{id8}"

        session_path = self._root / "sessions" / folder_name
        session_path.mkdir(parents=True, exist_ok=True)
        (session_path / "embryos").mkdir(exist_ok=True)
        (session_path / "snapshots").mkdir(exist_ok=True)

        session_data = {
            "session_id": session_id,
            "name": name,
            "description": description,
            "created_at": now,
            "last_active": now,
            "metadata": metadata,
        }
        _write_yaml(session_path / "session.yaml", session_data)

        # Update index
        self._index[session_id] = folder_name
        self._save_index()

        logger.info("Created session %s -> %s", session_id, folder_name)
        return session_id

    def get_session(self, session_id: str) -> SessionInfo | None:
        """Return session info as dict, or None."""
        sd = self._session_dir(session_id)
        if sd is None or not sd.exists():
            return None
        data = _read_yaml(sd / "session.yaml")
        if data is None:
            return None
        return data

    def list_sessions(self) -> list[SessionInfo]:
        """Return all sessions ordered by last_active descending."""
        sessions = []
        for sid in self._index:
            info = self.get_session(sid)
            if info is not None:
                sessions.append(info)
        sessions.sort(key=lambda s: s.get("last_active", ""), reverse=True)
        return sessions

    def recent_session_ids(self, limit: int = 8) -> list[str]:
        """Most-recent session IDs by folder-name date prefix, *cheaply*.

        Folder names are ``{YYYYMMDD}_{HHMM}_{slug}_{id8}`` so a reverse lexical
        sort of the index orders them newest-first by creation time — no
        ``session.yaml`` parse required. This is a creation-recency proxy (a
        long-dormant session that was just resumed sorts by its original date),
        which is fine for at-a-glance landing views; use ``list_sessions`` when
        exact ``last_active`` ordering matters.
        """
        items = sorted(self._index.items(), key=lambda kv: kv[1], reverse=True)
        if limit and limit > 0:
            items = items[:limit]
        return [sid for sid, _ in items]

    def touch_session(self, session_id: str) -> None:
        """Update last_active timestamp."""
        sd = self._session_dir(session_id)
        if sd is None or not sd.exists():
            return
        yaml_path = sd / "session.yaml"
        data = _read_yaml(yaml_path)
        if data is None:
            return
        data["last_active"] = _now()
        _write_yaml(yaml_path, data)

    def save_session_snapshot(self, session_id: str, snapshot: dict) -> None:
        """Write conversation.json in the session folder."""
        sd = self._require_session_dir(session_id)
        path = sd / "conversation.json"
        # Write atomically via temp file
        fd, tmp = tempfile.mkstemp(suffix=".tmp", prefix="conversation", dir=str(sd))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.touch_session(session_id)

    def load_session_snapshot(self, session_id: str) -> dict | None:
        """Load conversation.json.  Returns None if missing."""
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        path = sd / "conversation.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def append_temperature_sample(self, session_id: str, sample: dict) -> None:
        """Append one temperature reading to the session's temperature.jsonl."""
        sd = self._require_session_dir(session_id)
        _append_jsonl(sd / "temperature.jsonl", sample)

    def read_temperature_log(self, session_id: str, since: str | None = None) -> list[dict]:
        """Return temperature samples for a session, optionally filtered to
        t >= since (ISO-UTC string).

        Reads lines tolerantly: a truncated trailing line (e.g. after a mid-append
        crash) is silently skipped rather than raising a JSONDecodeError.
        """
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        path = sd / "temperature.jsonl"
        if not path.exists():
            return []
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # truncated or corrupt line — skip
        if since is not None:
            rows = [r for r in rows if str(r.get("t", "")) >= since]
        return rows

    # ------------------------------------------------------------------
    # Session lock
    # ------------------------------------------------------------------

    def acquire_session_lock(self, session_id: str) -> None:
        """Write a lock file containing PID + hostname."""
        sd = self._require_session_dir(session_id)
        lock_data = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": _now(),
        }
        _write_yaml(sd / "session.lock", lock_data)
        logger.debug("Acquired lock for session %s", session_id)

    def release_session_lock(self, session_id: str) -> None:
        """Remove the session lock file."""
        sd = self._session_dir(session_id)
        if sd is None:
            return
        lock_path = sd / "session.lock"
        if lock_path.exists():
            lock_path.unlink()
            logger.debug("Released lock for session %s", session_id)

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
        position_coarse: dict | None = None,
        position_fine: dict | None = None,
        calibration: dict | None = None,
        role: str | None = None,
        strain: str | None = None,
    ) -> None:
        """Register or update an embryo in a session.

        ``role`` is the experimental role key from gently.harness.roles.REGISTRY
        (e.g. ``"test"``, ``"calibration"``, ``"unassigned"``). Persisted in
        embryo.yaml. None preserves the existing value on update.

        ``strain`` is a free-form biological sample descriptor (e.g.
        ``"pan-nuclear GFP"``). Orthogonal to role. None preserves the existing
        value on update.

        Position has two stages: coarse (bottom-camera / manual map placement)
        and fine (future SPIM-objective alignment). New callers should pass
        position_coarse / position_fine as dicts of shape {"x": float, "y":
        float}. Legacy callers passing position_x / position_y get folded into
        coarse automatically.
        """
        ed = self._embryo_dir(session_id, embryo_id)
        ed.mkdir(parents=True, exist_ok=True)

        # Fold legacy position_x / position_y into coarse if caller used the
        # old kwargs and didn't pass coarse explicitly.
        if position_coarse is None and (position_x is not None or position_y is not None):
            position_coarse = {}
            if position_x is not None:
                position_coarse["x"] = position_x
            if position_y is not None:
                position_coarse["y"] = position_y

        yaml_path = ed / "embryo.yaml"
        existing = _read_yaml(yaml_path)

        if existing is not None:
            # COALESCE update — keep existing values when new ones are None.
            existing_coarse = _coarse_from_legacy(existing)
            embryo_data = {
                "embryo_id": embryo_id,
                "session_id": session_id,
                "embryo_uid": embryo_uid if embryo_uid is not None else existing.get("embryo_uid"),
                "nickname": nickname if nickname is not None else existing.get("nickname"),
                "position_coarse": position_coarse
                if position_coarse is not None
                else existing_coarse,
                "position_fine": position_fine
                if position_fine is not None
                else existing.get("position_fine"),
                "calibration": calibration
                if calibration is not None
                else existing.get("calibration"),
                "role": role if role is not None else existing.get("role", "test"),
                "strain": strain if strain is not None else existing.get("strain"),
                "created_at": existing.get("created_at", _now()),
            }
        else:
            embryo_data = {
                "embryo_id": embryo_id,
                "session_id": session_id,
                "embryo_uid": embryo_uid,
                "nickname": nickname,
                "position_coarse": position_coarse,
                "position_fine": position_fine,
                "calibration": calibration,
                "role": role if role is not None else "test",
                "strain": strain,
                "created_at": _now(),
            }

        _write_yaml(yaml_path, embryo_data)

    def get_embryo(self, session_id: str, embryo_id: str) -> EmbryoInfo | None:
        """Read embryo.yaml. Returns None if not found.

        Backfills position_coarse from legacy position_x / position_y so
        callers don't need to know about the old schema.
        """
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        yaml_path = sd / "embryos" / embryo_id / "embryo.yaml"
        data = _read_yaml(yaml_path)
        return _normalize_embryo_record(data)

    def list_embryos(self, session_id: str) -> list[EmbryoInfo]:
        """List all embryos for a session, sorted by embryo_id."""
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        embryos_dir = sd / "embryos"
        if not embryos_dir.exists():
            return []

        result: list[EmbryoInfo] = []
        for entry in sorted(embryos_dir.iterdir()):
            if entry.is_dir():
                yaml_path = entry / "embryo.yaml"
                data = _read_yaml(yaml_path)
                if data is not None:
                    record = _normalize_embryo_record(data)
                    if record is not None:
                        result.append(record)
        return result

    def list_embryo_ids(self, session_id: str) -> list[str]:
        """Embryo IDs from directory names only — no ``embryo.yaml`` parse.

        The directory name *is* the embryo_id in this layout (see
        ``_embryo_dir`` / ``put_embryo``), so callers that only need the ids
        (e.g. enumerating projections) can skip the per-embryo YAML read that
        ``list_embryos`` pays.
        """
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        embryos_dir = sd / "embryos"
        if not embryos_dir.exists():
            return []
        return [e.name for e in sorted(embryos_dir.iterdir()) if e.is_dir()]

    # ==================================================================
    # Volumes
    # ==================================================================

    def put_volume(
        self,
        session_id: str,
        embryo_id: str,
        timepoint: int,
        volume: np.ndarray,
        metadata: dict | None = None,
    ) -> Path:
        """
        Write a volume to disk, generate a JPEG projection, write sidecar metadata.

        Parameters
        ----------
        session_id, embryo_id, timepoint
            Natural key for the volume.
        volume : np.ndarray
            Raw volume data (3D or 4D).
        metadata : dict, optional
            Extra metadata stored in the sidecar YAML.

        Returns
        -------
        Path
            Absolute path to the written TIFF file.
        """
        import tifffile

        vol_dir = self._volume_dir(session_id, embryo_id)
        vol_path = vol_dir / self._volume_filename(timepoint)

        tifffile.imwrite(str(vol_path), volume, compression="zlib")

        # Write sidecar metadata
        meta = {
            "session_id": session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "shape": list(volume.shape),
            "dtype": str(volume.dtype),
            "acquired_at": _now(),
            "metadata": metadata,
        }
        _write_yaml(vol_dir / self._volume_meta_filename(timepoint), meta)

        # Generate projection
        self._generate_projection(session_id, embryo_id, timepoint, volume)

        logger.debug("put_volume: %s/%s t=%d -> %s", session_id, embryo_id, timepoint, vol_path)
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
        Zero-copy path: move an existing TIFF to its canonical location.

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
            Canonical path after move.
        """
        incoming_path = Path(incoming_path)
        if not incoming_path.exists():
            raise FileNotFoundError(f"Incoming file not found: {incoming_path}")

        vol_dir = self._volume_dir(session_id, embryo_id)
        canonical = vol_dir / self._volume_filename(timepoint)

        # Move (rename if same drive, copy+delete otherwise)
        if canonical.exists():
            canonical.unlink()
        try:
            incoming_path.rename(canonical)
        except OSError:
            shutil.copy2(str(incoming_path), str(canonical))
            incoming_path.unlink()

        # Use caller-provided array or read from disk
        if volume_data is not None:
            volume = volume_data
        else:
            from .imaging import load_volume

            volume = load_volume(canonical)

        # Write sidecar metadata
        meta = {
            "session_id": session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "shape": list(volume.shape),
            "dtype": str(volume.dtype),
            "acquired_at": _now(),
            "metadata": metadata,
        }
        _write_yaml(vol_dir / self._volume_meta_filename(timepoint), meta)

        # Generate projection
        self._generate_projection(session_id, embryo_id, timepoint, volume)

        logger.debug("register_volume: %s -> %s", incoming_path.name, canonical)
        return canonical

    def get_volume(self, session_id: str, embryo_id: str, timepoint: int) -> np.ndarray | None:
        """Load a volume from disk.  Returns None if not found."""
        path = self.get_volume_path(session_id, embryo_id, timepoint)
        if path is None or not path.exists():
            return None
        import tifffile

        return tifffile.imread(str(path))

    def get_volume_path(self, session_id: str, embryo_id: str, timepoint: int) -> Path | None:
        """Return the absolute path to a volume TIFF, or None."""
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        vol_path = sd / "embryos" / embryo_id / "volumes" / self._volume_filename(timepoint)
        if vol_path.exists():
            return vol_path
        return None

    def get_volume_meta(self, session_id: str, embryo_id: str, timepoint: int) -> dict | None:
        """Read the sidecar metadata YAML for a volume.  Returns None if not found."""
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        meta_path = sd / "embryos" / embryo_id / "volumes" / self._volume_meta_filename(timepoint)
        return _read_yaml(meta_path)

    def list_volumes(self, session_id: str, embryo_id: str | None = None) -> list[VolumeInfo]:
        """List volume metadata by scanning sidecar YAML files on disk."""
        sd = self._session_dir(session_id)
        if sd is None:
            return []

        embryos_dir = sd / "embryos"
        if not embryos_dir.exists():
            return []

        # Determine which embryo dirs to scan
        if embryo_id:
            dirs = [embryos_dir / embryo_id]
        else:
            dirs = sorted(d for d in embryos_dir.iterdir() if d.is_dir())

        result: list[VolumeInfo] = []
        for edir in dirs:
            vol_dir = edir / "volumes"
            if not vol_dir.exists():
                continue
            for meta_file in sorted(vol_dir.glob("t*.meta.yaml")):
                data = _read_yaml(meta_file)
                if data is None:
                    continue
                # Build a VolumeInfo dict
                tif_path = meta_file.parent / meta_file.name.replace(".meta.yaml", ".tif")
                info: VolumeInfo = {
                    "session_id": data.get("session_id", session_id),
                    "embryo_id": data.get("embryo_id", edir.name),
                    "timepoint": data.get("timepoint", 0),
                    "file_path": str(tif_path),
                    "shape": data.get("shape"),
                    "dtype": data.get("dtype"),
                    "acquired_at": data.get("acquired_at", ""),
                    "metadata": data.get("metadata"),
                }
                result.append(info)

        # Sort by embryo_id then timepoint
        result.sort(key=lambda v: (v["embryo_id"], v["timepoint"]))
        return result

    def get_acquisition_params(self, session_id: str, embryo_id: str | None = None) -> dict | None:
        """
        Get acquisition parameters from the most recent volume sidecar.

        Returns the ``metadata`` field from the latest volume, which
        contains num_slices, exposure_ms, interval_seconds, calibration, etc.
        """
        volumes = self.list_volumes(session_id, embryo_id)
        # Walk backwards to find the first one with non-None metadata
        for vol in reversed(volumes):
            if vol.get("metadata") is not None:
                return vol["metadata"]
        return None

    # ==================================================================
    # Projections
    # ==================================================================

    def get_projection_path(self, session_id: str, embryo_id: str, timepoint: int) -> Path | None:
        """Return absolute path to the JPEG projection, or None."""
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        proj_path = (
            sd / "embryos" / embryo_id / "projections" / self._projection_filename(timepoint)
        )
        if proj_path.exists():
            return proj_path
        return None

    def list_projection_timepoints(self, session_id: str, embryo_id: str) -> list[int]:
        """Cheaply list projection timepoints (glob only, no PIL/meta reads).

        Used to rehydrate the viz image store on resume without paying the
        per-file cost of list_projections().
        """
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        proj_dir = sd / "embryos" / embryo_id / "projections"
        if not proj_dir.exists():
            return []
        tps: list[int] = []
        for jpg in proj_dir.glob("t*.jpg"):
            m = re.match(r"t(\d+)\.jpg$", jpg.name)
            if m:
                tps.append(int(m.group(1)))
        return sorted(tps)

    def get_projection_b64(self, session_id: str, embryo_id: str, timepoint: int) -> str | None:
        """Return base64-encoded JPEG projection, or None."""
        path = self.get_projection_path(session_id, embryo_id, timepoint)
        if path is None or not path.exists():
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def list_projections(self, session_id: str, embryo_id: str) -> list[ProjectionInfo]:
        """List projection info for an embryo by scanning projection files."""
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        proj_dir = sd / "embryos" / embryo_id / "projections"
        if not proj_dir.exists():
            return []

        result: list[ProjectionInfo] = []
        for jpg in sorted(proj_dir.glob("t*.jpg")):
            # Extract timepoint from filename  t0003.jpg -> 3
            match = re.match(r"t(\d+)\.jpg$", jpg.name)
            if not match:
                continue
            tp = int(match.group(1))

            # Read image dimensions
            width, height, size_kb = None, None, None
            try:
                from PIL import Image as PILImage

                img = PILImage.open(str(jpg))
                width, height = img.size
                size_kb = round(jpg.stat().st_size / 1024, 1)
            except Exception:
                pass

            # Use the volume sidecar's acquired_at as the projection created_at
            # if available; otherwise use the file mtime.
            meta_path = sd / "embryos" / embryo_id / "volumes" / self._volume_meta_filename(tp)
            vol_meta = _read_yaml(meta_path)
            created = (
                vol_meta.get("acquired_at", "")
                if vol_meta
                else datetime.fromtimestamp(jpg.stat().st_mtime).isoformat()
            )

            info: ProjectionInfo = {
                "session_id": session_id,
                "embryo_id": embryo_id,
                "timepoint": tp,
                "file_path": str(jpg),
                "width": width,
                "height": height,
                "size_kb": size_kb,
                "created_at": created,
            }
            result.append(info)
        return result

    # ==================================================================
    # Snapshots (bottom camera, etc.)
    # ==================================================================

    def register_snapshot(
        self,
        session_id: str,
        source: str,
        incoming_path: Path,
        metadata: dict | None = None,
    ) -> Path:
        """Move a transient TIFF from incoming/ to ``snapshots/``."""
        incoming_path = Path(incoming_path)
        if not incoming_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {incoming_path}")

        sd = self._require_session_dir(session_id)
        snap_dir = sd / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Use original stem (UUID) to avoid collisions
        canonical = snap_dir / f"{source}_{incoming_path.stem}.tif"

        try:
            incoming_path.rename(canonical)
        except OSError:
            shutil.copy2(str(incoming_path), str(canonical))
            incoming_path.unlink()

        # Write sidecar metadata
        sidecar: dict[str, Any] = {
            "session_id": session_id,
            "source": source,
            "file_path": str(canonical),
            "metadata": metadata,
            "captured_at": _now(),
        }
        # Read shape for the sidecar
        try:
            import tifffile

            arr = tifffile.imread(str(canonical))
            sidecar["width"] = int(arr.shape[-1]) if arr.ndim >= 2 else None
            sidecar["height"] = int(arr.shape[-2]) if arr.ndim >= 2 else None
        except Exception:
            sidecar["width"] = None
            sidecar["height"] = None

        _write_yaml(
            canonical.with_suffix(".meta.yaml"),
            sidecar,
        )

        logger.debug("register_snapshot: %s -> %s", incoming_path.name, canonical)
        return canonical

    def list_snapshots(self, session_id: str, source: str | None = None) -> list[dict[str, Any]]:
        """List snapshot records for a session, optionally filtered by source."""
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        snap_dir = sd / "snapshots"
        if not snap_dir.exists():
            return []

        result = []
        for meta_file in sorted(snap_dir.glob("*.meta.yaml")):
            data = _read_yaml(meta_file)
            if data is None:
                continue
            if source and data.get("source") != source:
                continue
            result.append(data)

        # Sort by captured_at
        result.sort(key=lambda s: s.get("captured_at", ""))
        return result

    # ==================================================================
    # Incoming cleanup
    # ==================================================================

    def cleanup_incoming(self, max_age_seconds: float = 300) -> int:
        """Delete stale files from the incoming staging directory.

        Files older than *max_age_seconds* (default 5 min) are assumed
        orphaned.

        Returns the number of files deleted.
        """
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

    # ==================================================================
    # Perception Runs & Predictions
    # ==================================================================

    def _perception_runs_path(self, session_id: str) -> Path:
        sd = self._require_session_dir(session_id)
        return sd / "perception_runs.yaml"

    def _load_perception_runs(self, session_id: str) -> dict[int, dict]:
        """Load perception_runs.yaml as {run_id: run_metadata}."""
        data = _read_yaml(self._perception_runs_path(session_id))
        if data is None:
            return {}
        # Ensure keys are ints
        return {int(k): v for k, v in data.items()}

    def _save_perception_runs(self, session_id: str, runs: dict[int, dict]) -> None:
        _write_yaml(self._perception_runs_path(session_id), runs)

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
        """Create a perception run.  Returns run_id (auto-increment)."""
        runs = self._load_perception_runs(session_id)

        # Auto-increment: next id is max existing + 1
        run_id = max(runs.keys(), default=0) + 1

        runs[run_id] = {
            "run_id": run_id,
            "session_id": session_id,
            "name": name,
            "perception_method": method,
            "model_name": model_name,
            "trace_type": trace_type,
            "source": source,
            "config": config,
            "status": "running",
            "created_at": _now(),
            "completed_at": None,
            "error_message": None,
        }
        self._save_perception_runs(session_id, runs)
        return run_id

    def complete_perception_run(
        self, run_id: int, status: str = "completed", error_message: str | None = None
    ) -> None:
        """Mark a perception run as completed or failed.

        Searches all sessions for the run_id since the caller may not
        provide a session_id.
        """
        for sid in self._index:
            runs = self._load_perception_runs(sid)
            if run_id in runs:
                runs[run_id]["status"] = status
                runs[run_id]["completed_at"] = _now()
                runs[run_id]["error_message"] = error_message
                self._save_perception_runs(sid, runs)
                return
        logger.warning("complete_perception_run: run_id %d not found", run_id)

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
        Append a prediction to predictions.jsonl and optionally write trace JSON.

        Returns
        -------
        int
            prediction_id (line number in the JSONL, 1-based).
        """
        now = _now()

        # Write trace file if provided
        trace_file = None
        if trace_data is not None:
            trace_dir = self._trace_dir(session_id, embryo_id)
            trace_path = trace_dir / f"t{timepoint:04d}.json"
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False, default=str)
            trace_file = str(trace_path)

        # Per-embryo prediction_id = previous max + 1. Derived from the LAST
        # record only (bounded tail read) rather than re-parsing the whole
        # predictions.jsonl on every append — ids stay sequential because we
        # only ever append in order.
        sd = self._require_session_dir(session_id)
        pred_path = sd / "embryos" / embryo_id / "predictions.jsonl"
        last = _last_jsonl_record(pred_path)
        prediction_id = (last.get("prediction_id", 0) + 1) if last else 1

        record: PredictionInfo = {
            "prediction_id": prediction_id,
            "run_id": run_id,
            "session_id": session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "predicted_stage": predicted_stage,
            "confidence": confidence,
            "reasoning": reasoning,
            "is_transitional": 1 if is_transitional else 0,
            "ground_truth_stage": ground_truth_stage,
            "is_correct": is_correct,
            "execution_time_ms": execution_time_ms,
            "trace_file": trace_file,
            "observed_features": observed_features,
            "created_at": now,
        }

        _append_jsonl(pred_path, record)
        return prediction_id

    def get_predictions(
        self,
        session_id: str,
        embryo_id: str | None = None,
        run_id: int | None = None,
    ) -> list[PredictionInfo]:
        """Query predictions with optional filters."""
        sd = self._session_dir(session_id)
        if sd is None:
            return []

        embryos_dir = sd / "embryos"
        if not embryos_dir.exists():
            return []

        # Determine which embryo dirs to read
        if embryo_id:
            dirs = [embryos_dir / embryo_id]
        else:
            dirs = sorted(d for d in embryos_dir.iterdir() if d.is_dir())

        result: list[PredictionInfo] = []
        for edir in dirs:
            pred_path = edir / "predictions.jsonl"
            records = _read_jsonl(pred_path)
            for rec in records:
                if run_id is not None and rec.get("run_id") != run_id:
                    continue
                result.append(cast("PredictionInfo", rec))

        # Sort by timepoint, then prediction_id
        result.sort(key=lambda p: (p.get("timepoint", 0), p.get("prediction_id", 0)))
        return result

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
    ) -> None:
        """Insert or update a ground-truth annotation."""
        ed = self._embryo_dir(session_id, embryo_id)
        gt_path = ed / "ground_truth.yaml"
        entries: list = _read_yaml(gt_path) or []

        now = _now()

        # Check for existing entry matching (session_id, embryo_id, stage)
        # -- mirrors the UNIQUE(session_id, embryo_id, stage) constraint
        found = False
        for entry in entries:
            if entry.get("stage") == stage:
                entry["start_timepoint"] = start_timepoint
                entry["end_timepoint"] = end_timepoint
                entry["annotator"] = annotator
                entry["notes"] = notes
                found = True
                break

        if not found:
            # Auto-increment id
            max_id = max((e.get("id", 0) for e in entries), default=0)
            entries.append(
                {
                    "id": max_id + 1,
                    "session_id": session_id,
                    "embryo_id": embryo_id,
                    "stage": stage,
                    "start_timepoint": start_timepoint,
                    "end_timepoint": end_timepoint,
                    "annotator": annotator,
                    "notes": notes,
                    "created_at": now,
                }
            )

        _write_yaml(gt_path, entries)

    def get_ground_truth(self, session_id: str, embryo_id: str) -> list[GroundTruthEntry]:
        """Get ground-truth annotations sorted by start_timepoint."""
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        gt_path = sd / "embryos" / embryo_id / "ground_truth.yaml"
        entries: list = _read_yaml(gt_path) or []
        entries.sort(key=lambda e: e.get("start_timepoint", 0))
        return entries

    # ==================================================================
    # Utility
    # ==================================================================

    def stats(self) -> StoreStats:
        """Return counts and disk-usage summary."""
        n_sessions = len(self._index)
        n_embryos = 0
        n_volumes = 0
        n_projections = 0
        n_perception_runs = 0
        n_predictions = 0
        n_ground_truth = 0

        for sid in self._index:
            sd = self._session_dir(sid)
            if sd is None or not sd.exists():
                continue

            embryos_dir = sd / "embryos"
            if embryos_dir.exists():
                for edir in embryos_dir.iterdir():
                    if not edir.is_dir():
                        continue
                    n_embryos += 1

                    # Count volumes
                    vol_dir = edir / "volumes"
                    if vol_dir.exists():
                        n_volumes += len(list(vol_dir.glob("t*.tif")))

                    # Count projections
                    proj_dir = edir / "projections"
                    if proj_dir.exists():
                        n_projections += len(list(proj_dir.glob("t*.jpg")))

                    # Count predictions
                    pred_path = edir / "predictions.jsonl"
                    if pred_path.exists():
                        n_predictions += len(_read_jsonl(pred_path))

                    # Count ground truth
                    gt_path = edir / "ground_truth.yaml"
                    gt = _read_yaml(gt_path)
                    if gt:
                        n_ground_truth += len(gt)

            # Count perception runs
            runs = self._load_perception_runs(sid)
            n_perception_runs += len(runs)

        # Disk usage (approximate)
        total_bytes = 0
        for subdir in ("sessions", "incoming", "logs"):
            d = self._root / subdir
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size

        return {
            "sessions": n_sessions,
            "embryos": n_embryos,
            "volumes": n_volumes,
            "projections": n_projections,
            "perception_runs": n_perception_runs,
            "predictions": n_predictions,
            "ground_truth": n_ground_truth,
            "disk_usage_mb": round(total_bytes / (1024 * 1024), 1),
            "db_size_mb": 0.0,  # No database
        }

    def close(self) -> None:
        """No-op.  No database to close."""
        logger.info("FileStore closed (no-op)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"FileStore(root={self._root})"
