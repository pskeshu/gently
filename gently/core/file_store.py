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
        acquisition.yaml                    # the plan the run was started with
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
            calibration/
              {YYYYMMDD_HHMMSS}/            # one calibration run's evidence
                calibration.yaml
                frames.jsonl
                frames/NNN_{kind}_....tif
                plots/NNN_{kind}.png
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

_INVALID_PATH_COMPONENT_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _validate_path_component(value: str, label: str) -> str:
    """Validate a caller-controlled value before using it as one path part."""
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    if value.strip() != value:
        raise ValueError(f"{label} must not start or end with whitespace: {value!r}")
    if value in {".", ".."} or _INVALID_PATH_COMPONENT_RE.search(value):
        raise ValueError(f"{label} contains unsafe path characters: {value!r}")
    if value.rstrip(" .") != value:
        raise ValueError(f"{label} must not end with a space or dot: {value!r}")
    return value


def _safe_child_path(parent: Path, component: str, label: str) -> Path:
    """Return parent/component after validating it stays below parent."""
    component = _validate_path_component(component, label)
    base = parent.resolve()
    child = (parent / component).resolve()
    try:
        child.relative_to(base)
    except ValueError:
        raise ValueError(f"{label} escapes storage root: {component!r}") from None
    return child


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

    Never constructs Python objects from YAML. Legacy files containing
    ``!!python/object`` or numpy constructor tags must be migrated by a
    trusted offline tool before FileStore will read them — we refuse to
    unsafe_load them, since a YAML file on disk is not a trust boundary
    (it may be synced, imported, or attacker-supplied).
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
        raise ValueError(
            f"Refusing to load unsafe YAML tags from {path}. "
            "Migrate this file to safe YAML before opening it in Gently."
        ) from err


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
        return self._embryo_dir_for_session(sd, embryo_id)

    def _embryo_dir_for_session(self, session_dir: Path, embryo_id: str) -> Path:
        """Resolve session_dir/embryos/<embryo_id>, rejecting traversal."""
        return _safe_child_path(session_dir / "embryos", embryo_id, "embryo_id")

    # ==================================================================
    # Calibration records
    # ==================================================================

    def calibration_dir(self, session_id: str, embryo_id: str) -> Path:
        """embryos/<embryo_id>/calibration, not created."""
        return self._embryo_dir(session_id, embryo_id) / "calibration"

    def open_calibration_record(
        self, session_id: str, embryo_id: str, requested: dict | None = None
    ):
        """A new folder for one calibration run, and the record that fills it.

        The folder is named for when the run started. Two runs in one second
        get distinct folders rather than sharing one.
        """
        from gently.core.calibration_record import CalibrationRecord, keep_policy

        base = self.calibration_dir(session_id, embryo_id)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = base / stamp
        n = 1
        while folder.exists():
            n += 1
            folder = base / f"{stamp}_{n}"
        return CalibrationRecord(
            folder,
            session_id=session_id,
            embryo_id=embryo_id,
            requested=requested,
            keep=keep_policy(),
        )

    def list_calibration_records(
        self, session_id: str, embryo_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Every calibration run recorded in the session, oldest first."""
        from gently.core.calibration_record import read_record

        sd = self._session_dir(session_id)
        if sd is None:
            return []
        root = sd / "embryos"
        if not root.exists():
            return []
        if embryo_id is not None:
            embryo_dirs = [self._embryo_dir_for_session(sd, embryo_id)]
        else:
            embryo_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        out: list[dict[str, Any]] = []
        for ed in embryo_dirs:
            cal = ed / "calibration"
            if not cal.is_dir():
                continue
            for run in sorted(p for p in cal.iterdir() if p.is_dir()):
                rec = read_record(run)
                if rec is not None:
                    out.append(rec)
        out.sort(key=lambda r: str(r.get("started_at") or ""))
        return out

    def calibration_record_dir(self, session_id: str, embryo_id: str, run: str) -> Path | None:
        """The folder of one recorded run, or None. ``run`` is matched against
        the folders that exist, never joined into a path."""
        base = self.calibration_dir(session_id, embryo_id)
        if not base.is_dir():
            return None
        return next((p for p in base.iterdir() if p.is_dir() and p.name == run), None)

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

    def mark_advanced_diagnostics(self, session_id: str, since: str) -> bool:
        """Record on the session that it was recorded with Advanced diagnostics
        on, from ``since``. Set once, never cleared. Returns True if written.
        """
        sd = self._session_dir(session_id)
        if sd is None or not sd.exists():
            return False
        yaml_path = sd / "session.yaml"
        data = _read_yaml(yaml_path)
        if data is None:
            return False
        meta = data.get("metadata")
        if not isinstance(meta, dict):
            meta = data["metadata"] = {}
        if meta.get("advanced_diagnostics"):
            return False
        meta["advanced_diagnostics"] = True
        meta["advanced_diagnostics_since"] = since
        _write_yaml(yaml_path, data)
        return True

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

    # ==================================================================
    # Acquisition plan
    # ==================================================================

    def save_acquisition_plan(self, session_id: str, plan: dict) -> Path:
        """Write ``acquisition.yaml``: the plan a run was started with.

        The same structure a saved template and a seeded standing_timelapse
        tactic carry, so the Acquisition pane can read it back on resume.
        """
        sd = self._require_session_dir(session_id)
        path = sd / "acquisition.yaml"
        _write_yaml(path, dict(plan))
        return path

    def get_acquisition_plan(self, session_id: str) -> dict | None:
        """The plan the session's run was started with, or None."""
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        doc = _read_yaml(sd / "acquisition.yaml")
        return doc if isinstance(doc, dict) else None

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
        yaml_path = self._embryo_dir_for_session(sd, embryo_id) / "embryo.yaml"
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

    # ------------------------------------------------------------------
    # Removing an embryo
    #
    # The x beside an embryo is for a false positive, and it sits one row
    # from the embryo that has been imaged all night. It used to delete the
    # folder: volumes, projections, traces and calibration, with no way back.
    # Nothing here deletes. A removed embryo's folder is moved, whole, to
    #
    #     <session>/removed/{embryo_id}__{YYYYMMDD_HHMMSS}/
    #
    # beside a removed.yaml saying what it held. It is out of every listing,
    # so it does not come back on a restart, and it can be put back.
    # ------------------------------------------------------------------

    _REMOVED = "removed"
    _REMOVED_RECORD = "removed.yaml"

    def _removed_dir(self, session_dir: Path) -> Path:
        return session_dir / self._REMOVED

    @staticmethod
    def _embryo_holdings(embryo_dir: Path) -> dict[str, Any]:
        """What an embryo's folder holds, counted from its directory names."""

        def count(sub: str, pattern: str) -> int:
            d = embryo_dir / sub
            return sum(1 for _ in d.glob(pattern)) if d.is_dir() else 0

        cal = embryo_dir / "calibration"
        return {
            "timepoints": count("volumes", "t*.tif"),
            "projections": count("projections", "t*.jpg"),
            "calibration_runs": sum(1 for p in cal.iterdir() if p.is_dir()) if cal.is_dir() else 0,
        }

    def embryo_holdings(self, session_id: str, embryo_id: str) -> dict[str, Any]:
        """What would be set aside with this embryo. Zeros if it has no folder."""
        sd = self._session_dir(session_id)
        if sd is None:
            return {"timepoints": 0, "projections": 0, "calibration_runs": 0}
        return self._embryo_holdings(self._embryo_dir_for_session(sd, embryo_id))

    def set_aside_embryo(
        self, session_id: str, embryo_id: str, by: Any = None
    ) -> dict[str, Any] | None:
        """Move an embryo's folder out of the session's embryos, whole.

        Returns the record of what was set aside, or None if the embryo has no
        folder. Raises ``OSError`` if the folder cannot be moved, which on
        Windows is what happens while one of its files is open: the move is a
        rename, so it is all of the folder or none of it.
        """
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        ed = self._embryo_dir_for_session(sd, embryo_id)
        if not ed.exists():
            return None
        record: dict[str, Any] = {
            "embryo_id": embryo_id,
            "removed_at": datetime.now().isoformat(timespec="seconds"),
            "removed_by": by,
            **self._embryo_holdings(ed),
        }
        info = _read_yaml(ed / "embryo.yaml") or {}
        calibration = info.get("calibration") or {}
        record["calibrated"] = bool(calibration.get("slope_um_per_deg"))

        removed = self._removed_dir(sd)
        removed.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = removed / f"{ed.name}__{stamp}"
        n = 1
        while target.exists():
            n += 1
            target = removed / f"{ed.name}__{stamp}_{n}"
        os.rename(ed, target)
        record["folder"] = target.name
        try:
            _write_yaml(target / self._REMOVED_RECORD, record)
        except Exception:
            # The folder's name says which embryo and when. The record is a
            # convenience, and failing to write it must not undo the move.
            logger.warning("Could not write %s in %s", self._REMOVED_RECORD, target, exc_info=True)
        return record

    def delete_embryo(self, session_id: str, embryo_id: str) -> bool:
        """Kept for its callers. Deletes nothing: see ``set_aside_embryo``."""
        return self.set_aside_embryo(session_id, embryo_id) is not None

    def list_removed_embryos(self, session_id: str) -> list[dict[str, Any]]:
        """The embryos set aside in this session, the latest removal first."""
        sd = self._session_dir(session_id)
        if sd is None:
            return []
        removed = self._removed_dir(sd)
        if not removed.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for entry in removed.iterdir():
            if not entry.is_dir() or "__" not in entry.name:
                continue
            record = _read_yaml(entry / self._REMOVED_RECORD) or {}
            embryo_id, _, stamp = entry.name.partition("__")
            record.setdefault("embryo_id", embryo_id)
            if not record.get("removed_at"):
                try:
                    record["removed_at"] = datetime.strptime(stamp[:15], "%Y%m%d_%H%M%S").isoformat(
                        timespec="seconds"
                    )
                except ValueError:
                    record["removed_at"] = None
            for key, value in self._embryo_holdings(entry).items():
                record.setdefault(key, value)
            record["folder"] = entry.name
            out.append(record)
        out.sort(key=lambda r: (str(r.get("removed_at") or ""), r["folder"]), reverse=True)
        return out

    def restore_embryo(
        self, session_id: str, embryo_id: str, folder: str | None = None
    ) -> dict[str, Any] | None:
        """Put a removed embryo's folder back. The latest removal of that
        embryo, unless ``folder`` names another.

        Returns its record, or None if nothing of that embryo was set aside.
        Raises ``FileExistsError`` if the session has an embryo of that id.
        """
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        found = [
            r
            for r in self.list_removed_embryos(session_id)
            if r["embryo_id"] == embryo_id and (folder is None or r["folder"] == folder)
        ]
        if not found:
            return None
        record = found[0]
        # The folder's name comes from the listing, never from the request.
        source = self._removed_dir(sd) / record["folder"]
        ed = self._embryo_dir_for_session(sd, embryo_id)
        if ed.exists():
            raise FileExistsError(f"{embryo_id} is in the session already")
        ed.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, ed)
        try:
            (ed / self._REMOVED_RECORD).unlink()
        except OSError:
            pass
        return record

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
        vol_path = (
            self._embryo_dir_for_session(sd, embryo_id)
            / "volumes"
            / self._volume_filename(timepoint)
        )
        if vol_path.exists():
            return vol_path
        return None

    def get_volume_meta(self, session_id: str, embryo_id: str, timepoint: int) -> dict | None:
        """Read the sidecar metadata YAML for a volume.  Returns None if not found."""
        sd = self._session_dir(session_id)
        if sd is None:
            return None
        meta_path = (
            self._embryo_dir_for_session(sd, embryo_id)
            / "volumes"
            / self._volume_meta_filename(timepoint)
        )
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
            dirs = [self._embryo_dir_for_session(sd, embryo_id)]
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
            self._embryo_dir_for_session(sd, embryo_id)
            / "projections"
            / self._projection_filename(timepoint)
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
        proj_dir = self._embryo_dir_for_session(sd, embryo_id) / "projections"
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
        proj_dir = self._embryo_dir_for_session(sd, embryo_id) / "projections"
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
            meta_path = (
                self._embryo_dir_for_session(sd, embryo_id)
                / "volumes"
                / self._volume_meta_filename(tp)
            )
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

    def put_snapshot(
        self,
        session_id: str,
        source: str,
        image: Any,
        metadata: dict | None = None,
        stem: str | None = None,
    ) -> Path:
        """File a snapshot from its pixels, for a frame that has no staged
        file to move. Same place and same sidecar as ``register_snapshot``."""
        import uuid

        import tifffile

        sd = self._require_session_dir(session_id)
        snap_dir = sd / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        canonical = snap_dir / f"{source}_{stem or uuid.uuid4().hex[:12]}.tif"
        arr = np.asarray(image)
        tifffile.imwrite(str(canonical), arr)
        sidecar: dict[str, Any] = {
            "session_id": session_id,
            "source": source,
            "file_path": str(canonical),
            "metadata": metadata,
            "captured_at": _now(),
            "width": int(arr.shape[-1]) if arr.ndim >= 2 else None,
            "height": int(arr.shape[-2]) if arr.ndim >= 2 else None,
        }
        _write_yaml(canonical.with_suffix(".meta.yaml"), sidecar)
        logger.debug("put_snapshot: %s", canonical)
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
        pred_path = self._embryo_dir_for_session(sd, embryo_id) / "predictions.jsonl"
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
            dirs = [self._embryo_dir_for_session(sd, embryo_id)]
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
        gt_path = self._embryo_dir_for_session(sd, embryo_id) / "ground_truth.yaml"
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
