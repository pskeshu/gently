"""What a calibration looked at, kept with the embryo it calibrated.

A calibration takes sixty to eighty exposures, scores each, fits two focus
curves and draws what it concluded. The conclusion was stored: a slope, an
offset, two R². The evidence was not. Every frame, plot and montage went to
the browser's image store, in memory, and was gone at the next restart. A
volume's geometry rested on numbers nobody could look behind.

One folder per run, under the embryo::

    embryos/{embryo_id}/calibration/{YYYYMMDD_HHMMSS}/
        calibration.yaml          what was asked, what came of it, the fit
        frames.jsonl              one line per image: what it is, its score
        frames/NNN_{kind}_....tif the exposures, as they came off the camera
        plots/NNN_{kind}.png      the focus curves, the montage, the summary

A run is recorded whether it succeeds, is refused, fails or is aborted: the
ones that did not work are the ones somebody will want to look at.

Recording never fails a calibration. An image that could not be written is a
warning in the log; the routine carries on.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# What is kept: every frame and plot, the plots alone, or nothing.
ALL, PLOTS, NONE = "all", "plots", "none"
KEEP = (ALL, PLOTS, NONE)

# The kinds calibration shows that are drawings, not exposures.
PLOT_KINDS = {"focus_plot", "focus_montage", "calibration_summary"}

_SAFE = re.compile(r"[^A-Za-z0-9._+-]+")

# The prefix of a Windows extended-length path, built rather than written so
# that no layer between here and the disk gets to reinterpret a backslash.
_BS = chr(92)
_EXTENDED = _BS + _BS + "?" + _BS
_EXTENDED_UNC = _EXTENDED + "UNC" + _BS


def long_path(path: Path | str) -> Path:
    """``path``, in the form Windows accepts past 260 characters.

    A run's images sit seven folders under the storage root and their names
    say what they are, so a deep root pushes them past MAX_PATH, and every
    write then fails with "No such file or directory". The extended form has
    no such limit. Elsewhere, and for a path already in that form, this is
    the path unchanged.
    """
    import os

    p = Path(path)
    if os.name != "nt":
        return p
    text = str(p)
    if text.startswith(_EXTENDED):
        return p
    text = os.path.abspath(text)
    if text.startswith(_BS + _BS):  # a share
        return Path(_EXTENDED_UNC + text[2:])
    return Path(_EXTENDED + text)


def short_path(path: Path | str) -> Path:
    """The ordinary form of a path: what an operator reads, and what a file
    manager is given."""
    text = str(path)
    if text.startswith(_EXTENDED_UNC):
        return Path(_BS + _BS + text[len(_EXTENDED_UNC) :])
    if text.startswith(_EXTENDED):
        return Path(text[len(_EXTENDED) :])
    return Path(text)


def _slug(text: Any) -> str:
    return _SAFE.sub("-", str(text)).strip("-")[:60] or "image"


def _plain(value: Any) -> Any:
    """Metadata as YAML and JSON can hold it."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def keep_policy() -> str:
    """all | plots | none, from the rig's settings."""
    try:
        from gently.settings import settings

        policy = str(getattr(settings.storage, "calibration_images", ALL)).lower()
    except Exception:
        policy = ALL
    return policy if policy in KEEP else ALL


class CalibrationRecord:
    """One run's evidence, filed as it is produced."""

    def __init__(
        self,
        folder: Path,
        *,
        session_id: str,
        embryo_id: str,
        requested: dict[str, Any] | None = None,
        keep: str = ALL,
    ) -> None:
        self.folder = long_path(folder)
        self.session_id = session_id
        self.embryo_id = embryo_id
        self.keep = keep if keep in KEEP else ALL
        self.requested = _plain(requested or {})
        self.started_at = datetime.now()
        self._n = 0
        self._frames = 0
        self._plots = 0
        self._lock = threading.Lock()
        self._closed = False
        self.folder.mkdir(parents=True, exist_ok=True)
        self._write_head("running", None, None)

    # ── adding ──────────────────────────────────────────────────────────
    def add(
        self,
        array: Any,
        kind: str,
        metadata: dict[str, Any] | None = None,
        uid: str | None = None,
    ) -> Path | None:
        """File one image. Returns where it went, or None when it was not
        kept (the policy says no, the run is closed, or the write failed)."""
        if self._closed or self.keep == NONE or array is None:
            return None
        is_plot = kind in PLOT_KINDS
        if self.keep == PLOTS and not is_plot:
            return None
        try:
            arr = np.asarray(array)
            if arr.ndim < 2:
                return None
            meta = _plain(metadata or {})
            with self._lock:
                self._n += 1
                n = self._n
                name = self._name(n, kind, meta, is_plot)
                sub = "plots" if is_plot else "frames"
                target = self.folder / sub / name
                target.parent.mkdir(parents=True, exist_ok=True)
                self._write_image(target, arr, is_plot)
                if is_plot:
                    self._plots += 1
                else:
                    self._frames += 1
                line = {
                    "n": n,
                    "at": datetime.now().isoformat(timespec="milliseconds"),
                    "kind": kind,
                    "file": f"{sub}/{name}",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "uid": uid,
                    **meta,
                }
                with open(self.folder / "frames.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            return target
        except Exception as exc:
            logger.warning(
                "calibration image (%s) for %s could not be kept: %s", kind, self.embryo_id, exc
            )
            return None

    @staticmethod
    def _name(n: int, kind: str, meta: dict, is_plot: bool) -> str:
        bits = [f"{n:03d}", _slug(kind)]
        if meta.get("sweep"):
            bits.append(_slug(meta["sweep"]))
        if meta.get("galvo_name"):
            bits.append(_slug(meta["galvo_name"]))
        if not is_plot:
            if isinstance(meta.get("galvo"), (int, float)):
                bits.append(f"g{meta['galvo']:+.3f}")
            if isinstance(meta.get("piezo"), (int, float)):
                bits.append(f"p{meta['piezo']:+.1f}")
        return "_".join(bits) + (".png" if is_plot else ".tif")

    @staticmethod
    def _write_image(target: Path, arr: np.ndarray, is_plot: bool) -> None:
        if is_plot:
            from PIL import Image

            img = arr
            if img.dtype != np.uint8:
                lo, hi = float(np.min(img)), float(np.max(img))
                img = ((img - lo) / (hi - lo + 1e-12) * 255).astype(np.uint8)
            Image.fromarray(img).save(target, format="PNG")
            return
        import tifffile

        # As it came off the camera: no scaling, so a score can be recomputed.
        tifffile.imwrite(str(target), arr, compression="zlib")

    # ── closing ─────────────────────────────────────────────────────────
    def finish(
        self,
        outcome: str,
        *,
        message: str | None = None,
        calibration: dict[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        self._write_head(outcome, message, calibration)

    def _write_head(self, outcome: str, message: str | None, calibration: dict | None) -> None:
        try:
            import yaml

            doc = {
                "session_id": self.session_id,
                "embryo_id": self.embryo_id,
                "started_at": self.started_at.isoformat(timespec="seconds"),
                "finished_at": (
                    None if outcome == "running" else datetime.now().isoformat(timespec="seconds")
                ),
                "outcome": outcome,
                "message": (message or "")[:2000] or None,
                "requested": self.requested,
                "calibration": _plain(calibration) if calibration else None,
                "kept": self.keep,
                "frames": self._frames,
                "plots": self._plots,
            }
            with open(self.folder / "calibration.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
        except Exception as exc:
            logger.warning(
                "calibration record for %s could not be written: %s", self.embryo_id, exc
            )

    @property
    def counts(self) -> dict[str, int]:
        return {"frames": self._frames, "plots": self._plots}


def read_record(folder: Path) -> dict[str, Any] | None:
    """One record's head, with where it is."""
    head = long_path(folder) / "calibration.yaml"
    if not head.exists():
        return None
    try:
        import yaml

        with open(head, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    doc["run"] = Path(folder).name
    doc["folder"] = str(short_path(folder))
    return doc


def read_frames(folder: Path) -> list[dict[str, Any]]:
    path = long_path(folder) / "frames.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except ValueError:
                continue
    return out
