"""Persistence for the launch gate's two toggles (+ advanced defaults).

The launch gate answers exactly two questions — *microscope hardware on/off* and
*AI agent on/off* — and everything else (device port, SAM device) is a remembered
default behind "Advanced options". Those choices live in
``config/launch.local.json`` (gitignored) so the gate is prefilled every boot.

See ``docs/superpowers/specs/2026-07-02-unified-launcher-design.md`` (RFC #78).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gently.settings import settings

logger = logging.getLogger(__name__)

# Repo root: gently/ui/web/launch_prefs.py -> parents[3]
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
PREFS_PATH = _CONFIG_DIR / "launch.local.json"

# Only these keys are read from / written to disk. Advanced values fall back to
# settings so a fresh install has sensible defaults without a prefs file.
_DEFAULTS: dict = {
    "hardware": True,  # start + connect the device layer
    "agent": True,  # enable chat / perception / planning (needs API key)
    "port": settings.network.device_port,
    # "auto" is resolved to cuda/cpu at load time by GPU auto-detection — a
    # biologist should never have to choose (RFC #78). Only a scope wrangler
    # pins a concrete value (in Settings).
    "sam_device": "auto",
}
_ALLOWED_KEYS = set(_DEFAULTS)

_sam_device_cache: str | None = None


def detect_sam_device() -> str:
    """Auto-detect the SAM inference device: ``cuda`` if an NVIDIA GPU is present,
    else ``cpu``. Cached — detection is cheap once torch is loaded."""
    global _sam_device_cache
    if _sam_device_cache is not None:
        return _sam_device_cache
    dev = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            dev = "cuda"
    except Exception:
        # torch missing/failed — fall back to a plain nvidia-smi presence check.
        import shutil
        import subprocess

        if shutil.which("nvidia-smi"):
            try:
                subprocess.run(["nvidia-smi"], capture_output=True, timeout=4, check=True)
                dev = "cuda"
            except Exception:
                pass
    _sam_device_cache = dev
    logger.info("SAM device auto-detected: %s", dev)
    return dev


def stored_prefs() -> dict:
    """The raw persisted prefs (unresolved), or {} — for the Settings UI which
    needs to show 'auto' vs a pinned SAM device."""
    try:
        if PREFS_PATH.exists():
            data = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return {}


def load_prefs() -> dict:
    """Return the persisted launch choices merged over defaults.

    Never raises — a missing or corrupt file just yields the defaults, so the
    gate always renders.
    """
    prefs = dict(_DEFAULTS)
    try:
        if PREFS_PATH.exists():
            stored = json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                prefs.update({k: stored[k] for k in _ALLOWED_KEYS if k in stored})
    except (OSError, ValueError) as e:
        logger.warning("launch prefs unreadable (%s) — using defaults", e)
    # Resolve the SAM device unless a concrete value was pinned in Settings.
    if prefs.get("sam_device") in (None, "", "auto"):
        prefs["sam_device"] = detect_sam_device()
    return prefs


def save_prefs(prefs: dict) -> dict:
    """Persist a (partial) set of launch choices; returns the merged result.

    Unknown keys are ignored; known keys are coerced to the expected types.
    """
    merged = load_prefs()
    for key in _ALLOWED_KEYS & set(prefs):
        merged[key] = _coerce(key, prefs[key])
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    except OSError as e:
        logger.error("could not persist launch prefs: %s", e)
    return merged


def _coerce(key: str, value):
    """Coerce a JSON-supplied value to the type its default implies."""
    default = _DEFAULTS[key]
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return value
