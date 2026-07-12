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
    "sam_device": "cuda",
}
_ALLOWED_KEYS = set(_DEFAULTS)


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
