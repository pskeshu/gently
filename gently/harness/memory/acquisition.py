"""The acquisition side of a plan item, in the shape Devices › Operate runs.

One shape in three places: the ``structure`` a saved tactic carries, what the
start route keeps with the session as ``acquisition.yaml``, and
``ImagingSpec.acquisition``. Its keys::

    cadence_s, interval        seconds between rounds (both, for older readers)
    volumes                    False is a brightfield-only run
    sides                      the imaging paths a round takes, ["A"] or ["A", "B"]
    num_slices, exposure_ms    the SPIM volume, when volumes run
    laser_config               the preset, named by the lines it routes ("488", "488+561")
    laser_powers               {"488": pct, "561": pct}
    bf                         the brightfield overview: {enabled, every_seconds,
                               position | positions, exposure_ms,
                               light: led | room | none, led_intensity_pct} or None
    stop_condition             the orchestrator's spec string: manual | timepoints:N |
                               duration:Nh | hatching | comma | all_test_hatched | stages(x)
    condition_value, monitoring_mode, stop_conditions

The older flat fields of ``ImagingSpec`` (interval_s, laser_wavelength_nm, a
stop word) are read into this shape by ``acquisition_from_legacy``; an explicit
``acquisition`` on the spec wins over them (``effective_acquisition``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

STRUCTURE_KEYS = (
    "cadence_s",
    "interval",
    "volumes",
    "sides",
    "num_slices",
    "exposure_ms",
    "laser_config",
    "laser_powers",
    "bf",
    "stop_condition",
    "condition_value",
    "monitoring_mode",
    "stop_conditions",
)
PANE_STOPS = ("manual", "timepoints", "duration", "hatching", "comma", "all_test_hatched")
STAGE_WORDS = (
    "early",
    "bean",
    "comma",
    "1.5fold",
    "2fold",
    "pretzel",
    "hatching",
    "hatched",
    "twitching",
)


def stop_spec_from_words(text: Any) -> str | None:
    """A stop condition said in words, as the orchestrator's spec string.
    Unknown words give None: no ending is invented."""
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    if s in PANE_STOPS or s.startswith(("timepoints:", "duration:", "stages(", "stage(")):
        return s
    m = re.match(r"^(?:after\s+)?(\d+)\s*timepoints?$", s)
    if m:
        return f"timepoints:{int(m.group(1))}"
    m = re.match(r"^(?:after\s+|for\s+)?(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?)$", s)
    if m:
        return f"duration:{float(m.group(1)):g}h"
    m = re.match(r"^(?:after\s+|for\s+)?(\d+(?:\.\d+)?)\s*(min|mins|minutes?)$", s)
    if m:
        return f"duration:{float(m.group(1)) / 60:g}h"
    if s in ("single pass", "one pass", "once", "single", "one frame", "one timepoint"):
        return "timepoints:1"
    if s == "hatched":
        return "hatching"
    if s in STAGE_WORDS:
        return f"stages({s})"
    return None


def acquisition_from_legacy(spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """The flat fields a June-era spec carried, in the pane's shape. Only what
    is there; no defaults, so a reader can tell what was never said."""
    spec = spec or {}
    out: dict[str, Any] = {}
    interval = spec.get("interval_s")
    if interval:
        out["cadence_s"] = out["interval"] = int(round(float(interval)))
    for key in ("num_slices", "exposure_ms", "monitoring_mode"):
        if spec.get(key) is not None:
            out[key] = spec[key]
    wl, pct = spec.get("laser_wavelength_nm"), spec.get("laser_power_pct")
    if wl and pct is not None:
        # The power goes on the line; the preset is the controller's to name,
        # so none is set here and the pane keeps whatever it has.
        out["laser_powers"] = {str(int(wl)): float(pct)}
    stop = stop_spec_from_words(spec.get("stop_condition"))
    if stop:
        out["stop_condition"] = stop
    return out


def structure_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """What the start route was sent, as the structure the session keeps."""
    volumes = payload.get("volumes") is not False
    interval = payload.get("interval_seconds", payload.get("cadence_s", 30))
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        interval = 30.0
    powers = payload.get("laser_powers") or {}
    out: dict[str, Any] = {
        "cadence_s": interval,
        "interval": interval,
        "stop_condition": str(payload.get("stop_condition") or "manual"),
        "condition_value": payload.get("condition_value"),
        "monitoring_mode": (payload.get("monitoring_mode") or "idle") if volumes else "idle",
        "num_slices": payload.get("num_slices") if volumes else None,
        "exposure_ms": payload.get("exposure_ms") if volumes else None,
        "laser_config": (payload.get("laser_config") or None) if volumes else None,
        "laser_powers": {str(k): v for k, v in dict(powers).items()}
        if volumes and powers
        else None,
        "bf": payload.get("bf") if isinstance(payload.get("bf"), dict) else None,
        "volumes": volumes,
        "stop_conditions": payload.get("stop_conditions") or None,
    }
    return out


def effective_acquisition(spec: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The acquisition a spec means: its explicit ``acquisition`` laid over what
    its older flat fields say. None when neither says anything."""
    spec = spec or {}
    merged = acquisition_from_legacy(spec)
    explicit = spec.get("acquisition")
    if isinstance(explicit, Mapping):
        given = {k: v for k, v in explicit.items() if v is not None}
        merged.update(given)
        # One cadence, two names for older readers: whichever the explicit
        # acquisition gave sets both.
        if "cadence_s" in given and "interval" not in given:
            merged["interval"] = given["cadence_s"]
        elif "interval" in given and "cadence_s" not in given:
            merged["cadence_s"] = given["interval"]
    if "cadence_s" in merged and "interval" not in merged:
        merged["interval"] = merged["cadence_s"]
    if "interval" in merged and "cadence_s" not in merged:
        merged["cadence_s"] = merged["interval"]
    return merged or None


def _interval_words(seconds: float) -> str:
    s = float(seconds)
    if s >= 3600 and s % 3600 == 0:
        return f"{s / 3600:g} h"
    if s >= 60 and s % 60 == 0:
        return f"{s / 60:g} min"
    return f"{s:g} s"


def stop_words(spec: Any, unit: str = "timepoint") -> str:
    s = str(spec or "manual").strip().lower()
    if s.startswith("timepoints:"):
        n = int(float(s.split(":", 1)[1] or 1))
        return f"after {n} {unit}{'' if n == 1 else 's'}"
    if s.startswith("duration:"):
        return f"after {s.split(':', 1)[1].rstrip('h')} h"
    m = re.match(r"^stages?\(([^)]*)\)$", s)
    if m:
        return "at " + m.group(1).split(",")[0].strip()
    return {
        "manual": "until stopped",
        "hatching": "at hatching",
        "comma": "at comma stage",
        "all_test_hatched": "when every test embryo has hatched",
    }.get(s, f"at {s}")


def describe_acquisition(acq: Mapping[str, Any] | None, n_embryos: int | None = None) -> str:
    """The plan in the pane's sentence, for the agent and the server. The
    browser's `AcquisitionPlan.describe` is the same sentence with the roster."""
    if not acq:
        return ""
    every = _interval_words(acq.get("cadence_s") or acq.get("interval") or 30)
    bf = acq.get("bf") if isinstance(acq.get("bf"), Mapping) else None
    light = (bf or {}).get("light") or "led"
    lit = {
        "led": "under the LED",
        "room": "under the room light",
        "none": "in the light as it is",
    }.get(light, light)
    if bf is not None and light == "led" and bf.get("led_intensity_pct") is not None:
        lit += f" at {bf['led_intensity_pct']} %"
    if acq.get("volumes") is False:
        exp = f" ({bf['exposure_ms']} ms)" if bf and bf.get("exposure_ms") is not None else ""
        return (
            f"Every {every}: one brightfield frame of the field on the bottom camera{exp}, {lit} · "
            f"{stop_words(acq.get('stop_condition'), 'frame')}. No SPIM volumes."
        )
    bits = [
        f"{acq['num_slices']} slices"
        if acq.get("num_slices") is not None
        else "slices as each embryo has",
        f"{acq['exposure_ms']:g} ms"
        if acq.get("exposure_ms") is not None
        else "exposure as each embryo has",
    ]
    if acq.get("laser_config"):
        bits.append(str(acq["laser_config"]))
    for wl, pct in (acq.get("laser_powers") or {}).items():
        bits.append(f"{wl} at {pct:g} %")
    who = (
        "no embryos"
        if not n_embryos
        else ("1 embryo" if n_embryos == 1 else f"{n_embryos} embryos")
    )
    s = f"Every {every}: SPIM volumes ({' · '.join(bits)}) of {who}"
    if bf and bf.get("enabled"):
        cadence = acq.get("cadence_s") or acq.get("interval") or 30
        rounds = max(1, int(round((bf.get("every_seconds") or cadence) / cadence)))
        every_rounds = "per round" if rounds == 1 else f"every {rounds} rounds"
        s += f" + one brightfield overview {every_rounds}, {lit}"
    return s + f" · {stop_words(acq.get('stop_condition'))}."
