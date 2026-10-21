"""An intention, said in a sentence, drafted into an imaging spec.

A plan item added by hand is a sentence: "image an exposure ramp to find where
the LED saturates". The rig runs from fields and tactics. Two passes make the
sentence actionable:

- ``parse_intent`` reads what the sentence states outright, deterministically
  and offline, and remembers the words each field came from.
- ``draft_spec_with_model`` asks Gently to reason the rest out, the way a
  collaborator would: what kind of acquisition this is, how many embryos,
  which light, what to sweep, what counts as done. It returns a draft with a
  one-line rationale and its assumptions. The draft is a proposal; nothing
  runs on it until a person accepts it.

Every filled field carries its provenance: ``sentence``, ``model``, ``typed``
or ``accepted``. A person's field is never replaced by a draft.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip
_WAVELENGTHS = {"405", "445", "488", "514", "532", "561", "594", "640", "647"}
_STAGES = (
    "bean", "comma", "1.5-fold", "1.5fold", "2-fold", "2fold", "3-fold", "pretzel",
    "hatching", "twitching",
)  # fmt: skip
_STAGE_CANON = {"1.5-fold": "1.5fold", "2-fold": "2fold", "3-fold": "pretzel"}
PERSON_SOURCES = ("typed", "accepted")


def _num(tok: str) -> float | None:
    tok = tok.strip().lower()
    if tok in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[tok])
    try:
        return float(tok)
    except ValueError:
        return None


def _seconds(value: float, unit: str) -> int:
    unit = unit.lower()
    if unit.startswith("h"):
        return int(round(value * 3600))
    if unit.startswith("m") and unit != "ms":
        return int(round(value * 60))
    return int(round(value))


def parse_intent(text: str) -> dict[str, Any]:
    """The fields a sentence states, with the words they came from.

    Returns ``{"spec": {...}, "provenance": {field: {"source": "sentence",
    "text": words}}}``; ``spec`` holds only what was found.
    """
    spec: dict[str, Any] = {}
    prov: dict[str, dict[str, str]] = {}
    s = " ".join((text or "").split())
    low = s.lower()

    def keep(field: str, value: Any, words: str) -> None:
        if value is None or field in spec:
            return
        spec[field] = value
        prov[field] = {"source": "sentence", "text": words.strip()}

    number = r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

    m = re.search(number + r"\s+(?:test\s+|control\s+)?embryos?\b", low)
    if m:
        n = _num(m.group(1))
        keep("num_embryos", int(n) if n is not None else None, m.group(0))

    m = re.search(
        r"every\s+" + number + r"\s*(ms|s|sec|secs|seconds?|min|mins|minutes?|h|hr|hrs|hours?)\b",
        low,
    )
    if not m:
        m = re.search(
            number + r"\s*(s|sec|secs|seconds?|min|mins|minutes?)\s+(?:cadence|interval)", low
        )
    if m:
        n = _num(m.group(1))
        if n is not None and m.group(2) != "ms":
            keep("interval_s", _seconds(n, m.group(2)), m.group(0))

    waves = [w for w in re.findall(r"\b(\d{3})\s*(?:nm)?\b", low) if w in _WAVELENGTHS]
    if waves:
        keep("laser_wavelength_nm", int(waves[0]), " + ".join(dict.fromkeys(waves)))

    m = re.search(
        r"(?:until|till|to|at)\s+(" + "|".join(re.escape(x) for x in _STAGES) + r")\b", low
    )
    if m:
        keep("stop_condition", _STAGE_CANON.get(m.group(1), m.group(1)), m.group(0))
    m = re.search(number + r"\s+timepoints?\b", low)
    if m and "stop_condition" not in spec:
        n = _num(m.group(1))
        if n is not None:
            keep("stop_condition", f"{int(n)} timepoints", m.group(0))
    m = re.search(r"\bfor\s+" + number + r"\s*(h|hr|hrs|hours?|min|mins|minutes?)\b", low)
    if m:
        n = _num(m.group(1))
        if n is not None:
            keep("estimated_duration_h", round(_seconds(n, m.group(2)) / 3600, 2), m.group(0))
            if "stop_condition" not in spec:
                keep("stop_condition", f"after {m.group(1)} {m.group(2)}", m.group(0))

    m = re.search(
        r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*(?:°\s*c|°c|º\s*c|degrees(?:\s+c(?:elsius)?)?|\bc\b)(?![a-z])",
        low,
    )
    if m:
        keep("temperature_c", float(m.group(1)), m.group(0))

    m = re.search(r"(\d+)\s*(?:z[- ]?)?slices?\b", low)
    if m:
        keep("num_slices", int(m.group(1)), m.group(0))
    m = re.search(r"(\d+(?:\.\d+)?)\s*ms\b(?:\s+exposure)?", low)
    if m:
        keep("exposure_ms", float(m.group(1)), m.group(0))
    m = re.search(r"(\d+(?:\.\d+)?)\s*%(?:\s+(?:laser\s+)?power)?", low)
    if m:
        keep("laser_power_pct", float(m.group(1)), m.group(0))

    m = re.search(r"\b([a-z]{2,5}-\d+[a-z]?(?:p|::)\S*)", s, re.I)
    if m:
        keep("reporter", m.group(1).rstrip(",.;"), m.group(0))
    for tok in re.findall(r"\b([A-Z]{1,3}\d{1,5})\b", s):
        digits = tok.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if digits in _WAVELENGTHS and len(tok) <= 4 and not tok.startswith("N"):
            continue  # "488" and friends are light, not strains
        keep("strain", tok, tok)
        break

    # What the sentence is for: the clause after "to check/see/find out…" is
    # the success criterion, the one field an intention always has.
    m = re.search(
        r"\bto\s+(?:check|see|find(?:\s+out)?|test|measure|determine|learn|know|ask|compare|verify|confirm)\s+(.+?)(?:[.;]|$)",
        s,
        re.I,
    )
    if m:
        keep("success_criteria", m.group(1).strip(), m.group(0))

    return {"spec": spec, "provenance": prov}


# ---------------------------------------------------------------------------
# Gently drafts the rest
# ---------------------------------------------------------------------------

# ponytail: a fixed card of what the instrument can do. The hardware profile
# (config/hardware.yaml) is where this belongs once it carries these numbers.
INSTRUMENT_FACTS = (
    "Instrument: dual-view light-sheet (diSPIM) for C. elegans embryos on an agar pad. "
    "Fluorescence excitation lasers at 488 nm and 561 nm, power set in percent; typical "
    "light-sheet exposures 5 to 50 ms; volumes of 40 to 120 z slices. A bottom camera "
    "takes brightfield frames of the whole field under an LED whose power is set in percent "
    "(default 1 %), typical exposures 1 to 200 ms; brightfield frames are 16-bit, so "
    "saturation is pixels at or near 65535. Embryos are tracked by id (e01, e02, …) and "
    "staged: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched. A standing "
    "timelapse images every embryo each round at a cadence; a scripted protocol runs a "
    "list of steps once (a sweep, a calibration); a oneshot takes a single acquisition."
)

DRAFT_TOOL = {
    "name": "draft_imaging_spec",
    "description": (
        "Draft an actionable imaging spec from the researcher's intention. Reason out what "
        "kind of acquisition it is and fill `acquisition` in the shape the Devices › Operate "
        "pane runs. Use only what the intention states or clearly implies, plus the instrument "
        "facts; list every assumption. Leave unknowable fields out rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "a short title for the card, at most eight words",
            },
            "item_type": {
                "type": "string",
                "enum": ["imaging", "analysis", "bench", "genetics", "decision_point"],
            },
            "rationale": {
                "type": "string",
                "description": "one sentence: what this is and why these settings",
            },
            "num_embryos": {"type": "integer", "description": "how many embryos the run images"},
            "acquisition": {
                "type": "object",
                "description": "what the pane runs; omit any key the intention does not settle",
                "properties": {
                    "cadence_s": {"type": "integer", "description": "seconds between rounds"},
                    "volumes": {
                        "type": "boolean",
                        "description": "false for a brightfield-only run",
                    },
                    "num_slices": {"type": "integer"},
                    "exposure_ms": {"type": "number", "description": "SPIM exposure per slice"},
                    "laser_config": {
                        "type": "string",
                        "description": "the lines routed, '488', '561' or '488+561'",
                    },
                    "laser_powers": {
                        "type": "object",
                        "description": 'percent per line, e.g. {"488": 3}',
                        "additionalProperties": {"type": "number"},
                    },
                    "bf": {
                        "type": "object",
                        "description": "the brightfield overview channel on the bottom camera",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "every_seconds": {"type": "number"},
                            "exposure_ms": {"type": "number"},
                            "light": {"type": "string", "enum": ["led", "room", "none"]},
                            "led_intensity_pct": {"type": "integer"},
                        },
                    },
                    "stop_condition": {
                        "type": "string",
                        "description": (
                            "manual | timepoints:N | duration:Nh | hatching | comma | "
                            "all_test_hatched"
                        ),
                    },
                    "monitoring_mode": {"type": "string"},
                },
            },
            "temperature_c": {"type": "number"},
            "strain": {"type": "string"},
            "reporter": {"type": "string"},
            "start_stage": {"type": "string"},
            "estimated_duration_h": {"type": "number"},
            "target_window": {"type": "string"},
            "sample_prep": {"type": "string"},
            "success_criteria": {
                "type": "string",
                "description": "what the data must show for the intention to be met, checkable",
            },
            "detectors": {"type": "array", "items": {"type": "string"}},
            "tactics": {
                "type": "array",
                "description": (
                    "the tactic outline the run executes, when it is not one standing timelapse"
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "standing_timelapse",
                                "reactive_monitor",
                                "scripted_protocol",
                                "exclusive_burst",
                                "oneshot",
                                "custom",
                            ],
                        },
                        "name": {"type": "string"},
                        "target": {
                            "type": "string",
                            "description": (
                                "which embryos or field, e.g. 'all', 'e01', 'empty field'"
                            ),
                        },
                        "scope": {
                            "type": "string",
                            "description": "brightfield | fluorescence | both",
                        },
                        "structure": {
                            "type": "object",
                            "description": (
                                "the steps or settings, e.g. "
                                "{steps: [{exposure_ms: 1}, …], channel: 'brightfield'}"
                            ),
                        },
                    },
                    "required": ["kind", "name"],
                },
            },
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "item_type", "rationale", "assumptions"],
    },
}

_SYSTEM = (
    "You are Gently, the microscope's agent, turning a researcher's intention into an "
    "imaging spec the rig can run. Reason like a collaborator: decide what kind of "
    "acquisition the intention needs, choose settings inside the instrument's limits, and "
    "write a success criterion that can be checked on the data. Use only what the "
    "intention states or clearly implies, the project's context and the instrument facts. "
    "Do not invent strains or biology. Name every assumption. A sweep or calibration is a "
    "scripted_protocol tactic with explicit steps; a developmental recording is a "
    "standing_timelapse with a cadence and a stop condition. The acquisition itself goes in "
    "`acquisition`, the shape the Devices › Operate pane runs: cadence_s; volumes false for a "
    "brightfield-only run; num_slices and exposure_ms for the SPIM volume; laser_config as the "
    "lines it routes ('488', '561', '488+561') with laser_powers per line; bf for the brightfield "
    "overview (enabled, every_seconds, exposure_ms, light led|room|none, led_intensity_pct); "
    "stop_condition as manual, timepoints:N, duration:Nh, hatching, comma or all_test_hatched."
)


def project_context(cs, campaign_id: str | None) -> str:
    """What the project already says, for the draft to rest on."""
    if cs is None or not campaign_id:
        return "No project context."
    try:
        root_id = cs.root_campaign_id(campaign_id) or campaign_id
        c = cs.get_campaign(root_id)
    except Exception:
        c = None
    if c is None:
        return "No project context."
    bits = [f"Project: {c.description}"]
    if c.target:
        bits.append(f"Target: {c.target}")
    strains = getattr(c, "strains", None) or []
    if strains:
        bits.append(
            "Strains: "
            + "; ".join(
                f"{s.get('key')} ({s.get('subtitle') or s.get('title') or ''})".strip()
                for s in strains[:6]
            )
        )
    try:
        items = cs.get_plan_items(campaign_id=root_id, include_children=True)
        done = [i for i in items if i.status.value == "completed" and i.outcome]
        if done:
            bits.append(
                "Measured so far: " + " | ".join(f"{i.title}: {i.outcome}" for i in done[:4])
            )
    except Exception:
        pass
    return "\n".join(bits)


async def draft_spec_with_model(
    client,
    model: str,
    text: str,
    *,
    known: dict[str, Any] | None = None,
    context: str = "",
    note: str | None = None,
) -> dict[str, Any]:
    """Ask the model for the draft. ``known`` is what the sentence stated
    outright, so the draft does not contradict it; ``note`` is a constraint
    the researcher added when asking again."""
    known_line = ", ".join(f"{k}={v}" for k, v in (known or {}).items()) or "nothing yet"
    user = (
        f"Intention: {text}\n\n"
        f"Read from the sentence already: {known_line}.\n"
        f"{context}\n\n{INSTRUMENT_FACTS}\n\n"
        + (f"The researcher adds: {note}\n\n" if note else "")
        + "Draft the spec and tactics; leave out what cannot be known."
    )
    resp = await client.messages.create(
        model=model,
        max_tokens=900,
        system=_SYSTEM,
        tools=[DRAFT_TOOL],
        tool_choice={"type": "tool", "name": DRAFT_TOOL["name"]},
        messages=[{"role": "user", "content": user}],
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            data = dict(block.input or {})
            meta = {
                "title": data.pop("title", None),
                "item_type": data.pop("item_type", None),
                "rationale": data.pop("rationale", None),
                "assumptions": [str(a) for a in data.pop("assumptions", []) or []],
            }
            fields = {k: v for k, v in data.items() if v not in (None, "", [], {})}
            return {
                "spec": fields,
                "provenance": {k: {"source": "model", "text": "drafted by Gently"} for k in fields},
                **meta,
            }
    return {"spec": {}, "provenance": {}, "title": None, "item_type": None, "rationale": None,
            "assumptions": ["the model returned no draft"]}  # fmt: skip


def merge_spec(
    existing: dict[str, Any] | None, *layers: dict[str, Any], overwrite: bool = False
) -> dict[str, Any]:
    """Lay drafts under what is already there.

    A field a person typed or accepted is kept unless ``overwrite``. A field the
    sentence stated is kept over a model draft, since the sentence is the
    researcher's. A previous model draft gives way to a new one. Provenance is
    carried per field.
    """
    out: dict[str, Any] = dict(existing or {})
    prov: dict[str, dict[str, str]] = dict(out.get("provenance") or {})
    for layer in layers:
        layer_prov = layer.get("provenance") or {}
        for field, value in (layer.get("spec") or {}).items():
            current = out.get(field)
            have = current not in (None, "", [], {})
            source = (prov.get(field) or {}).get("source")
            incoming = (layer_prov.get(field) or {}).get("source", "sentence")
            if have and not overwrite:
                if source in PERSON_SOURCES or source is None:
                    continue  # a person's value
                if source == "sentence" and incoming == "model":
                    continue  # the sentence outranks the draft
            out[field] = value
            prov[field] = dict(layer_prov.get(field) or {"source": "sentence", "text": ""})
    out["provenance"] = prov
    return out


def draft_record(status: str, **extra: Any) -> dict[str, Any]:
    """The ``draft`` block on a spec: drafting | drafted | unavailable | accepted."""
    rec: dict[str, Any] = {"status": status, "at": datetime.now().isoformat(timespec="seconds")}
    rec.update({k: v for k, v in extra.items() if v is not None})
    return rec
