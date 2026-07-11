"""
Strategy snapshot builder for the Experiment overview tab.

Reads ``session.yaml``, ``timelapse.yaml``, ``embryos/{eid}/embryo.yaml`` and
replays ``timeline.jsonl`` to reconstruct the dynamic history the frontend
swimlane view needs (phase boundaries, trigger firings, power history).

This is a pure read-only view over the on-disk state. The live orchestrator
saves its state every acquisition so the data is at most one-acquisition
stale; the viz server doesn't need direct access to the running
orchestrator object.

Output shape mirrors the ``STUB_STRATEGY`` literal in
``static/js/experiment-overview.js`` — see that file for the contract the
frontend expects.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Role -> (color, single-glyph icon). Mirrors gently.harness.roles.REGISTRY's
# ui_icon names but resolves them to actual unicode glyphs the swimlane SVG
# can render directly.
_ROLE_ICONS = {
    "star": "★",  # ★
    "diamond": "◆",  # ◆
    "circle": "●",  # ●
    "triangle": "▲",  # ▲
}

# Default per-timepoint exposure when nothing on disk tells us otherwise.
_DEFAULT_PER_TP_MS = 500.0

# Default initial laser power when no power-changed events exist yet.
_DEFAULT_INITIAL_POWER_PCT = 5.0

# Cluster two trigger_fired events of the same rule on the same embryo if
# they're within this many seconds. Reduces visual noise from power ramps
# that step every acquisition.
_TRIGGER_CLUSTER_GAP_S = 600.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _pick_timelapse_yaml(session_dir: Path, legacy_session_dir: Path) -> dict:
    """Choose the freshest timelapse.yaml between new and legacy paths.

    New layout writes into the FileStore-indexed timestamped folder. The
    legacy layout (pre path-fix) wrote into ``<sessions>/<bare_id>/``. If
    both exist we take whichever has the larger ``saved_at`` so an
    orchestrator that hasn't yet been restarted (still writing to legacy)
    isn't shadowed by a stale new-path file.
    """
    candidates: list[Path] = [
        session_dir / "timelapse.yaml",
        legacy_session_dir / "timelapse.yaml",
    ]
    docs = []
    for p in candidates:
        d = _read_yaml(p)
        if d:
            docs.append((p, d))
    if not docs:
        return {}
    if len(docs) == 1:
        return docs[0][1]

    # Pick by saved_at if present, falling back to file mtime.
    def _saved_at_key(item):
        path, doc = item
        s = doc.get("saved_at")
        if isinstance(s, str):
            t = _parse_iso(s)
            if t is not None:
                return t.timestamp()
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    docs.sort(key=_saved_at_key, reverse=True)
    return docs[0][1]


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _elapsed_s(t: datetime | None, started_at: datetime) -> float | None:
    if t is None:
        return None
    return (t - started_at).total_seconds()


# ---------------------------------------------------------------------------
# Per-embryo accumulator
# ---------------------------------------------------------------------------


@dataclass
class _EmbryoAccum:
    """Mutable accumulator while replaying timeline events for one embryo."""

    eid: str
    phases: list[dict]
    trigger_events: list[dict]
    power_history_488: list[dict]

    def open_phase(
        self, mode: str, start_s: float, cadence_s: float | None = None, **extra
    ) -> None:
        # If the last phase has no end yet, close it at start_s.
        if self.phases:
            last = self.phases[-1]
            if "end" not in last or last["end"] is None:
                last["end"] = start_s
        ph: dict[str, Any] = {"mode": mode, "start": start_s, "end": None}
        if cadence_s is not None:
            ph["cadence_s"] = cadence_s
        ph.update(extra)
        self.phases.append(ph)

    def close_open_phase(self, at_s: float) -> None:
        if self.phases:
            last = self.phases[-1]
            if last.get("end") is None:
                last["end"] = at_s


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_strategy_snapshot(
    session_dir: Path,
    session_id: str,
    *,
    horizon_padding_s: float = 1800.0,
) -> dict[str, Any]:
    """Read the session folder and return the strategy dict the frontend wants.

    Parameters
    ----------
    session_dir : Path
        Absolute path to ``D:/Gently3/sessions/{folder}/``.
    session_id : str
        The logical session id (key in ``sessions/_index.yaml``).
    horizon_padding_s : float
        How much projected-future time to leave past ``now`` so the SVG
        always has room for the projected continuation bar. Default 30 min.

    Legacy compatibility
    --------------------
    Older builds of the orchestrator wrote ``timelapse.yaml`` to a parallel
    folder ``<root>/sessions/<bare_session_id>/`` and merged every session's
    events into one global ``<root>/sessions/timeline.jsonl``. This builder
    falls back to those locations so a currently-running session whose
    process predates the path fix can still be visualized. See
    ``_pick_timelapse_yaml`` and ``_replay_timeline``.
    """
    legacy_session_dir = session_dir.parent / session_id

    session_yaml = _read_yaml(session_dir / "session.yaml") or {}
    timelapse_yaml = _pick_timelapse_yaml(session_dir, legacy_session_dir)

    # ---- Session metadata -------------------------------------------------
    session_name = session_yaml.get("name") or session_id
    # Prefer the timelapse's started_at (more accurate for what's on the
    # swimlane); fall back to session creation.
    started_at = (
        _parse_iso(timelapse_yaml.get("started_at"))
        or _parse_iso(session_yaml.get("created_at"))
        or datetime.now()
    )
    now = datetime.now()
    now_offset_s = max(0.0, (now - started_at).total_seconds())

    base_interval_s = float(timelapse_yaml.get("base_interval_seconds") or 120.0)
    dose_budget_base_ms = timelapse_yaml.get("dose_budget_base_ms")
    # per_timepoint_ms is a session-wide approximation. The per-embryo
    # exposure × num_slices is the truth but the frontend only uses this
    # for the "dose budget" footer hint; the per-embryo dose_used_ms is
    # what drives the gauge.
    per_timepoint_ms = _DEFAULT_PER_TP_MS

    # ---- Monitoring modes -------------------------------------------------
    monitoring_modes = _build_monitoring_modes(
        timelapse_yaml.get("active_monitoring_modes") or [],
    )

    # ---- Triggers (rule catalog) -----------------------------------------
    # First read embryo roles so the rule applies_to (which carries embryo
    # ids) can be resolved to role names for the chips.
    embryo_roles = _read_embryo_roles(session_dir)
    triggers = _build_triggers(
        interval_rules=timelapse_yaml.get("interval_rules") or [],
        power_rules=timelapse_yaml.get("power_rules") or [],
        embryo_roles=embryo_roles,
    )

    # ---- Embryos: static fields from timelapse.yaml + embryo.yaml --------
    tl_embryos = timelapse_yaml.get("embryos") or {}
    embryo_dicts = _build_embryos_static(
        session_dir=session_dir,
        tl_embryos=tl_embryos,
        embryo_roles=embryo_roles,
        dose_budget_base_ms=dose_budget_base_ms,
        base_interval_s=base_interval_s,
        started_at=started_at,
        now_offset_s=now_offset_s,
    )

    # ---- Replay timeline.jsonl -> dynamic history -----------------------
    _replay_timeline(
        session_dir=session_dir,
        legacy_session_dir=legacy_session_dir,
        session_id=session_id,
        embryo_dicts=embryo_dicts,
        triggers=triggers,
        started_at=started_at,
        now_offset_s=now_offset_s,
        base_interval_s=base_interval_s,
    )

    # ---- Project current cadence + dose exhaustion forward ---------------
    _project_forward(
        embryo_dicts=embryo_dicts,
        now_offset_s=now_offset_s,
        per_timepoint_ms=per_timepoint_ms,
    )

    # Horizon = either 2× elapsed, the latest projected_end + padding, or a
    # floor of 4h. Whichever is largest, capped at 24h so the SVG stays
    # readable.
    horizon_s = _compute_horizon(now_offset_s, embryo_dicts, horizon_padding_s)

    return {
        "session_id": session_id,
        "session_name": session_name,
        "started_at": started_at.isoformat(),
        "now_offset_s": now_offset_s,
        "horizon_s": horizon_s,
        "base_interval_s": base_interval_s,
        "dose_budget_base_ms": float(dose_budget_base_ms)
        if dose_budget_base_ms is not None
        else None,
        "per_timepoint_ms": per_timepoint_ms,
        "monitoring_modes": monitoring_modes,
        "triggers": triggers,
        "embryos": embryo_dicts,
    }


# ---------------------------------------------------------------------------
# Monitoring modes
# ---------------------------------------------------------------------------


def _build_monitoring_modes(mode_names: list[str]) -> list[dict]:
    """Resolve each active monitoring mode name into a serialized dict.

    The orchestrator persists only the names in ``timelapse.yaml``; we
    instantiate them from the MONITORING_MODES factory to recover
    description / applies_to_roles / params.
    """
    if not mode_names:
        return []
    try:
        from gently.app.orchestration.monitoring_modes import MONITORING_MODES
    except Exception:
        logger.debug("Could not import MONITORING_MODES; skipping mode resolution")
        return []

    out: list[dict] = []
    for name in mode_names:
        factory = MONITORING_MODES.get(name)
        if factory is None:
            out.append(
                {
                    "name": name,
                    "description": "",
                    "applies_to_roles": [],
                    "params": {},
                }
            )
            continue
        try:
            mode = factory()
        except Exception as e:
            logger.debug("Could not instantiate mode %s: %s", name, e)
            continue
        # Pull declarative knobs (fast_interval, rampdown_*) off the instance.
        excluded = {"name", "description", "applies_to_roles"}
        params = {
            k: v for k, v in vars(mode).items() if not k.startswith("_") and k not in excluded
        }
        out.append(
            {
                "name": mode.name,
                "description": mode.description,
                "applies_to_roles": list(mode.applies_to_roles),
                "params": params,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Triggers (rule catalog)
# ---------------------------------------------------------------------------


def _build_triggers(
    *,
    interval_rules: list[dict],
    power_rules: list[dict],
    embryo_roles: dict[str, str],
) -> list[dict]:
    triggers: list[dict] = []
    for r in interval_rules:
        triggers.append(
            {
                "id": r["name"],
                "kind": "interval_rule",
                "label": _humanize_rule_name(r["name"]),
                "when_text": _interval_when_text(r),
                "then_text": _interval_then_text(r),
                "applies_to": _resolve_applies_to_roles(r.get("applies_to"), embryo_roles),
                "one_time": bool(r.get("one_time", True)),
            }
        )
    for r in power_rules:
        triggers.append(
            {
                "id": r["name"],
                "kind": "power_rule",
                "label": _humanize_rule_name(r["name"]),
                "when_text": _power_when_text(r),
                "then_text": _power_then_text(r),
                "applies_to": _resolve_applies_to_roles(r.get("applies_to"), embryo_roles),
                "one_time": bool(r.get("one_time", False)),
            }
        )
    return triggers


def _humanize_rule_name(name: str) -> str:
    # "test_onset_speedup" -> "test onset speedup"
    return name.replace("_", " ")


def _interval_when_text(r: dict) -> str:
    parts = []
    if r.get("trigger_stage"):
        parts.append(f"stage = {r['trigger_stage']}")
    if r.get("trigger_detector"):
        parts.append(f"detector = {r['trigger_detector']}")
    return " AND ".join(parts) or "(no predicate)"


def _interval_then_text(r: dict) -> str:
    return f"interval -> {r.get('new_interval_seconds', '?')}s"


def _power_when_text(r: dict) -> str:
    parts = []
    if r.get("trigger_intensity_levels"):
        parts.append("intensity = " + "/".join(r["trigger_intensity_levels"]))
    if r.get("trigger_stage"):
        parts.append(f"stage = {r['trigger_stage']}")
    if r.get("trigger_detector"):
        parts.append(f"detector = {r['trigger_detector']}")
    return " AND ".join(parts) or "(no predicate)"


def _power_then_text(r: dict) -> str:
    arrow = "down" if r.get("direction", "down") == "down" else "up"
    step = r.get("step_pct", 1.0)
    floor = r.get("floor_pct", 2.0)
    ceiling = r.get("ceiling_pct", 6.0)
    wavelength = r.get("wavelength", 488)
    bound = f"floor {floor}%" if arrow == "down" else f"ceiling {ceiling}%"
    return f"{wavelength}nm {arrow} {step}%/step, {bound}"


def _resolve_applies_to_roles(
    applies_to: list[str] | None,
    embryo_roles: dict[str, str],
) -> list[str]:
    """``applies_to`` is a list of embryo ids; resolve to a deduplicated
    list of role names for the chips. ``None`` means "all roles in the
    timelapse".
    """
    if applies_to is None:
        return sorted(set(embryo_roles.values())) or ["all"]
    roles = []
    seen = set()
    for eid in applies_to:
        role = embryo_roles.get(eid, "unassigned")
        if role not in seen:
            seen.add(role)
            roles.append(role)
    return roles or ["all"]


# ---------------------------------------------------------------------------
# Embryo static fields
# ---------------------------------------------------------------------------


def _read_embryo_roles(session_dir: Path) -> dict[str, str]:
    """Map embryo_id -> role by scanning ``embryos/*/embryo.yaml``.

    We read this from the durable per-embryo file rather than timelapse.yaml
    so the role is correct even when the embryo isn't in the active
    timelapse (yet).
    """
    out: dict[str, str] = {}
    embryos_dir = session_dir / "embryos"
    if not embryos_dir.is_dir():
        return out
    for child in embryos_dir.iterdir():
        if not child.is_dir():
            continue
        data = _read_yaml(child / "embryo.yaml")
        if not data:
            continue
        out[child.name] = data.get("role") or "unassigned"
    return out


def _stop_condition_from_serialized(d: Any) -> tuple[str, str]:
    """Read the per-embryo stop_condition dict and return ``(spec, kind)``.

    ``kind`` is ``"bounded"`` when ANY component of the (possibly composite)
    condition is auto-stopping (stage-based / duration / fixed timepoints /
    all-test-hatched), and ``"open_ended"`` only when every component is
    manual. Mirrors what the user thinks of as bounded vs open-ended on
    the swimlane.
    """
    if not isinstance(d, dict):
        return "manual", "open_ended"
    spec = d.get("spec") or "manual"
    types: list[str] = []
    if d.get("condition_type"):
        types.append(d["condition_type"])
    for ad in d.get("additional") or []:
        if ad.get("condition_type"):
            types.append(ad["condition_type"])
    auto_stop = any(t != "manual" for t in types)
    return spec, ("bounded" if auto_stop else "open_ended")


def _build_embryos_static(
    *,
    session_dir: Path,
    tl_embryos: dict[str, dict],
    embryo_roles: dict[str, str],
    dose_budget_base_ms: float | None,
    base_interval_s: float,
    started_at: datetime | None = None,
    now_offset_s: float | None = None,
) -> list[dict]:
    """Build the per-embryo static portion of the snapshot.

    Dynamic fields (phases, trigger_events, power_history_488) are seeded
    with their initial values here and filled in by ``_replay_timeline``.
    """
    try:
        from gently.harness.roles import REGISTRY as ROLE_REGISTRY
    except Exception:
        ROLE_REGISTRY = {}

    out: list[dict] = []
    # Sort embryo ids so the snapshot is deterministic.
    for eid in sorted(tl_embryos.keys()):
        ed = tl_embryos[eid] or {}
        role = ed.get("role") or embryo_roles.get(eid) or "unassigned"
        role_def = ROLE_REGISTRY.get(role)
        color = role_def.ui_color if role_def else "#888"
        icon = _ROLE_ICONS.get(role_def.ui_icon if role_def else "circle", "●")
        mult = role_def.photodose_budget_multiplier if role_def else 1.0
        dose_budget_ms = (
            float(dose_budget_base_ms) * float(mult) if dose_budget_base_ms is not None else 0.0
        )
        laser_488 = ed.get("laser_power_488_pct")
        if laser_488 is None:
            laser_488 = _DEFAULT_INITIAL_POWER_PCT

        initial_cadence = float(ed.get("interval_seconds") or base_interval_s)
        # stop_condition is now serialized into timelapse.yaml; resolve
        # the user-visible spec string and the bounded/open-ended kind
        # from it. Legacy saves that predate the field fall back to
        # manual / open-ended.
        stop_spec, stop_kind = _stop_condition_from_serialized(ed.get("stop_condition"))
        # Seed: one base-cadence phase from t=0 until now (replay will
        # split it as cadence_changed events come in).
        out.append(
            {
                "id": eid,
                "role": role,
                "color": color,
                "icon": icon,
                "dose_used_ms": float(ed.get("total_exposure_ms") or 0.0),
                "dose_budget_ms": dose_budget_ms,
                "tp_acquired": int(ed.get("timepoints_acquired") or 0),
                "stop_condition": stop_spec,
                "stop_kind": stop_kind,
                "laser_488_pct_now": float(laser_488),
                "phases": [
                    {
                        "mode": "base",
                        "start": 0.0,
                        "end": None,
                        "cadence_s": initial_cadence,
                    }
                ],
                "trigger_events": [],
                "power_history_488": [{"at": 0.0, "pct": float(laser_488)}],
                # Filled in by _replay_timeline when temperature-tactic events arrive.
                "temp_protocol": None,
                "setpoint_changes": [],
                # Filled in by _project_forward.
                "projected_cadence_s": initial_cadence,
                "projected_end_s": None,
                # When the embryo was marked complete/terminated, as seconds
                # from session start. Null while still acquiring. The
                # frontend uses this to draw a TERMINATED cap and stop the
                # projection bar; without it, a finished embryo's row would
                # appear to still be acquiring forever.
                "terminated_at_s": _terminated_at_offset(ed, started_at, now_offset_s),
            }
        )
    return out


def _terminated_at_offset(
    ed: dict,
    started_at: datetime | None,
    now_offset_s: float | None,
) -> float | None:
    """Map an embryo's ``completed_at`` ISO timestamp into seconds-from-
    session-start. Returns ``None`` if the embryo isn't complete yet or
    we don't have the data to compute the offset.
    """
    if not ed.get("is_complete"):
        return None
    iso = ed.get("completed_at")
    if not iso or started_at is None:
        return None
    t = _parse_iso(iso)
    if t is None:
        return None
    delta = (t - started_at).total_seconds()
    if delta < 0:
        return 0.0
    if now_offset_s is not None:
        return min(delta, now_offset_s)
    return delta


# ---------------------------------------------------------------------------
# Timeline replay
# ---------------------------------------------------------------------------


def _resolve_timeline_paths(
    session_dir: Path,
    legacy_session_dir: Path,
) -> list[tuple[Path, bool]]:
    """Return the timeline.jsonl paths to read, with a per-source flag.

    The flag indicates whether the file is the global legacy timeline
    (which mixes multiple sessions and must be filtered by session_id) or
    a per-session file (no filtering needed).
    """
    paths: list[tuple[Path, bool]] = []
    # Per-session (new) location.
    p = session_dir / "timeline.jsonl"
    if p.exists():
        paths.append((p, False))
    # Per-session at the legacy bare-id folder, in case an old orchestrator
    # process happens to be writing there (it shouldn't — only timelapse.yaml
    # did — but easy to cover for cheap).
    p2 = legacy_session_dir / "timeline.jsonl"
    if p2.exists() and p2 not in (q for q, _ in paths):
        paths.append((p2, False))
    # Global legacy file shared across all sessions (sessions/timeline.jsonl).
    p3 = session_dir.parent / "timeline.jsonl"
    if p3.exists() and p3 not in (q for q, _ in paths):
        paths.append((p3, True))
    return paths


def _replay_timeline(
    *,
    session_dir: Path,
    legacy_session_dir: Path,
    session_id: str,
    embryo_dicts: list[dict],
    triggers: list[dict],
    started_at: datetime,
    now_offset_s: float,
    base_interval_s: float,
) -> None:
    """Walk timeline.jsonl chronologically and update the per-embryo dicts.

    Mutates ``embryo_dicts`` in place. Best-effort: any malformed line is
    skipped with a debug log.

    Reads from multiple timeline files in priority order (per-session first,
    legacy global second). Lines are merged and sorted by timestamp before
    replay so events from different files interleave correctly.
    """
    paths = _resolve_timeline_paths(session_dir, legacy_session_dir)
    if not paths:
        # No event history yet — close the seed phase at now and bail.
        for emb in embryo_dicts:
            _close_open_phase(emb, now_offset_s)
            _ensure_tail_power(emb, now_offset_s)
        return

    # Build embryo_id -> dict map so events can look up the right embryo cheaply.
    by_id = {e["id"]: e for e in embryo_dicts}
    trigger_ids = {t["id"] for t in triggers}

    # We need to know each embryo's current cadence_s as we go (for the
    # phase records). Seed from each embryo's initial phase.
    current_cadence: dict[str, float] = {
        e["id"]: e["phases"][0].get("cadence_s", base_interval_s) for e in embryo_dicts
    }
    # Track last trigger_fired per (embryo, rule) so we can cluster
    # consecutive fires into one event with a count.
    last_trigger: dict[tuple[str, str], dict] = {}

    # Collect events from all sources, filtering global file by session_id.
    events: list[tuple[datetime, dict]] = []
    seen_ids: set = set()
    for path, is_global in paths:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if is_global:
                        if ev.get("session_id") != session_id:
                            continue
                    ts = _parse_iso(ev.get("timestamp"))
                    if ts is None:
                        continue
                    # Dedup across files in case the same event made it into
                    # both the per-session and global logs.
                    eid = ev.get("event_id")
                    if eid:
                        if eid in seen_ids:
                            continue
                        seen_ids.add(eid)
                    events.append((ts, ev))
        except OSError as e:
            logger.debug("Could not read timeline %s: %s", path, e)
            continue

    events.sort(key=lambda pair: pair[0])

    for ts, ev in events:
        subtype = ev.get("event_subtype")
        at_s = (ts - started_at).total_seconds()
        if at_s < 0:
            # Pre-timelapse event from a previous run in this session — skip.
            continue
        data = ev.get("data") or {}
        embryo_id = str(ev.get("embryo_id") or data.get("embryo_id") or "")

        if subtype == "cadence_changed" and embryo_id in by_id:
            emb = by_id[embryo_id]
            new_phase_name = data.get("new_phase") or "normal"
            new_interval = data.get("new_interval_s")
            if new_interval is not None:
                try:
                    current_cadence[embryo_id] = float(new_interval)
                except (TypeError, ValueError):
                    pass
            mode = _phase_mode_from_name(new_phase_name)
            _close_open_phase(emb, at_s)
            emb["phases"].append(
                {
                    "mode": mode,
                    "start": at_s,
                    "end": None,
                    "cadence_s": current_cadence.get(embryo_id, base_interval_s),
                }
            )

        elif subtype == "power_changed" and embryo_id in by_id:
            wavelength = data.get("wavelength")
            if wavelength not in (488, None):
                continue
            new_pct = data.get("new_pct")
            if new_pct is None:
                continue
            emb = by_id[embryo_id]
            emb["power_history_488"].append(
                {
                    "at": at_s,
                    "pct": float(new_pct),
                }
            )
            emb["laser_488_pct_now"] = float(new_pct)

        elif subtype == "trigger_fired" and embryo_id in by_id:
            rule_name = data.get("rule_name")
            if not rule_name or rule_name not in trigger_ids:
                continue
            emb = by_id[embryo_id]
            key = (embryo_id, rule_name)
            prev = last_trigger.get(key)
            if prev is not None and at_s - prev["at"] <= _TRIGGER_CLUSTER_GAP_S:
                prev["count"] = prev.get("count", 1) + 1
                prev["at"] = at_s  # extend cluster to last fire
            else:
                rec = {"trigger_id": rule_name, "at": at_s, "count": 1}
                emb["trigger_events"].append(rec)
                last_trigger[key] = rec

        elif subtype == "burst_started" and embryo_id in by_id:
            emb = by_id[embryo_id]
            mode = data.get("mode") or "1hz"
            hz = 1.0 if mode == "1hz" else 20.0
            _close_open_phase(emb, at_s)
            emb["phases"].append(
                {
                    "mode": "burst",
                    "start": at_s,
                    "end": None,
                    "frames": int(data.get("frames") or 0),
                    "hz": hz,
                    "phase": data.get("phase"),
                }
            )

        elif subtype == "burst_completed" and embryo_id in by_id:
            emb = by_id[embryo_id]
            # Close burst phase if open. New orchestrator fires a
            # cadence_changed event right after to open the next phase;
            # legacy data without that event will get its open phase
            # closed at now_offset_s by the tail pass below.
            _close_open_phase(emb, at_s)

        elif subtype == "temp_protocol_started" and embryo_id in by_id:
            emb = by_id[embryo_id]
            emb["temp_protocol"] = {
                "start": at_s,
                "end": None,
                "target_setpoint_c": data.get("target_setpoint_c"),
                "frames": data.get("frames"),
                "bursts_before": data.get("bursts_before"),
                "bursts_after": data.get("bursts_after"),
            }

        elif subtype == "temp_protocol_completed" and embryo_id in by_id:
            emb = by_id[embryo_id]
            tp = emb.get("temp_protocol")
            if tp is not None:
                tp["end"] = at_s

        elif subtype == "setpoint_changed" and embryo_id in by_id:
            emb = by_id[embryo_id]
            to_val = data.get("to")
            emb["setpoint_changes"].append({"t": at_s, "to": to_val})

        elif subtype == "stopped":
            # Session-level stop: close every open phase at this time.
            for emb in embryo_dicts:
                _close_open_phase(emb, at_s)

    # Close any still-open phases at "now".
    for emb in embryo_dicts:
        _close_open_phase(emb, now_offset_s)
        _ensure_tail_power(emb, now_offset_s)


def _close_open_phase(emb: dict, at_s: float) -> None:
    if not emb["phases"]:
        return
    last = emb["phases"][-1]
    if last.get("end") is None or last["end"] < at_s:
        last["end"] = at_s


def _ensure_tail_power(emb: dict, now_offset_s: float) -> None:
    """Append a final power_history point at ``now_offset_s`` so the SVG
    extends the steady segment to the right edge."""
    hist = emb.get("power_history_488") or []
    if not hist:
        emb["power_history_488"] = [
            {
                "at": now_offset_s,
                "pct": emb.get("laser_488_pct_now", _DEFAULT_INITIAL_POWER_PCT),
            }
        ]
        return
    if hist[-1]["at"] < now_offset_s:
        hist.append({"at": now_offset_s, "pct": hist[-1]["pct"]})


def _phase_mode_from_name(name: str) -> str:
    """Map orchestrator cadence_phase values onto the swimlane's mode set.

    Orchestrator uses: ``normal``, ``fast``, ``burst``, ``paused``.
    Swimlane uses:    ``base``, ``fast``, ``burst``, ``cooldown``, ``paused``.

    ``normal`` maps to ``base`` so the leftmost phase reads as "base
    cadence" in the legend.
    """
    return {"normal": "base"}.get(name, name)


# ---------------------------------------------------------------------------
# Forward projection
# ---------------------------------------------------------------------------


def _project_forward(
    *,
    embryo_dicts: list[dict],
    now_offset_s: float,
    per_timepoint_ms: float,
) -> None:
    """Fill in ``projected_cadence_s`` / ``projected_end_s`` / ``dose_exhaust_at_s``."""
    for emb in embryo_dicts:
        last = emb["phases"][-1] if emb["phases"] else None
        cadence_s = (last or {}).get("cadence_s")
        if cadence_s is None or cadence_s <= 0:
            cadence_s = float(emb.get("projected_cadence_s") or 120.0)
        emb["projected_cadence_s"] = float(cadence_s)

        # Dose exhaust: how long can we keep going at the current cadence
        # before total_exposure_ms hits dose_budget_ms?
        budget = emb.get("dose_budget_ms") or 0.0
        used = emb.get("dose_used_ms") or 0.0
        if budget > 0 and budget > used:
            ms_per_acquisition = per_timepoint_ms
            acqs_per_second = (1.0 / cadence_s) if cadence_s > 0 else 0.0
            ms_per_second = ms_per_acquisition * acqs_per_second
            if ms_per_second > 0:
                remaining_s = (budget - used) / ms_per_second
                emb["dose_exhaust_at_s"] = now_offset_s + remaining_s

        # projected_end_s stays None for manual-stop / open-ended cases. The
        # frontend handles ``null`` by dashing into infinity.
        emb["projected_end_s"] = None


def _compute_horizon(
    now_offset_s: float,
    embryo_dicts: list[dict],
    padding_s: float,
) -> float:
    """Pick a horizon that comfortably contains the past + projected future."""
    floor = 4 * 3600.0
    ceiling = 24 * 3600.0
    candidates = [floor, now_offset_s * 2.0, now_offset_s + padding_s]
    for emb in embryo_dicts:
        for key in ("projected_end_s", "dose_exhaust_at_s"):
            v = emb.get(key)
            if v is not None:
                candidates.append(float(v) + padding_s)
    return float(min(max(candidates), ceiling))
