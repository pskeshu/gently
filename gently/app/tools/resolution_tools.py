"""
Resolution-mode tools.

These run at session start (or whenever the user wants to re-resolve)
to record what this session is for and apply the plan to the run.

Three groups:

- **Lifecycle**: attach_session_to_plan, mark_session_standalone,
  detach_session_from_plan, mark_plan_item_status. These record the
  decision and, when called from resolution mode, transition the agent
  into run mode.
- **Application**: apply_plan_acquisition_spec lifts an ImagingSpec
  from the plan into the experiment defaults so existing and future
  embryos pick up the plan's parameters.
- **Context recall**: recall_sibling_sessions and
  summarize_campaign_history give the agent the comparative state it
  needs to make a confident proposal. list_imaging_candidates is the
  escape-hatch that re-creates the old verbose briefing on demand.
"""

import logging
from types import SimpleNamespace
from typing import Any

from gently.harness.tools.helpers import require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _exit_resolution_if_active(agent, outcome: str) -> None:
    """Transition out of resolution mode if we're in it. Safe to call
    from other modes — does nothing in that case."""
    try:
        if getattr(agent, "mode", None) == "resolution":
            agent.exit_resolution_mode(outcome)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning(f"exit_resolution_mode failed: {e}")


def _set_active_plan_item(agent, plan_item_id: str | None) -> None:
    """Update both copies of the active plan item — ExperimentState
    (persisted) and AgentMemory (in-memory awareness)."""
    try:
        if agent.experiment is not None:
            agent.experiment.active_plan_item_id = plan_item_id
    except Exception:
        pass
    try:
        if getattr(agent, "memory", None) is not None:
            agent.memory.active_plan_item_id = plan_item_id
    except Exception:
        pass


def _short(text: str | None, n: int = 80) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Lifecycle: attach / standalone / detach / mark status
# ---------------------------------------------------------------------------


@tool(
    name="attach_session_to_plan",
    description=(
        "Record that this session is fulfilling a specific plan item. "
        "Sets the active plan item, links the session to the plan's "
        "campaign, and (when called from resolution mode) transitions "
        "the agent into run mode. Does NOT load acquisition parameters "
        "— call apply_plan_acquisition_spec for that. Accepts plan "
        "references like '1.3', 'task 3', or a UUID prefix."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Yes, continue Session 4",
            tool_input={
                "plan_item_id": "1.4",
                "rationale": "User confirmed continuing Session 4",
            },
        ),
    ],
)
async def attach_session_to_plan(
    plan_item_id: str,
    rationale: str = "",
    context: dict | None = None,
) -> str:
    """Attach the current session to a plan item."""
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if not cs:
        return "Error: context store unavailable — cannot attach to plan."

    item = cs.resolve_plan_item(plan_item_id)
    if not item:
        return f"Plan item '{plan_item_id}' not found."

    # Two sources of truth for the active plan item
    _set_active_plan_item(agent, item.id)

    # Link the session into the campaign's intent record AND onto the plan item
    # itself (item-level, appends — an item may have several sessions).
    session_id = getattr(agent, "session_id", None)
    linked = False
    if session_id:
        try:
            cs.link_session_campaign(session_id, item.campaign_id)
            cs.link_plan_item_session(item.id, session_id)
            linked = True
        except Exception as e:
            logger.warning(f"session↔plan link failed: {e}")

    # Invalidate prompt cache so the next system prompt picks up the
    # active item (the memory awareness layer injects its spec).
    try:
        agent.prompts.invalidate_context_cache()
    except Exception:
        pass

    outcome = f"attached to plan item {item.id[:8]} ({_short(item.title, 60)})"
    _exit_resolution_if_active(agent, outcome)

    lines = [f"Attached to plan: **{item.title}**"]
    if rationale:
        lines.append(f"Rationale: {rationale}")
    if linked:
        lines.append(f"Session linked to campaign {item.campaign_id[:8]}.")
    lines.append(
        "Next: call `apply_plan_acquisition_spec` to load the plan's "
        "acquisition parameters into the experiment defaults."
    )
    return "\n".join(lines)


@tool(
    name="mark_session_standalone",
    description=(
        "Record that this session is standalone — not attached to any "
        "plan or campaign. Use when the researcher is exploring, "
        "testing alignment, or otherwise not following a planned "
        "imaging item. Clears any prior plan attachment on this session."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Standalone, I'm just testing focus",
            tool_input={"description": "Standalone focus test"},
        ),
    ],
)
async def mark_session_standalone(
    description: str,
    context: dict | None = None,
) -> str:
    """Mark the session as a standalone (non-plan) run."""
    agent, err = require_agent(context)
    if err:
        return err

    # Clear plan attachment on both state stores
    _set_active_plan_item(agent, None)

    # Stash the stated intent on the experiment metadata so it's
    # preserved across save/resume and visible to downstream tools.
    try:
        agent.experiment.metadata.setdefault("session_resolution", {})
        agent.experiment.metadata["session_resolution"] = {
            "kind": "standalone",
            "description": description,
        }
    except Exception:
        pass

    try:
        agent.prompts.invalidate_context_cache()
    except Exception:
        pass

    _exit_resolution_if_active(agent, f"standalone: {description}")

    return (
        f"Recorded standalone session: {description}\n"
        f"No plan or campaign attached. Default acquisition parameters apply."
    )


@tool(
    name="detach_session_from_plan",
    description=(
        "Remove this session's attachment to a plan item. Use when the "
        "researcher wants to abandon the plan mid-session and continue "
        "as standalone, or wants to re-attach to a different plan item "
        "(call attach_session_to_plan after detaching)."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Actually let's detach from Session 4 and keep going freely",
            tool_input={"reason": "User wants to continue exploration"},
        ),
    ],
)
async def detach_session_from_plan(
    reason: str = "",
    context: dict | None = None,
) -> str:
    """Detach the current session from its plan item."""
    agent, err = require_agent(context)
    if err:
        return err

    prior = None
    try:
        prior = agent.experiment.active_plan_item_id
    except Exception:
        pass

    if not prior:
        return "No plan currently attached to this session."

    _set_active_plan_item(agent, None)

    try:
        agent.experiment.metadata["session_resolution"] = {
            "kind": "detached",
            "from_plan_item_id": prior,
            "reason": reason or "no reason given",
        }
    except Exception:
        pass

    try:
        agent.prompts.invalidate_context_cache()
    except Exception:
        pass

    _exit_resolution_if_active(agent, f"detached from {prior[:8]}")

    msg = f"Detached from plan item {prior[:8]}."
    if reason:
        msg += f" Reason: {reason}"
    return msg


@tool(
    name="mark_plan_item_status",
    description=(
        "Update the status of a plan item — typically used at session "
        "end to mark an item completed or skipped. Valid statuses: "
        "'completed', 'skipped', 'blocked', 'in_progress', 'planned'. "
        "For completion, pass an outcome describing what happened."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Mark Session 3 as completed — all four embryos hatched",
            tool_input={
                "plan_item_id": "1.3",
                "status": "completed",
                "notes": "4/4 hatched within 13.5h",
            },
        ),
    ],
)
async def mark_plan_item_status(
    plan_item_id: str,
    status: str,
    notes: str = "",
    context: dict | None = None,
) -> str:
    """Update a plan item's status."""
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if not cs:
        return "Error: context store unavailable."

    item = cs.resolve_plan_item(plan_item_id)
    if not item:
        return f"Plan item '{plan_item_id}' not found."

    from gently.harness.memory.model import PlanItemStatus

    status_map = {
        "completed": PlanItemStatus.COMPLETED,
        "skipped": PlanItemStatus.SKIPPED,
        "blocked": PlanItemStatus.BLOCKED,
        "in_progress": PlanItemStatus.IN_PROGRESS,
        "planned": PlanItemStatus.PLANNED,
    }
    key = status.strip().lower()
    if key not in status_map:
        return f"Unknown status '{status}'. Valid: {', '.join(status_map.keys())}."

    target = status_map[key]
    try:
        if target == PlanItemStatus.COMPLETED:
            cs.complete_plan_item(item.id, notes or "Completed")
        elif target == PlanItemStatus.SKIPPED:
            cs.skip_plan_item(item.id, notes or None)
        else:
            cs.update_plan_item(item_id=item.id, status=target, outcome=notes or None)
    except Exception as e:
        return f"Failed to update plan item status: {e}"

    return f"Plan item '{item.title}' → {key}" + (f" ({notes})" if notes else "")


# ---------------------------------------------------------------------------
# Application: apply_plan_acquisition_spec
# ---------------------------------------------------------------------------


def _apply_spec_to_embryo(embryo, spec) -> list[str]:
    """Write per-embryo acquisition fields from an ImagingSpec.
    Returns a list of human-readable changes made."""
    applied = []
    if spec.num_slices is not None:
        embryo.num_slices = int(spec.num_slices)
        applied.append(f"num_slices={spec.num_slices}")
    if spec.exposure_ms is not None:
        embryo.exposure_ms = float(spec.exposure_ms)
        applied.append(f"exposure_ms={spec.exposure_ms}")
    if spec.interval_s is not None:
        embryo.interval_seconds = float(spec.interval_s)
        applied.append(f"interval_s={spec.interval_s}")
    return applied


@tool(
    name="apply_plan_acquisition_spec",
    description=(
        "Load a plan item's ImagingSpec into the experiment defaults. "
        "Sets per-embryo acquisition parameters (num_slices, exposure_ms, "
        "interval_s) for all existing embryos and stashes the resolved "
        "spec for future-added embryos. Optional `overrides` dict lets "
        "you skip specific fields (e.g. {'interval_s': null} to leave "
        "the current interval alone)."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Load the spec for Session 4",
            tool_input={"plan_item_id": "1.4"},
        ),
    ],
)
async def apply_plan_acquisition_spec(
    plan_item_id: str,
    overrides: dict | None = None,
    context: dict | None = None,
) -> str:
    """Apply a plan's ImagingSpec to the experiment."""
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if not cs:
        return "Error: context store unavailable."

    item = cs.resolve_plan_item(plan_item_id)
    if not item:
        return f"Plan item '{plan_item_id}' not found."

    spec = cs.resolve_imaging_spec(item)
    if not spec:
        return f"Plan item '{item.title}' has no imaging spec to apply."

    overrides = overrides or {}

    # Apply per-embryo to anything we already have
    applied_per_embryo: list[str] = []
    embryo_count = 0
    experiment = getattr(agent, "experiment", None)
    if experiment and experiment.embryos:
        for embryo in experiment.embryos.values():
            # Build an "effective" spec respecting overrides — copy then
            # zero out any field the caller asked us to skip.
            eff = SimpleNamespace()
            eff.num_slices = (
                None
                if "num_slices" in overrides and overrides["num_slices"] is None
                else (overrides.get("num_slices") if "num_slices" in overrides else spec.num_slices)
            )
            eff.exposure_ms = (
                None
                if "exposure_ms" in overrides and overrides["exposure_ms"] is None
                else (
                    overrides.get("exposure_ms") if "exposure_ms" in overrides else spec.exposure_ms
                )
            )
            eff.interval_s = (
                None
                if "interval_s" in overrides and overrides["interval_s"] is None
                else (overrides.get("interval_s") if "interval_s" in overrides else spec.interval_s)
            )
            changes = _apply_spec_to_embryo(embryo, eff)
            if changes and not applied_per_embryo:
                applied_per_embryo = changes
            embryo_count += 1 if changes else 0

    # Stash on experiment.metadata so newly added embryos can inherit
    # and so the resolution outcome survives session save/resume.
    if experiment is not None:
        try:
            experiment.metadata["active_plan_spec"] = {
                "plan_item_id": item.id,
                "plan_item_title": item.title,
                "strain": spec.strain,
                "temperature_c": spec.temperature_c,
                "num_slices": spec.num_slices,
                "exposure_ms": spec.exposure_ms,
                "interval_s": spec.interval_s,
                "stop_condition": spec.stop_condition,
                "detectors": list(spec.detectors) if spec.detectors else None,
                "success_criteria": spec.success_criteria,
                "adaptive_intervals": dict(spec.adaptive_intervals)
                if spec.adaptive_intervals
                else None,
            }
        except Exception:
            pass

    # Build a narratable summary for the agent to quote
    parts: list[str] = []
    if spec.strain:
        parts.append(f"strain={spec.strain}")
    if spec.temperature_c is not None:
        parts.append(f"temp={spec.temperature_c}°C")
    if spec.num_slices is not None:
        parts.append(f"{spec.num_slices} slices")
    if spec.exposure_ms is not None:
        parts.append(f"{spec.exposure_ms}ms exposure")
    if spec.laser_wavelength_nm is not None:
        laser = f"{spec.laser_wavelength_nm}nm"
        if spec.laser_power_pct is not None:
            laser += f" at {spec.laser_power_pct}%"
        parts.append(laser)
    if spec.interval_s is not None:
        parts.append(f"every {spec.interval_s}s")
    if spec.stop_condition:
        parts.append(f"stop: {spec.stop_condition}")
    spec_summary = ", ".join(parts) if parts else "(spec is empty)"

    lines = [f"Loaded spec for **{item.title}**: {spec_summary}"]
    if embryo_count:
        lines.append(
            f"Applied per-embryo to {embryo_count} existing embryo(s) "
            f"(e.g. {', '.join(applied_per_embryo)})."
        )
    else:
        lines.append(
            "No embryos in the experiment yet — spec is stashed and "
            "will apply to embryos as they're added."
        )
    if spec.adaptive_intervals:
        keys = ", ".join(spec.adaptive_intervals.keys())
        lines.append(
            f"Note: spec defines adaptive intervals ({keys}) — "
            "these are stashed but require timelapse wiring to take effect "
            "and will be honored once apply_plan_adaptive_intervals lands."
        )
    if spec.detectors:
        lines.append(
            f"Note: spec lists detectors {spec.detectors} — these are "
            "stashed but not yet enabled (separate apply tool to come)."
        )
    if spec.success_criteria:
        lines.append(f"Success criteria recorded: {spec.success_criteria}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context recall: sibling sessions, campaign history, candidates listing
# ---------------------------------------------------------------------------


@tool(
    name="recall_sibling_sessions",
    description=(
        "List recent sessions that share a plan item or campaign with "
        "the given identifier. Use during resolution to surface what "
        "the researcher has already done — partway-through sessions are "
        "the strongest hint about what they're resuming. Pass either a "
        "plan_item_id (returns sessions linked to that item's campaign) "
        "or a campaign_id."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Find sessions in this campaign",
            tool_input={"identifier": "1.4"},
        ),
    ],
)
async def recall_sibling_sessions(
    identifier: str,
    limit: int = 10,
    context: dict | None = None,
) -> str:
    """Return sessions sharing the given plan item's campaign or the campaign itself."""
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if not cs:
        return "Error: context store unavailable."

    # Try plan-item-first; fall back to treating identifier as campaign id
    campaign_id: str | None = None
    plan_item = cs.resolve_plan_item(identifier) if hasattr(cs, "resolve_plan_item") else None
    if plan_item:
        campaign_id = plan_item.campaign_id
    else:
        try:
            campaign = cs.get_campaign(identifier)
            if campaign:
                campaign_id = campaign.id
        except Exception:
            campaign_id = None

    if not campaign_id:
        return f"Could not resolve '{identifier}' to a plan item or campaign."

    # Get the campaign tree and walk plan items, collecting session ids
    sessions: list[dict] = []
    try:
        items = cs.get_plan_items(
            campaign_id=campaign_id,
            include_children=True,
        )
    except Exception as e:
        return f"Could not list plan items: {e}"

    file_store = getattr(agent, "store", None)
    for item in items:
        sid = getattr(item, "session_id", None)
        if not sid:
            continue
        meta: dict[str, Any] = {}
        if file_store is not None:
            try:
                meta = dict(file_store.get_session(sid) or {})
            except Exception:
                meta = {}
        sessions.append(
            {
                "plan_item_title": item.title,
                "plan_item_status": item.status.value
                if hasattr(item.status, "value")
                else str(item.status),
                "session_id": sid,
                "last_active": meta.get("last_active"),
                "name": meta.get("name"),
            }
        )

    if not sessions:
        return f"No sessions yet for campaign {campaign_id[:8]}."

    sessions.sort(key=lambda s: s.get("last_active") or "", reverse=True)
    sessions = sessions[:limit]

    lines = [f"Sessions in this campaign ({len(sessions)} shown):"]
    for s in sessions:
        last = s.get("last_active") or "unknown"
        lines.append(
            f"- {s['plan_item_title']} — {s['plan_item_status']} — "
            f"session {s['session_id'][:8]} (last active: {last})"
        )
    return "\n".join(lines)


@tool(
    name="summarize_campaign_history",
    description=(
        "Brief overview of a campaign's progress — plan-item counts by "
        "status, recent learnings on the campaign, and any open "
        "questions. Use during resolution to ground a confident proposal."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="What's the state of Embryo Classification Dataset?",
            tool_input={"campaign_id": "abc123"},
        ),
    ],
)
async def summarize_campaign_history(
    campaign_id: str,
    context: dict | None = None,
) -> str:
    """Compact campaign-progress summary for resolution-mode reasoning."""
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if not cs:
        return "Error: context store unavailable."

    campaign = None
    try:
        campaign = cs.get_campaign(campaign_id)
    except Exception:
        pass
    if not campaign:
        return f"Campaign '{campaign_id}' not found."

    lines = [f"# {campaign.description or campaign.id[:8]}"]
    if campaign.target:
        lines.append(f"Target: {campaign.target}")
    if campaign.progress:
        lines.append(f"Progress: {campaign.progress}")

    try:
        status = cs.get_plan_status(campaign.id)
        lines.append(
            f"Plan items: {status['completed']}/{status['total']} done"
            + (f", {status['in_progress']} in progress" if status.get("in_progress") else "")
            + (f", {status['blocked']} blocked" if status.get("blocked") else "")
        )
        next_actions = status.get("next_actions") or []
        if next_actions:
            titles = [a.title for a in next_actions[:3]]
            lines.append("Next actions: " + "; ".join(titles))
    except Exception:
        pass

    return "\n".join(lines)


@tool(
    name="list_imaging_candidates",
    description=(
        "Escape-hatch listing of every unblocked imaging plan item, "
        "with its full spec. Call this only when the researcher "
        "explicitly asks to see all options or says 'show me everything'. "
        "Resolution mode should default to proposing the top 1-3 "
        "candidates, not dumping the full list."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Show me all imaging tasks",
            tool_input={},
        ),
    ],
)
async def list_imaging_candidates(
    context: dict | None = None,
) -> str:
    """Full deterministic listing of unblocked imaging plan items."""
    agent, err = require_agent(context)
    if err:
        return err

    memory = getattr(agent, "memory", None)
    if not memory:
        return "Error: agent memory unavailable."

    try:
        _, candidates = memory.resolve_plan_context()
    except Exception as e:
        return f"Could not resolve plan context: {e}"

    if not candidates:
        return "No unblocked imaging tasks. (Standalone is the natural choice here.)"

    lines = [f"{len(candidates)} imaging task(s) unblocked:"]
    for item, spec, campaign in candidates:
        spec_summary = memory.format_imaging_spec_summary(spec) if spec else "no spec"
        c_label = (campaign.description or campaign.id[:8]) if campaign else "?"
        lines.append(f"- **{item.title}** — {spec_summary}")
        lines.append(f"  Campaign: {_short(c_label, 80)} (plan item id: {item.id[:8]})")
    return "\n".join(lines)
