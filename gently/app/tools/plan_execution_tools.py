"""
Plan → Execution Bridge — run-mode tools for executing plan items.

These tools let the agent translate structured plan items into
live microscope actions: configure acquisition, enable detectors,
start timelapses, and track completion.
"""

import logging

from gently.harness.tools.helpers import (
    require_agent,
)
from gently.harness.tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# execute_plan_item
# ---------------------------------------------------------------------------


@tool(
    name="execute_plan_item",
    description=(
        "Execute a planned imaging item — resolve its spec, configure "
        "acquisition parameters, enable detectors, link the session, and "
        "start the timelapse. Accepts item references like '1.3', 'task 3', "
        "or a UUID."
    ),
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
    examples=[
        ToolExample(
            user_query="Start imaging for task 1.1",
            tool_input={"item_ref": "1.1"},
        ),
    ],
)
async def execute_plan_item(
    item_ref: str,
    embryo_ids: list[str] | None = None,
    context: dict | None = None,
) -> str:
    """Execute a planned imaging item."""
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if not cs:
        return "Error: Context store not available"

    # 1. Resolve plan item
    item = cs.resolve_plan_item(item_ref)
    if not item:
        return f"Plan item '{item_ref}' not found"

    from gently.harness.memory.model import PlanItemStatus, PlanItemType

    # 2. Verify type and status
    if item.type != PlanItemType.IMAGING:
        return (
            f"Plan item '{item.title}' is type '{item.type.value}', not 'imaging'. "
            f"Only imaging items can be executed from run mode."
        )
    if item.status not in (PlanItemStatus.PLANNED, PlanItemStatus.BLOCKED):
        return (
            f"Plan item '{item.title}' has status '{item.status.value}'. "
            f"Expected 'planned'. Cannot re-execute."
        )

    # 3. Resolve imaging spec (follows inheritance)
    spec = cs.resolve_imaging_spec(item)
    if not spec:
        return f"Plan item '{item.title}' has no imaging spec"

    actions: list[str] = []

    # 4. Configure acquisition params on the experiment state
    experiment = getattr(agent, "experiment", None)
    if experiment and embryo_ids:
        if spec.num_slices:
            for eid in embryo_ids:
                try:
                    experiment.update_embryo_params(eid, num_slices=spec.num_slices)
                except Exception:
                    pass
            actions.append(f"num_slices={spec.num_slices}")
        if spec.exposure_ms:
            for eid in embryo_ids:
                try:
                    experiment.update_embryo_params(eid, exposure_ms=spec.exposure_ms)
                except Exception:
                    pass
            actions.append(f"exposure_ms={spec.exposure_ms}")

    # 5. Enable detectors
    if spec.detectors:
        detector_registry = getattr(agent, "detector_registry", None)
        if detector_registry:
            for det_name in spec.detectors:
                try:
                    detector_registry.enable(det_name)
                    actions.append(f"detector '{det_name}' enabled")
                except Exception as e:
                    actions.append(f"detector '{det_name}' failed: {e}")

    # 6. Start timelapse via orchestrator — this is what activates the session, so the
    #    plan↔session link must happen AFTER it (step 7), not before (the old bug:
    #    agent.session_id was still None here, so the link silently dropped).
    orchestrator = getattr(agent, "timelapse_orchestrator", None)
    if orchestrator:
        try:
            stop_cond = spec.stop_condition or "manual"
            interval = spec.interval_s or 120
            await orchestrator.start(
                embryo_ids=embryo_ids,
                stop_condition=stop_cond,
                base_interval_seconds=interval,
            )
            actions.append(f"timelapse started: {stop_cond}, {interval}s interval")

            # 9. Set up adaptive interval rules
            if spec.adaptive_intervals:
                for stage_key, new_interval in spec.adaptive_intervals.items():
                    try:
                        orchestrator.add_speedup_on_stage(
                            stage_name=stage_key,
                            new_interval_seconds=new_interval,
                        )
                        actions.append(f"interval rule: {stage_key} → {new_interval}s")
                    except Exception as e:
                        logger.error(
                            "Failed to install adaptive interval rule "
                            "(stage=%s, new_interval=%s): %s",
                            stage_key,
                            new_interval,
                            e,
                        )
                        actions.append(f"interval rule FAILED: {stage_key} → {new_interval}s ({e})")
        except Exception as e:
            actions.append(f"timelapse start error: {e}")

    # 7. Link this run to the plan item + campaign — AFTER start, so the session exists.
    #    Appends (an item may run several sessions). Surface failures, don't swallow.
    session_id = getattr(agent, "session_id", None)
    if session_id:
        try:
            cs.link_session_campaign(session_id, item.campaign_id)
            cs.link_plan_item_session(item.id, session_id)
            actions.append(
                f"linked session {session_id[:8]} → plan item + campaign (status → in_progress)"
            )
        except Exception as e:
            actions.append(f"⚠ link failed: {e}")
    else:
        actions.append("⚠ no active session — could not link this run to the plan item")

    # Summary
    lines = [f"Executing plan item: {item.title}"]
    if spec.strain:
        lines.append(f"  Strain: {spec.strain}")
    if spec.stop_condition:
        lines.append(f"  Stop condition: {spec.stop_condition}")
    if spec.interval_s:
        lines.append(f"  Interval: {spec.interval_s}s")
    lines.append("")
    lines.append("Actions taken:")
    for a in actions:
        lines.append(f"  • {a}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# complete_current_plan_item
# ---------------------------------------------------------------------------


@tool(
    name="complete_current_plan_item",
    description=(
        "Mark a plan item as completed with an optional outcome description. "
        "Reports any newly unblocked items that can now be started."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Mark task 1.1 as done",
            tool_input={
                "item_ref": "1.1",
                "outcome": "GFP signal visible in 3/4 embryos, good SNR",
            },
        ),
    ],
)
async def complete_current_plan_item(
    item_ref: str,
    outcome: str | None = None,
    context: dict | None = None,
) -> str:
    """Complete a plan item and report newly unblocked items."""
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if not cs:
        return "Error: Context store not available"

    # 1. Resolve
    item = cs.resolve_plan_item(item_ref)
    if not item:
        return f"Plan item '{item_ref}' not found"

    # 2. Complete it
    outcome_text = outcome or "Completed"
    cs.complete_plan_item(item.id, outcome_text)

    # 3. Find newly unblocked items
    # Walk up to root campaign for full picture
    campaign = cs.get_campaign(item.campaign_id)
    root_id = item.campaign_id
    if campaign and campaign.parent_id:
        root_id = campaign.parent_id

    unblocked = cs.get_unblocked_plan_items(root_id)
    # Filter to only items that were previously blocked by *this* item
    newly_ready = []
    for u in unblocked:
        if u.id != item.id:
            newly_ready.append(u)

    # 4. Build response
    lines = [f"Completed: '{item.title}'"]
    if outcome:
        lines.append(f"  Outcome: {outcome}")
    lines.append("")

    if newly_ready:
        lines.append(f"Newly unblocked ({len(newly_ready)}):")
        for u in newly_ready[:10]:
            lines.append(f"  → [{u.type.value}] {u.title}")
    else:
        lines.append("No new items unblocked.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-link helper (used by timelapse_tools.py, Feature 3)
# ---------------------------------------------------------------------------


def try_auto_link_plan_item(
    cs,
    session_id: str,
    stop_condition: str,
    interval_seconds: float,
) -> str | None:
    """
    Best-effort auto-link: find a planned imaging item that matches the
    current timelapse parameters and link the session.

    Returns the matched item title, or None.
    """
    from gently.harness.memory.model import PlanItemStatus

    try:
        campaigns = cs.get_root_campaigns()
    except Exception:
        return None

    matched_item = None

    for campaign in campaigns:
        items = cs.get_plan_items(
            campaign_id=campaign.id,
            status="planned",
            type="imaging",
            include_children=True,
        )

        candidates = []
        for item in items:
            spec = cs.resolve_imaging_spec(item)
            if not spec:
                continue

            # Match on stop_condition (exact) and interval_s (within 20%)
            if spec.stop_condition and spec.stop_condition.lower() != stop_condition.lower():
                continue
            if spec.interval_s:
                ratio = interval_seconds / spec.interval_s
                if ratio < 0.8 or ratio > 1.2:
                    continue
            elif not spec.stop_condition:
                # Neither matches — skip
                continue

            candidates.append(item)

        # Conservative: only auto-link if exactly one match
        if len(candidates) == 1:
            matched_item = candidates[0]
            break

    if not matched_item:
        return None

    try:
        cs.update_plan_item(
            item_id=matched_item.id,
            status=PlanItemStatus.IN_PROGRESS,
            session_id=session_id,
        )
        cs.link_session_campaign(session_id, matched_item.campaign_id)
        logger.info(
            f"Auto-linked session {session_id} to plan item "
            f"'{matched_item.title}' ({matched_item.id})"
        )
        return matched_item.title
    except Exception:
        return None
