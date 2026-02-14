"""
Planning tools — create and manage experimental plans.

These tools let the copilot build structured plans from
scientific conversations: campaigns, phases, plan items,
and their dependencies.
"""

import dataclasses
import json
from typing import Dict, List, Optional

from ...tool_registry import tool, ToolCategory, ToolExample


# ---------------------------------------------------------------------------
# Campaign / Phase Management
# ---------------------------------------------------------------------------

@tool(
    name="create_campaign",
    description=(
        "Create a research campaign or phase. Campaigns are top-level research "
        "goals (e.g., 'Nerve ring formation study'). Phases are sub-campaigns "
        "within a parent (e.g., 'Phase 1 — Reporter validation'). Returns the "
        "campaign ID."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Let's plan a nerve ring formation study",
            tool_input={
                "description": "Nerve ring formation study",
                "shorthand": "nrf-2026",
                "target": "Characterize nerve ring assembly dynamics in live embryos",
            },
        ),
    ],
)
async def create_campaign(
    description: str,
    shorthand: str = None,
    target: str = None,
    parent_id: str = None,
    context: Dict = None,
) -> str:
    """Create a campaign or sub-campaign (phase)."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    # Fix year in shorthand — models sometimes hallucinate the wrong year
    if shorthand:
        import re
        from datetime import datetime
        current_year = str(datetime.now().year)
        shorthand = re.sub(r'-20\d{2}$', f'-{current_year}', shorthand)

    store = copilot.context_store
    cid = store.create_campaign(
        description=description,
        shorthand=shorthand,
        target=target,
        parent_id=parent_id,
    )
    if parent_id:
        return f"Created phase '{description}' (id: {cid}) under campaign {parent_id}"
    return f"Created campaign '{description}' (id: {cid})"


# ---------------------------------------------------------------------------
# Plan Item Management
# ---------------------------------------------------------------------------

@tool(
    name="create_plan_item",
    description=(
        "Create a plan item — a single task in the experimental plan. "
        "Types: 'imaging' (with imaging_spec), 'bench' (with bench_spec), "
        "'genetics', 'analysis', 'decision_point'. "
        "Use depends_on to set dependencies on other plan item IDs."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Add a pilot imaging session to phase 1",
            tool_input={
                "campaign_id": "phase1_id",
                "type": "imaging",
                "title": "Pilot — rab-3p::GFP visibility test",
                "spec": {
                    "strain": "OH904",
                    "reporter": "rab-3p::GFP",
                    "num_slices": 80,
                    "exposure_ms": 10.0,
                    "interval_s": 180,
                    "stop_condition": "pretzel",
                    "num_embryos": 4,
                    "success_criteria": "Nerve ring visible in ≥3/4 embryos",
                },
            },
        ),
    ],
)
async def create_plan_item(
    campaign_id: str,
    type: str,
    title: str,
    description: str = None,
    spec: Dict = None,
    inherit_from: str = None,
    depends_on: List[str] = None,
    phase_order: int = -1,
    context: Dict = None,
) -> str:
    """Create a plan item within a campaign/phase."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    store = copilot.context_store
    item_id = store.create_plan_item(
        campaign_id=campaign_id,
        type=type,
        title=title,
        description=description,
        spec=spec,
        inherit_from=inherit_from,
        depends_on=depends_on,
        phase_order=phase_order,
    )

    # Include the human-friendly task number in the response
    item = store.get_plan_item(item_id)
    task_num = ""
    if item:
        # Determine phase number if this is a subcampaign
        campaign = store.get_campaign(campaign_id)
        if campaign and campaign.parent_id:
            phases = store.get_subcampaigns(campaign.parent_id)
            for pi, phase in enumerate(phases, 1):
                if phase.id == campaign_id:
                    # Count items in this phase up to this one
                    items = store.get_plan_items(campaign_id=campaign_id)
                    items.sort(key=lambda x: x.phase_order)
                    for ti, it in enumerate(items, 1):
                        if it.id == item_id:
                            task_num = f" #{pi}.{ti}"
                            break
                    break
        else:
            items = store.get_plan_items(campaign_id=campaign_id)
            items.sort(key=lambda x: x.phase_order)
            for ti, it in enumerate(items, 1):
                if it.id == item_id:
                    task_num = f" #{ti}"
                    break

    dep_str = f", depends on: {depends_on}" if depends_on else ""
    return f"Created [{type}] plan item{task_num} '{title}' (id: {item_id}){dep_str}"


@tool(
    name="update_plan_item",
    description=(
        "Update an existing plan item — change status, title, description, "
        "outcome, or spec. Use this to mark items as completed, skipped, or "
        "to update imaging specifications."
    ),
    category=ToolCategory.UTILITY,
)
async def update_plan_item(
    item_id: str,
    status: str = None,
    title: str = None,
    description: str = None,
    outcome: str = None,
    spec: Dict = None,
    context: Dict = None,
) -> str:
    """Update a plan item. item_id can be a UUID, task number (e.g. '3'),
    or phase.task reference (e.g. '1.3')."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    store = copilot.context_store

    # Resolve natural references
    item = store.resolve_plan_item(item_id)
    if not item:
        return f"Plan item '{item_id}' not found"
    resolved_id = item.id

    from gently.context.model import PlanItemStatus

    status_enum = PlanItemStatus(status) if status else None
    store.update_plan_item(
        item_id=resolved_id,
        status=status_enum,
        title=title,
        description=description,
        outcome=outcome,
        spec=spec,
    )
    changes = []
    if status:
        changes.append(f"status → {status}")
    if outcome:
        changes.append(f"outcome recorded")
    if spec:
        changes.append(f"spec updated")
    if title:
        changes.append(f"title → {title}")
    return f"Updated plan item '{item.title}' ({resolved_id}): {', '.join(changes) or 'updated'}"


@tool(
    name="link_plan_items",
    description=(
        "Set a dependency between plan items. The first item (item_id) "
        "cannot start until the second item (depends_on_id) is completed. "
        "Items can be referenced by UUID, task number (e.g. '3'), or "
        "phase.task (e.g. '1.3')."
    ),
    category=ToolCategory.UTILITY,
)
async def link_plan_items(
    item_id: str,
    depends_on_id: str,
    context: Dict = None,
) -> str:
    """Add a dependency between plan items."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    store = copilot.context_store

    # Resolve natural references
    item = store.resolve_plan_item(item_id)
    dep = store.resolve_plan_item(depends_on_id)
    if not item:
        return f"Plan item '{item_id}' not found"
    if not dep:
        return f"Plan item '{depends_on_id}' not found"

    store.add_plan_item_dependency(item.id, dep.id)
    return f"Linked: '{item.title}' now depends on '{dep.title}'"


@tool(
    name="get_plan_item",
    description=(
        "Look up a plan item by reference. Accepts a UUID, task number "
        "(e.g. '3'), phase.task (e.g. '1.3'), or natural language like "
        "'task 3 of phase 1'. Returns full details including spec, "
        "dependencies, and status."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Show me task 3 of phase 1",
            tool_input={"ref": "1.3"},
        ),
        ToolExample(
            user_query="What's the status of the pilot imaging task?",
            tool_input={"ref": "1"},
        ),
    ],
)
async def get_plan_item_tool(
    ref: str,
    campaign_id: str = None,
    context: Dict = None,
) -> str:
    """Look up a plan item by natural reference."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    store = copilot.context_store
    item = store.resolve_plan_item(ref, campaign_id=campaign_id)
    if not item:
        return f"No plan item matching '{ref}' found"

    return _format_plan_item(item, store, task_num="")


# ---------------------------------------------------------------------------
# Plan Review
# ---------------------------------------------------------------------------

@tool(
    name="propose_plan",
    description=(
        "Present the full experimental plan for the researcher to review. "
        "Renders the campaign hierarchy, all plan items with their specs, "
        "dependencies, and status. Call this after building the plan to get "
        "researcher approval before committing."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Show me the plan",
            tool_input={"campaign_id": "nrf-2026"},
        ),
    ],
)
async def propose_plan(
    campaign_id: str,
    context: Dict = None,
) -> str:
    """Render the full plan for review."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    store = copilot.context_store
    campaign = store.get_campaign(campaign_id)
    if not campaign:
        return f"Campaign {campaign_id} not found"

    lines = []
    lines.append(f"{'=' * 55}")
    lines.append(f" EXPERIMENTAL PLAN: {campaign.description}")
    if campaign.shorthand:
        lines.append(f" Campaign: {campaign.shorthand}")
    if campaign.target:
        lines.append(f" Goal: {campaign.target}")
    lines.append(f"{'=' * 55}")
    lines.append("")

    # Get phases (sub-campaigns)
    phases = store.get_subcampaigns(campaign_id)

    if not phases:
        # No sub-campaigns — show items directly
        items = store.get_plan_items(campaign_id=campaign_id)
        items.sort(key=lambda x: x.phase_order)
        for task_idx, item in enumerate(items, 1):
            lines.append(_format_plan_item(item, store, task_num=str(task_idx)))
    else:
        for phase_idx, phase in enumerate(phases, 1):
            lines.append(f"── Phase {phase_idx}: {phase.description} ──")
            lines.append("")
            items = store.get_plan_items(campaign_id=phase.id)
            items.sort(key=lambda x: x.phase_order)
            for task_idx, item in enumerate(items, 1):
                num = f"{phase_idx}.{task_idx}"
                lines.append(_format_plan_item(item, store, task_num=num))
            lines.append("")

    # Summary
    status = store.get_plan_status(campaign_id)
    lines.append(f"── Summary ──")
    lines.append(f"Total items: {status['total']}")
    lines.append(f"Completed: {status['completed']}")
    if status["pending_decisions"]:
        lines.append(f"Pending decisions: {len(status['pending_decisions'])}")
    if status["next_actions"]:
        lines.append(f"Next actions: {', '.join(a.title for a in status['next_actions'][:3])}")

    return "\n".join(lines)


def _format_plan_item(item, store, task_num: str = "") -> str:
    """Format a single plan item for display."""
    from gently.context.model import PlanItemStatus

    status_icons = {
        PlanItemStatus.PLANNED: " ",
        PlanItemStatus.IN_PROGRESS: "▶",
        PlanItemStatus.COMPLETED: "✓",
        PlanItemStatus.SKIPPED: "–",
        PlanItemStatus.BLOCKED: "⏳",
    }
    icon = status_icons.get(item.status, " ")
    type_tag = item.type.value.upper()
    num_label = f"#{task_num} " if task_num else ""
    line = f" {icon} {num_label}[{type_tag}] {item.title}  ({item.id})"

    details = []
    if item.imaging_spec:
        spec = store.resolve_imaging_spec(item) or item.imaging_spec
        if spec.strain:
            details.append(f"   Strain: {spec.strain}")
            if spec.reporter:
                details[-1] += f" ({spec.reporter})"
        params = []
        if spec.num_slices:
            params.append(f"{spec.num_slices} slices")
        if spec.exposure_ms:
            params.append(f"{spec.exposure_ms}ms")
        if spec.laser_wavelength_nm:
            params.append(f"{spec.laser_wavelength_nm}nm")
        if spec.laser_power_pct:
            params.append(f"{spec.laser_power_pct}%")
        if params:
            details.append(f"   Params: {', '.join(params)}")
        timing = []
        if spec.interval_s:
            timing.append(f"{spec.interval_s}s interval")
        if spec.target_window:
            timing.append(spec.target_window)
        elif spec.stop_condition:
            timing.append(f"stop: {spec.stop_condition}")
        if timing:
            details.append(f"   Timing: {', '.join(timing)}")
        if spec.num_embryos:
            details.append(f"   Embryos: {spec.num_embryos}")
        if spec.success_criteria:
            details.append(f"   Criteria: {spec.success_criteria}")

    if item.bench_spec:
        spec = item.bench_spec
        if spec.protocol:
            details.append(f"   Protocol: {spec.protocol}")
        if spec.strains:
            details.append(f"   Strains: {', '.join(spec.strains)}")
        if spec.target_genotype:
            details.append(f"   Target: {spec.target_genotype}")
        if spec.estimated_days:
            details.append(f"   Timeline: ~{spec.estimated_days} days")
        if spec.success_criteria:
            details.append(f"   Criteria: {spec.success_criteria}")

    if item.depends_on:
        dep_items = [store.get_plan_item(d) for d in item.depends_on]
        dep_names = [d.title for d in dep_items if d]
        if dep_names:
            details.append(f"   Depends on: {', '.join(dep_names)}")

    if item.outcome:
        details.append(f"   Outcome: {item.outcome}")

    if details:
        return line + "\n" + "\n".join(details)
    return line


@tool(
    name="get_plan_status",
    description=(
        "Get the current status of an experimental plan — how many items "
        "are completed, what's next, any pending decisions."
    ),
    category=ToolCategory.UTILITY,
)
async def get_plan_status(
    campaign_id: str,
    context: Dict = None,
) -> str:
    """Get plan progress summary."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    store = copilot.context_store
    campaign = store.get_campaign(campaign_id)
    if not campaign:
        return f"Campaign {campaign_id} not found"

    status = store.get_plan_status(campaign_id)

    lines = [f"Plan: {campaign.description}"]
    lines.append(f"Progress: {status['completed']}/{status['total']} items completed")

    if status["in_progress"] > 0:
        lines.append(f"In progress: {status['in_progress']}")

    # By type
    for type_name, counts in status["by_type"].items():
        lines.append(f"  {type_name}: {counts['completed']}/{counts['total']}")

    # Next actions
    if status["next_actions"]:
        lines.append("\nNext actions:")
        for item in status["next_actions"][:5]:
            lines.append(f"  [{item.type.value}] {item.title}")

    # Pending decisions
    if status["pending_decisions"]:
        lines.append("\nPending decisions:")
        for item in status["pending_decisions"]:
            lines.append(f"  {item.title}")
            if item.description:
                lines.append(f"    {item.description[:100]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch Operations
# ---------------------------------------------------------------------------

@tool(
    name="batch_update_status",
    description=(
        "Batch-update the status of plan items in a campaign or phase. "
        "Optionally filter by phase number and item type. Skips items "
        "with unresolved dependencies when marking complete."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Mark all phase 1 items as completed",
            tool_input={
                "campaign_id": "nrf-2026",
                "new_status": "completed",
                "outcome": "Phase 1 completed successfully",
                "phase_number": 1,
            },
        ),
    ],
)
async def batch_update_status(
    campaign_id: str,
    new_status: str,
    outcome: str = None,
    phase_number: int = None,
    item_type: str = None,
    context: Dict = None,
) -> str:
    """Batch-update status of plan items."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    from gently.context.model import PlanItemStatus

    store = copilot.context_store

    # Resolve phase number to a campaign ID
    target_campaign_id = campaign_id
    if phase_number is not None:
        phase = store.get_nth_subcampaign(campaign_id, phase_number)
        if not phase:
            return f"Phase {phase_number} not found under campaign {campaign_id}"
        target_campaign_id = phase.id

    # Get matching items
    items = store.get_plan_items(
        campaign_id=target_campaign_id,
        type=item_type,
    )

    status_enum = PlanItemStatus(new_status)
    updated = 0
    skipped = 0

    for item in items:
        # Guard: skip items with unresolved deps when marking complete
        if status_enum == PlanItemStatus.COMPLETED and item.depends_on:
            all_resolved = True
            for dep_id in item.depends_on:
                dep = store.get_plan_item(dep_id)
                if dep and dep.status not in (
                    PlanItemStatus.COMPLETED, PlanItemStatus.SKIPPED,
                ):
                    all_resolved = False
                    break
            if not all_resolved:
                skipped += 1
                continue

        kwargs = {"item_id": item.id, "status": status_enum}
        if outcome and status_enum == PlanItemStatus.COMPLETED:
            kwargs["outcome"] = outcome
        store.update_plan_item(**kwargs)
        updated += 1

    parts = [f"Updated {updated} items to '{new_status}'"]
    if skipped:
        parts.append(f"skipped {skipped} (unresolved dependencies)")
    return ", ".join(parts)


@tool(
    name="batch_update_spec",
    description=(
        "Batch-update a single spec field across all imaging items in a "
        "campaign or phase. Useful for globally adjusting parameters like "
        "interval_s, num_slices, or temperature_c."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Change all imaging intervals to 120s",
            tool_input={
                "campaign_id": "nrf-2026",
                "field_name": "interval_s",
                "field_value": 120,
            },
        ),
    ],
)
async def batch_update_spec(
    campaign_id: str,
    field_name: str,
    field_value: object,
    phase_number: int = None,
    context: Dict = None,
) -> str:
    """Batch-update a spec field on imaging items."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "Error: Context store not available"

    store = copilot.context_store

    # Validate field name against ImagingSpec
    from gently.context.model import ImagingSpec
    valid_fields = {f.name for f in dataclasses.fields(ImagingSpec)}
    if field_name not in valid_fields:
        return (
            f"'{field_name}' is not a valid ImagingSpec field. "
            f"Valid fields: {', '.join(sorted(valid_fields))}"
        )

    # Resolve phase
    target_campaign_id = campaign_id
    if phase_number is not None:
        phase = store.get_nth_subcampaign(campaign_id, phase_number)
        if not phase:
            return f"Phase {phase_number} not found under campaign {campaign_id}"
        target_campaign_id = phase.id

    items = store.get_plan_items(
        campaign_id=target_campaign_id,
        type="imaging",
    )

    updated = 0
    for item in items:
        # Merge the field into existing spec
        spec_data = {}
        if item.imaging_spec:
            spec_data = {
                f.name: getattr(item.imaging_spec, f.name)
                for f in dataclasses.fields(item.imaging_spec)
                if getattr(item.imaging_spec, f.name) is not None
            }
        spec_data[field_name] = field_value
        store.update_plan_item(item_id=item.id, spec=spec_data)
        updated += 1

    return f"Updated '{field_name}' to {field_value} on {updated} imaging items"
