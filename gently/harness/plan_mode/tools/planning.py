"""
Planning tools — create and manage experimental plans.

These tools let the agent build structured plans from
scientific conversations: campaigns, phases, plan items,
and their dependencies.
"""

import dataclasses
import json

from ...tools.registry import ToolCategory, ToolExample, tool


def _coerce_plan_args(spec, references, estimated_days):
    """The model often serializes nested args (spec/references) as JSON strings
    instead of objects — accept either so plan-item creation doesn't store a raw
    string (which later breaks ImagingSpec/BenchSpec hydration). Returns the
    normalized (spec, references, estimated_days)."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except (json.JSONDecodeError, TypeError):
            spec = None
    if isinstance(references, str):
        try:
            references = json.loads(references)
        except (json.JSONDecodeError, TypeError):
            references = None
    if isinstance(estimated_days, str):
        try:
            estimated_days = int(estimated_days)
        except (ValueError, TypeError):
            estimated_days = None
    return spec, references, estimated_days


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
    shorthand: str | None = None,
    target: str | None = None,
    parent_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Create a campaign or sub-campaign (phase)."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    # Fix year in shorthand — models sometimes hallucinate the wrong year
    if shorthand:
        import re
        from datetime import datetime

        current_year = str(datetime.now().year)
        shorthand = re.sub(r"-20\d{2}$", f"-{current_year}", shorthand)

    store = agent.context_store
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
        "Use depends_on to set dependencies on other plan item IDs. "
        "Use phase_number (e.g. 1) to add the item to a specific phase "
        "instead of looking up the subcampaign ID. "
        "Use estimated_days to indicate how many days this task takes "
        "(e.g. 1 for a quick imaging session, 14 for strain expansion). "
        "Use references to cite literature, databases, or other sources "
        "(each with source, citation, and optional id/note)."
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
    description: str | None = None,
    spec: dict | None = None,
    inherit_from: str | None = None,
    depends_on: list[str] | None = None,
    phase_number: int | None = None,
    phase_order: int = -1,
    references: list[dict] | None = None,
    estimated_days: int | None = None,
    context: dict | None = None,
) -> str:
    """Create a plan item within a campaign/phase.

    If phase_number is given (e.g. 1), the item is added to the Nth
    subcampaign (phase) of campaign_id instead of the root campaign.
    """
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    spec, references, estimated_days = _coerce_plan_args(spec, references, estimated_days)
    if isinstance(phase_number, str):
        try:
            phase_number = int(phase_number)
        except (ValueError, TypeError):
            phase_number = None
    if isinstance(phase_order, str):
        try:
            phase_order = int(phase_order)
        except (ValueError, TypeError):
            phase_order = -1

    # Resolve phase_number → subcampaign ID
    target_campaign_id = campaign_id
    if phase_number is not None:
        phase = store.get_nth_subcampaign(campaign_id, phase_number)
        if not phase:
            return (
                f"Phase {phase_number} not found under campaign {campaign_id}. "
                f"Use create_campaign with parent_id to create it first."
            )
        target_campaign_id = phase.id

    item_id = store.create_plan_item(
        campaign_id=target_campaign_id,
        type=type,
        title=title,
        description=description,
        spec=spec,
        inherit_from=inherit_from,
        depends_on=depends_on,
        phase_order=phase_order,
        references=references,
        estimated_days=estimated_days,
    )

    # Include the human-friendly task number in the response
    item = store.get_plan_item(item_id)
    task_num = ""
    if item:
        # Determine phase number if this is a subcampaign
        actual_campaign = store.get_campaign(target_campaign_id)
        if actual_campaign and actual_campaign.parent_id:
            phases = store.get_subcampaigns(actual_campaign.parent_id)
            for pi, phase in enumerate(phases, 1):
                if phase.id == target_campaign_id:
                    # Count items in this phase up to this one
                    items = store.get_plan_items(campaign_id=target_campaign_id)
                    items.sort(key=lambda x: x.phase_order)
                    for ti, it in enumerate(items, 1):
                        if it.id == item_id:
                            task_num = f" #{pi}.{ti}"
                            break
                    break
        else:
            items = store.get_plan_items(campaign_id=target_campaign_id)
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
        "outcome, spec, or references. Use this to mark items as completed, "
        "skipped, update imaging specifications, or attach source citations."
    ),
    category=ToolCategory.UTILITY,
)
async def update_plan_item(
    item_id: str,
    status: str | None = None,
    title: str | None = None,
    description: str | None = None,
    outcome: str | None = None,
    spec: dict | None = None,
    references: list[dict] | None = None,
    estimated_days: int | None = None,
    campaign_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Update a plan item. item_id can be a UUID, task number (e.g. '3'),
    or phase.task reference (e.g. '1.3'). campaign_id scopes resolution
    when using shorthand refs with multiple plans."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    # Resolve natural references
    item = store.resolve_plan_item(item_id, campaign_id=campaign_id)
    if not item:
        return f"Plan item '{item_id}' not found"
    resolved_id = item.id

    from gently.harness.memory.model import PlanItemStatus

    status_enum = PlanItemStatus(status) if status else None
    spec, references, estimated_days = _coerce_plan_args(spec, references, estimated_days)
    store.update_plan_item(
        item_id=resolved_id,
        status=status_enum,
        title=title,
        description=description,
        outcome=outcome,
        spec=spec,
        references=references,
        estimated_days=estimated_days,
    )
    changes = []
    if status:
        changes.append(f"status -> {status}")
    if outcome:
        changes.append("outcome recorded")
    if spec:
        changes.append("spec updated")
    if title:
        changes.append(f"title -> {title}")
    if references:
        changes.append(f"{len(references)} references attached")
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
    campaign_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Add a dependency between plan items. campaign_id scopes resolution
    when using shorthand refs (e.g. '1.3') with multiple plans."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    # Resolve natural references
    item = store.resolve_plan_item(item_id, campaign_id=campaign_id)
    dep = store.resolve_plan_item(depends_on_id, campaign_id=campaign_id)
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
    campaign_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Look up a plan item by natural reference."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
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
    context: dict | None = None,
) -> str:
    """Render the full plan for review."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    campaign = store.resolve_campaign(campaign_id)
    if not campaign:
        return f"Campaign '{campaign_id}' not found. Try a shorthand, name, or UUID."
    campaign_id = campaign.id

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
        # Show items directly in the root campaign (if any)
        root_items = store.get_plan_items(campaign_id=campaign_id)
        root_items.sort(key=lambda x: x.phase_order)
        if root_items:
            lines.append("── Unassigned (not in a phase) ──")
            lines.append("")
            for task_idx, item in enumerate(root_items, 1):
                lines.append(_format_plan_item(item, store, task_num=f"?.{task_idx}"))
            lines.append("")

        for phase_idx, phase in enumerate(phases, 1):
            lines.append(f"── Phase {phase_idx}: {phase.description}  (id: {phase.id}) ──")
            lines.append("")
            items = store.get_plan_items(campaign_id=phase.id)
            items.sort(key=lambda x: x.phase_order)
            for task_idx, item in enumerate(items, 1):
                num = f"{phase_idx}.{task_idx}"
                lines.append(_format_plan_item(item, store, task_num=num))
            lines.append("")

    # Summary
    status = store.get_plan_status(campaign_id)
    lines.append("── Summary ──")
    lines.append(f"Total items: {status['total']}")
    lines.append(f"Completed: {status['completed']}")
    if status["pending_decisions"]:
        lines.append(f"Pending decisions: {len(status['pending_decisions'])}")
    if status["next_actions"]:
        lines.append(f"Next actions: {', '.join(a.title for a in status['next_actions'][:3])}")

    return "\n".join(lines)


def _format_plan_item(item, store, task_num: str = "") -> str:
    """Format a single plan item for display."""
    from gently.harness.memory.model import PlanItemStatus

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

    if item.references:
        ref_strs = []
        for r in item.references:
            tag = f"[{r.get('source', '').upper()}]" if r.get("source") else ""
            cite = r.get("citation", "")
            ref_strs.append(f"{tag} {cite}")
        details.append(f"   Refs: {'; '.join(ref_strs)}")

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
    context: dict | None = None,
) -> str:
    """Get plan progress summary."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    campaign = store.resolve_campaign(campaign_id)
    if not campaign:
        return f"Campaign '{campaign_id}' not found. Try a shorthand, name, or UUID."
    campaign_id = campaign.id

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
    outcome: str | None = None,
    phase_number: int | None = None,
    item_type: str | None = None,
    context: dict | None = None,
) -> str:
    """Batch-update status of plan items."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    from gently.harness.memory.model import PlanItemStatus

    store = agent.context_store

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
                    PlanItemStatus.COMPLETED,
                    PlanItemStatus.SKIPPED,
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
    phase_number: int | None = None,
    context: dict | None = None,
) -> str:
    """Batch-update a spec field on imaging items."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    # Validate field name against ImagingSpec
    from gently.harness.memory.model import ImagingSpec

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


# ---------------------------------------------------------------------------
# Plan Reorganization
# ---------------------------------------------------------------------------


@tool(
    name="move_plan_item",
    description=(
        "Move a plan item to a different phase. Accepts the item by "
        "reference (UUID, task number, or phase.task) and the target "
        "phase by number or campaign ID. Optionally set the position "
        "within the new phase."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Move task 1.3 to phase 2",
            tool_input={
                "campaign_id": "nrf-2026",
                "item_ref": "1.3",
                "to_phase_number": 2,
            },
        ),
    ],
)
async def move_plan_item(
    campaign_id: str,
    item_ref: str,
    to_phase_number: int | None = None,
    to_campaign_id: str | None = None,
    position: int | None = None,
    context: dict | None = None,
) -> str:
    """Move a plan item to a different phase."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    item = store.resolve_plan_item(item_ref, campaign_id=campaign_id)
    if not item:
        return f"Plan item '{item_ref}' not found"

    # Resolve destination
    dest_id = to_campaign_id
    if to_phase_number is not None:
        phase = store.get_nth_subcampaign(campaign_id, to_phase_number)
        if not phase:
            return f"Phase {to_phase_number} not found under campaign {campaign_id}"
        dest_id = phase.id

    if not dest_id:
        return "Specify to_phase_number or to_campaign_id"

    if dest_id == item.campaign_id:
        return f"Item '{item.title}' is already in that phase"

    kwargs = {"item_id": item.id, "campaign_id": dest_id}
    if position is not None:
        kwargs["phase_order"] = position
    else:
        # Append to end of destination
        dest_items = store.get_plan_items(campaign_id=dest_id)
        max_order = max((i.phase_order for i in dest_items), default=0)
        kwargs["phase_order"] = max_order + 1

    store.update_plan_item(**kwargs)

    dest_campaign = store.get_campaign(dest_id)
    dest_label = dest_campaign.description if dest_campaign else dest_id
    return f"Moved '{item.title}' to {dest_label}"


@tool(
    name="delete_plan_item",
    description=(
        "Delete a plan item from the plan. Also removes all dependency "
        "links to/from this item. Use with caution."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Remove task 1.3 from the plan",
            tool_input={"item_ref": "1.3"},
        ),
    ],
)
async def delete_plan_item_tool(
    item_ref: str,
    campaign_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Delete a plan item."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    item = store.resolve_plan_item(item_ref, campaign_id=campaign_id)
    if not item:
        return f"Plan item '{item_ref}' not found"

    # Auto-snapshot before destructive operation
    try:
        root_id = item.campaign_id
        parent_campaign = store.get_campaign(root_id)
        if parent_campaign and parent_campaign.parent_id:
            root_id = parent_campaign.parent_id
        store.create_plan_snapshot(
            root_id,
            label=f"auto: before deleting item '{item.title}'",
        )
    except Exception:
        pass  # Don't block deletion if snapshot fails

    title = item.title
    deleted = store.delete_plan_item(item.id)
    if deleted:
        return f"Deleted plan item '{title}' and its dependency links"
    return f"Plan item '{title}' could not be deleted"


@tool(
    name="reorder_plan_items",
    description=(
        "Set the display order of plan items within a phase. Pass an "
        "ordered list of item references — they will be assigned "
        "sequential phase_order values. Items not listed keep their "
        "current position (appended after listed items)."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Reorder phase 1: put the bench task first, then imaging",
            tool_input={
                "campaign_id": "nrf-2026",
                "phase_number": 1,
                "item_order": ["1.2", "1.1", "1.3"],
            },
        ),
    ],
)
async def reorder_plan_items(
    campaign_id: str,
    item_order: list[str],
    phase_number: int | None = None,
    context: dict | None = None,
) -> str:
    """Reorder plan items within a phase."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    target_campaign_id = campaign_id
    if phase_number is not None:
        phase = store.get_nth_subcampaign(campaign_id, phase_number)
        if not phase:
            return f"Phase {phase_number} not found under campaign {campaign_id}"
        target_campaign_id = phase.id

    # Resolve all referenced items
    resolved_ids = []
    for ref in item_order:
        item = store.resolve_plan_item(ref, campaign_id=campaign_id)
        if not item:
            return f"Plan item '{ref}' not found"
        resolved_ids.append(item.id)

    # Get remaining items not in the list
    all_items = store.get_plan_items(campaign_id=target_campaign_id)
    all_items.sort(key=lambda x: x.phase_order)
    remaining = [i for i in all_items if i.id not in resolved_ids]

    # Assign new order
    order = 1
    for item_id in resolved_ids:
        store.update_plan_item(item_id=item_id, phase_order=order)
        order += 1
    for item in remaining:
        store.update_plan_item(item_id=item.id, phase_order=order)
        order += 1

    return f"Reordered {len(resolved_ids)} items (+ {len(remaining)} unchanged)"


@tool(
    name="update_phase",
    description=(
        "Update a phase's description, shorthand, or target. "
        "Identify the phase by number within a campaign."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Rename phase 1 to 'Reporter Validation'",
            tool_input={
                "campaign_id": "nrf-2026",
                "phase_number": 1,
                "description": "Reporter Validation",
            },
        ),
    ],
)
async def update_phase(
    campaign_id: str,
    phase_number: int,
    description: str | None = None,
    shorthand: str | None = None,
    target: str | None = None,
    context: dict | None = None,
) -> str:
    """Update a phase's metadata."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    phase = store.get_nth_subcampaign(campaign_id, phase_number)
    if not phase:
        return f"Phase {phase_number} not found under campaign {campaign_id}"

    store.update_campaign(
        campaign_id=phase.id,
        description=description,
        shorthand=shorthand,
        target=target,
    )

    changes = []
    if description:
        changes.append(f"description → '{description}'")
    if shorthand:
        changes.append(f"shorthand → '{shorthand}'")
    if target:
        changes.append(f"target → '{target}'")

    return f"Updated Phase {phase_number}: {', '.join(changes) or 'no changes'}"


@tool(
    name="delete_phase",
    description=(
        "Delete a phase and all its plan items. Use with caution — "
        "this cannot be undone. Items and dependency links are removed."
    ),
    category=ToolCategory.UTILITY,
)
async def delete_phase(
    campaign_id: str,
    phase_number: int,
    context: dict | None = None,
) -> str:
    """Delete a phase and its contents."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    phase = store.get_nth_subcampaign(campaign_id, phase_number)
    if not phase:
        return f"Phase {phase_number} not found under campaign {campaign_id}"

    # Auto-snapshot before destructive operation
    try:
        store.create_plan_snapshot(
            campaign_id,
            label=f"auto: before deleting phase {phase_number} ({phase.description})",
        )
    except Exception:
        pass  # Don't block deletion if snapshot fails

    desc = phase.description
    counts = store.delete_campaign(phase.id, cascade=True)
    return (
        f"Deleted Phase {phase_number} ('{desc}'): "
        f"{counts.get('plan_items', 0)} items, "
        f"{counts.get('dependencies', 0)} dependency links removed"
    )


# ---------------------------------------------------------------------------
# Plan Export
# ---------------------------------------------------------------------------


@tool(
    name="export_plan",
    description=(
        "Export the experimental plan as a clean, shareable markdown "
        "document. Suitable for emailing to collaborators, printing, "
        "or saving as a file. Includes campaign summary, phases, "
        "all items with specs, dependencies, and timeline."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Export this plan so I can send it to my PI",
            tool_input={"campaign_id": "nrf-2026"},
        ),
    ],
)
async def export_plan(
    campaign_id: str,
    include_validation: bool = False,
    context: dict | None = None,
) -> str:
    """Export a plan as a shareable markdown document."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    campaign = store.resolve_campaign(campaign_id)
    if not campaign:
        return f"Campaign '{campaign_id}' not found. Try a shorthand, name, or UUID."
    campaign_id = campaign.id

    lines = []

    # Title
    lines.append(f"# {campaign.description}")
    lines.append("")
    if campaign.shorthand:
        lines.append(f"**Campaign:** {campaign.shorthand}")
    if campaign.target:
        lines.append(f"**Goal:** {campaign.target}")
    if campaign.summary:
        lines.append("")
        lines.append(campaign.summary)
    lines.append("")
    lines.append(f"*Exported: {_export_date()}*")
    lines.append("")

    # Status overview
    status = store.get_plan_status(campaign_id)
    lines.append("## Status")
    lines.append("")
    lines.append(f"- **Total items:** {status['total']}")
    lines.append(f"- **Completed:** {status['completed']}")
    if status["in_progress"]:
        lines.append(f"- **In progress:** {status['in_progress']}")
    if status["planned"]:
        lines.append(f"- **Planned:** {status['planned']}")
    lines.append("")

    # Phases and items
    phases = store.get_subcampaigns(campaign_id)

    if not phases:
        lines.append("## Plan Items")
        lines.append("")
        items = store.get_plan_items(campaign_id=campaign_id)
        items.sort(key=lambda x: x.phase_order)
        for task_idx, item in enumerate(items, 1):
            lines.extend(_export_item(item, store, str(task_idx)))
    else:
        for phase_idx, phase in enumerate(phases, 1):
            lines.append(f"## Phase {phase_idx}: {phase.description}")
            if phase.target:
                lines.append(f"*{phase.target}*")
            lines.append("")

            items = store.get_plan_items(campaign_id=phase.id)
            items.sort(key=lambda x: x.phase_order)

            if not items:
                lines.append("*No items in this phase.*")
                lines.append("")
                continue

            for task_idx, item in enumerate(items, 1):
                num = f"{phase_idx}.{task_idx}"
                lines.extend(_export_item(item, store, num))
            lines.append("")

    # References appendix — collect all unique references across items
    all_items = store.get_plan_items(campaign_id=campaign_id, include_children=True)
    all_refs = []
    seen_ids = set()
    for item in all_items:
        for ref in item.references:
            ref_key = ref.get("id") or ref.get("citation", "")
            if ref_key and ref_key not in seen_ids:
                seen_ids.add(ref_key)
                all_refs.append(ref)

    if all_refs:
        lines.append("---\n## References\n")
        for i, r in enumerate(all_refs, 1):
            tag = f"[{r.get('source', '').upper()}]" if r.get("source") else ""
            cite = r.get("citation", "")
            rid = r.get("id", "")
            note = r.get("note", "")
            line = f"{i}. {tag} {cite}"
            if rid:
                line += f" ({rid})"
            if note:
                line += f" — *{note}*"
            lines.append(line)
        lines.append("")

    # Validation report (optional)
    if include_validation:
        lines.append("## Validation Report")
        lines.append("")
        try:
            report = await validate_plan_for_export(campaign_id, store)
            lines.append(report)
        except Exception:
            lines.append("*Validation could not be run.*")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by Gently Agent*")

    return "\n".join(lines)


def _export_date() -> str:
    """Return current date in human-readable format."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d")


def _export_item(item, store, num: str) -> list[str]:
    """Format a plan item for the export document."""
    from gently.harness.memory.model import PlanItemStatus

    status_labels = {
        PlanItemStatus.PLANNED: "Planned",
        PlanItemStatus.IN_PROGRESS: "In Progress",
        PlanItemStatus.COMPLETED: "Completed",
        PlanItemStatus.SKIPPED: "Skipped",
        PlanItemStatus.BLOCKED: "Blocked",
    }
    status_label = status_labels.get(item.status, item.status.value)

    lines = []
    lines.append(f"### {num}. [{item.type.value.upper()}] {item.title}")
    lines.append(f"**Status:** {status_label}")
    lines.append("")

    if item.description:
        lines.append(item.description)
        lines.append("")

    spec = store.resolve_imaging_spec(item) if item.imaging_spec else None
    if spec:
        lines.append("**Imaging Parameters:**")
        if spec.strain:
            reporter = f" ({spec.reporter})" if spec.reporter else ""
            lines.append(f"- Strain: {spec.strain}{reporter}")
        if spec.num_slices:
            lines.append(f"- Z-slices: {spec.num_slices}")
        if spec.exposure_ms:
            lines.append(f"- Exposure: {spec.exposure_ms} ms")
        if spec.interval_s:
            lines.append(f"- Interval: {spec.interval_s} s")
        if spec.laser_power_pct is not None:
            wl = f" ({spec.laser_wavelength_nm} nm)" if spec.laser_wavelength_nm else ""
            lines.append(f"- Laser power: {spec.laser_power_pct}%{wl}")
        if spec.stop_condition:
            lines.append(f"- Stop condition: {spec.stop_condition}")
        elif spec.target_window:
            lines.append(f"- Target window: {spec.target_window}")
        if spec.num_embryos:
            lines.append(f"- Embryos: {spec.num_embryos}")
        if spec.temperature_c:
            lines.append(f"- Temperature: {spec.temperature_c} C")
        if spec.detectors:
            lines.append(f"- Detectors: {', '.join(spec.detectors)}")
        if spec.success_criteria:
            lines.append(f"- Success criteria: {spec.success_criteria}")
        lines.append("")

    if item.bench_spec:
        bs = item.bench_spec
        lines.append("**Bench Details:**")
        if bs.protocol:
            lines.append(f"- Protocol: {bs.protocol}")
        if bs.strains:
            lines.append(f"- Strains: {', '.join(bs.strains)}")
        if bs.target_genotype:
            lines.append(f"- Target genotype: {bs.target_genotype}")
        if bs.estimated_days:
            lines.append(f"- Estimated time: ~{bs.estimated_days} days")
        if bs.success_criteria:
            lines.append(f"- Success criteria: {bs.success_criteria}")
        lines.append("")

    if item.depends_on:
        dep_items = [store.get_plan_item(d) for d in item.depends_on]
        dep_names = [d.title for d in dep_items if d]
        if dep_names:
            lines.append(f"**Depends on:** {', '.join(dep_names)}")
            lines.append("")

    if item.outcome:
        lines.append(f"**Outcome:** {item.outcome}")
        lines.append("")

    if item.references:
        lines.append("**References:**")
        for r in item.references:
            tag = f"[{r.get('source', '').upper()}]" if r.get("source") else ""
            cite = r.get("citation", "")
            rid = r.get("id", "")
            note = r.get("note", "")
            line = f"- {tag} {cite}"
            if rid:
                line += f" ({rid})"
            if note:
                line += f" — *{note}*"
            lines.append(line)
        lines.append("")

    return lines


async def validate_plan_for_export(campaign_id: str, store) -> str:
    """Run validation and return a markdown-formatted report for export."""
    from .validation import (
        CONTROL_KEYWORDS,
        HARDWARE_LIMITS,
        _check_dependency_cycles,
    )

    items = store.get_plan_items(campaign_id=campaign_id, include_children=True)
    if not items:
        return "No items to validate."

    try:
        from gently.organisms import get_organism

        org = get_organism()
        presets_mod = __import__(
            f"gently.organisms.{org.ORGANISM_NAME}.detector_presets",
            fromlist=["get_detector_presets"],
        )
        set(presets_mod.get_detector_presets().keys())
    except ImportError:
        pass

    issues = []
    has_control = False

    for item in items:
        label = f"{item.title}"
        text_blob = " ".join(filter(None, [item.title, item.description])).lower()
        if item.imaging_spec:
            text_blob += (
                " "
                + " ".join(
                    filter(
                        None,
                        [
                            item.imaging_spec.strain,
                            item.imaging_spec.genotype,
                        ],
                    )
                ).lower()
            )
        if any(kw in text_blob for kw in CONTROL_KEYWORDS):
            has_control = True

        spec = store.resolve_imaging_spec(item) if item.imaging_spec else None
        if item.type.value == "imaging" and spec:
            for field_name, (lo, hi) in HARDWARE_LIMITS.items():
                val = getattr(spec, field_name, None)
                if val is None:
                    continue
                if lo is not None and val < lo:
                    issues.append(f"- **Error:** {label} — {field_name}={val} below min {lo}")
                if hi is not None and val > hi:
                    issues.append(f"- **Error:** {label} — {field_name}={val} exceeds max {hi}")

    cycle_errors = _check_dependency_cycles(items)
    for cyc in cycle_errors:
        issues.append(f"- **Error:** {cyc}")

    if not has_control:
        issues.append("- **Warning:** No control condition found in the plan")

    if not issues:
        return "All checks passed."
    return "\n".join(issues)


# ---------------------------------------------------------------------------
# Plan Versioning
# ---------------------------------------------------------------------------


@tool(
    name="snapshot_plan",
    description=(
        "Save a snapshot of the current plan state. Use this before major "
        "revisions to preserve the current version. Snapshots are also "
        "created automatically before destructive operations."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Save the current plan before we revise it",
            tool_input={
                "campaign_id": "nrf-2026",
                "label": "before PI feedback revision",
            },
        ),
    ],
)
async def snapshot_plan(
    campaign_id: str,
    label: str | None = None,
    context: dict | None = None,
) -> str:
    """Save a snapshot of the current plan."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    campaign = store.resolve_campaign(campaign_id)
    if not campaign:
        return f"Campaign '{campaign_id}' not found. Try a shorthand, name, or UUID."
    campaign_id = campaign.id

    version_id = store.create_plan_snapshot(campaign_id, label=label)

    # Look up the version number
    snapshot = store.get_plan_snapshot(version_id)
    version_number = snapshot["version_number"] if snapshot else "?"

    label_str = f" — {label}" if label else ""
    return f"Snapshot v{version_number} created (id: {version_id}){label_str}"


@tool(
    name="list_plan_versions",
    description=(
        "List all saved versions (snapshots) of a plan. Shows version "
        "number, label, summary, and timestamp for each."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Show me the version history for this plan",
            tool_input={"campaign_id": "nrf-2026"},
        ),
    ],
)
async def list_plan_versions(
    campaign_id: str,
    context: dict | None = None,
) -> str:
    """List saved plan versions."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    snapshots = store.list_plan_snapshots(campaign_id)

    if not snapshots:
        return f"No snapshots found for campaign {campaign_id}"

    lines = [f"Plan versions ({len(snapshots)} snapshots):"]
    lines.append("")
    for s in snapshots:
        label = f"  [{s['label']}]" if s.get("label") else ""
        summary_line = ""
        if s.get("summary"):
            first_line = s["summary"].split("\n")[0]
            summary_line = f"  {first_line}"
        lines.append(f"  v{s['version_number']}{label}  ({s['version_id']})  {s['created_at']}")
        if summary_line:
            lines.append(f"    {summary_line}")

    return "\n".join(lines)


@tool(
    name="restore_plan_version",
    description=(
        "Restore a plan to a previous version. The current state is "
        "auto-saved before restoring. Accepts either a version_id or "
        "version_number. Returns the new campaign ID."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Roll back to version 2 of the plan",
            tool_input={
                "campaign_id": "nrf-2026",
                "version_number": 2,
            },
        ),
    ],
)
async def restore_plan_version(
    campaign_id: str,
    version_id: str | None = None,
    version_number: int | None = None,
    context: dict | None = None,
) -> str:
    """Restore a plan to a previous snapshot."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    if not version_id and version_number is None:
        return "Provide either version_id or version_number"

    # Resolve version_number to version_id
    if version_number is not None and not version_id:
        snapshots = store.list_plan_snapshots(campaign_id)
        match = next(
            (s for s in snapshots if s.get("version_number") == version_number),
            None,
        )
        if not match:
            return f"Version {version_number} not found for campaign {campaign_id}"
        version_id = match["version_id"]

    try:
        new_campaign_id = store.restore_plan_snapshot(version_id)
    except ValueError as e:
        return str(e)

    snapshot = store.get_plan_snapshot(version_id)
    v_label = f"v{snapshot['version_number']}" if snapshot else version_id
    return (
        f"Restored plan to {v_label}. "
        f"New campaign ID: {new_campaign_id} "
        f"(previous state was auto-saved before restoring)"
    )
