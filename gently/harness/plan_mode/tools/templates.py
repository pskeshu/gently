"""
Plan templates — save, list, and apply reusable experimental plan templates.

Templates capture an entire campaign structure (phases, items, specs,
dependencies) for re-use with different strains, temperatures, etc.
"""

from ...tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="save_plan_template",
    description=(
        "Save the current campaign as a reusable template. Captures the "
        "full plan structure: phases, items, specs, and dependencies. "
        "The template can later be applied with overrides (e.g. different "
        "strain or temperature)."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Save this plan as a template called 'standard-hatching'",
            tool_input={
                "campaign_id": "nrf-2026",
                "name": "standard-hatching",
                "description": "Standard hatching observation protocol with pilot + full imaging",
            },
        ),
    ],
)
async def save_plan_template(
    campaign_id: str,
    name: str,
    description: str | None = None,
    context: dict | None = None,
) -> str:
    """Save a campaign as a reusable template."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    try:
        tid = store.save_plan_template(
            name=name,
            description=description,
            campaign_id=campaign_id,
        )
        return f"Saved template '{name}' (id: {tid})"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            return f"Error: A template named '{name}' already exists. Choose a different name."
        return f"Error saving template: {e}"


@tool(
    name="list_templates",
    description=(
        "List all saved plan templates. Shows name, description, and "
        "creation date for each template."
    ),
    category=ToolCategory.UTILITY,
)
async def list_templates(
    context: dict | None = None,
) -> str:
    """List available plan templates."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store
    templates = store.list_plan_templates()

    if not templates:
        return "No plan templates saved yet."

    lines = [f"Plan Templates ({len(templates)}):"]
    lines.append("")
    for t in templates:
        lines.append(f"  {t['name']}  (id: {t['id']})")
        if t.get("description"):
            lines.append(f"    {t['description']}")
        lines.append(f"    Created: {t['created_at'][:10]}")
        lines.append("")

    return "\n".join(lines)


@tool(
    name="apply_template",
    description=(
        "Create a new campaign from a saved template. Optionally override "
        "fields like strain, temperature, or interval that get applied to "
        "all imaging specs. Returns the new campaign ID."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Apply the standard-hatching template for strain OH904 at 25°C",
            tool_input={
                "template_id": "standard-hatching",
                "overrides": {"strain": "OH904", "temperature_c": 25.0},
            },
        ),
    ],
)
async def apply_template(
    template_id: str,
    overrides: dict | None = None,
    context: dict | None = None,
) -> str:
    """Instantiate a template into a new campaign."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "Error: Context store not available"

    store = agent.context_store

    try:
        new_campaign_id = store.apply_plan_template(
            template_id=template_id,
            overrides=overrides,
        )
    except ValueError as e:
        return f"Error: {e}"

    campaign = store.get_campaign(new_campaign_id)
    items = store.get_plan_items(campaign_id=new_campaign_id, include_children=True)

    lines = [f"Created campaign from template: '{campaign.description}' (id: {new_campaign_id})"]
    lines.append(f"Total plan items: {len(items)}")

    if overrides:
        lines.append("Applied overrides:")
        for k, v in overrides.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)
