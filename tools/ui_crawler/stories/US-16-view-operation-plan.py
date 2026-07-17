# ruff: noqa: E501
"""US-16 — View the Operation Plan. As an operator, I open the Operations tab and see the agent's tactic plan (the spine) so I know what the run will do."""

from _harness import dom_count, exists, goto, skip_landing, tab

META = {
    "id": "US-16",
    "title": "View the Operation Plan",
    "cluster": "8 Operations & tactics",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    # ?scenario=<name> renders a plan fixture with no backend (the dev/audit path);
    # a real spine needs a live experiment, which the headless build has none of.
    await goto(page, url, "/?scenario=temp_strain")
    await skip_landing(page)
    await tab(page, "experiment")  # nav label "Operations"
    title = await exists(page, ".ops-title")
    spine = await exists(page, ".ops-spine")
    nodes = await dom_count(page, ".ops-node")  # one card per tactic
    await rec.shot("operation-plan")
    if title and spine and nodes >= 2:
        rec.ok(
            f"Operations tab renders the plan: title + tactic spine with {nodes} tactic cards (via audit scenario fixture; live needs a running experiment)"
        )
    elif spine or nodes:
        rec.partial(
            f"spine present but plan render is thin (title={title}, spine={spine}, nodes={nodes})"
        )
    else:
        rec.gap("Operations tab shows no plan/tactic spine")
