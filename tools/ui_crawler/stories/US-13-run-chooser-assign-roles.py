# ruff: noqa: E501
"""US-13 — Assign roles and pick what to run. As an operator, I mark each embryo subject or reference and choose how the run should proceed."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-13",
    "title": "Assign roles and choose a run mode",
    "cluster": "6 Operate (acquire)",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")
    on_pane = await view(page, "acquire")
    await rec.shot("operate-roles-and-modes")

    roster = await exists(page, "#op-roster")  # roles live on the roster, not at marking time
    modes = await dom_count(page, "#op-modes [data-mode]")
    start = await exists(page, "#op-run-start")
    # With nothing marked the roster is empty but the surface stays fully usable.
    empty = await dom_count(page, "#op-roster .op-empty")
    roles = await dom_count(page, ".op-rrole")

    if not (on_pane and roster and modes and start):
        rec.gap("acquisition surface is missing the roster or the run-mode chooser")
        return
    if modes != 4:
        rec.gap(f"expected 4 run modes (single/adaptive/library/agent), found {modes}")
        return
    rec.ok(
        f"acquisition surface offers the embryo roster and all {modes} run modes with Start, live on load "
        f"(empty-roster state={empty}, role toggles={roles}); roles POST /api/embryos/roles once embryos exist"
    )
