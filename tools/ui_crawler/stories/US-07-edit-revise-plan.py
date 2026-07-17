# ruff: noqa: E501
"""US-07 — Edit / revise an existing plan. As a researcher, I want to revise an existing plan item (its imaging spec, its linked sessions) from the Plans tab, so I don't have to re-plan from scratch."""

from _harness import dom_count, goto, skip_landing, tab

META = {
    "id": "US-07",
    "title": "Edit / revise an existing plan",
    "cluster": "3 Planning (access)",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "plans")
    await page.wait_for_timeout(1200)  # campaigns auto-load + auto-select the first
    await rec.shot("plans-tab")
    items = await dom_count(page, ".doc-item")  # plan-item cards in the canvas (doc view)
    if items:
        try:
            await page.click(".doc-item", timeout=5000)  # open the first item → inspector
            await page.wait_for_timeout(800)
        except Exception:
            pass
    spec_edit = await dom_count(
        page, '[data-action="spec-edit"]'
    )  # inline imaging-spec ✎ Edit → Save
    sess_link = await dom_count(
        page, '[data-action="session-picker-open"]'
    )  # + link session / delink ×
    await rec.shot("item-inspector")
    if spec_edit or sess_link:
        rec.partial(
            f"item inspector exposes inline field edits (imaging-spec edit={spec_edit}, session link/delink={sess_link}); structural edits — add/remove/reorder items, dependencies — have no UI and stay agent-only"
        )
    elif items:
        rec.partial(
            f"opened a plan item ({items} cards) but no inline edit control surfaced — spec/session edits need a spec-bearing imaging item; structural edits are agent-only"
        )
    else:
        rec.partial(
            "plans tab loads but no plan data in dev to open the item inspector; the inline spec-edit + session-link controls live there, and structural edits (add/remove/reorder) remain agent-only"
        )
