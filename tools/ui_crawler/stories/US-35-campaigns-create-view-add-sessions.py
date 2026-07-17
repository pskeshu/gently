# ruff: noqa: E501
"""US-35 — Campaigns (create / view / add sessions). As a user, I want to create a
campaign, browse it, and attach sessions to its plan items, so my work has a spine."""

from _harness import count_text, goto, present, skip_landing, tab

META = {
    "id": "US-35",
    "title": "Campaigns (create / view / add sessions)",
    "cluster": "11 Memory & campaigns",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    opened = await tab(page, "plans")  # campaign workspace lives in Plans
    viewer = await present(page, "#navigator") and await present(page, "#canvas-content")
    create = await count_text(
        page, r"new campaign|create campaign|\+\s*campaign|add campaign|start.*campaign"
    )
    await rec.shot("plans-campaign-viewer")
    if opened and viewer and not create:
        rec.gap(
            "Plans tab shows the campaign VIEWER (navigator + canvas + inspector) and add-sessions lives in an item's inspector, but there is NO create-campaign control — campaigns are created only through the agent (create_campaign tool via landing/chat), never from the UI"
        )
    elif opened and viewer and create:
        rec.ok(
            f"campaign viewer + a create-campaign control ({create}) are both present in the Plans tab"
        )
    else:
        rec.gap("no campaign viewer surface reachable in the Plans tab")
