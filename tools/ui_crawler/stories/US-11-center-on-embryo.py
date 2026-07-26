# ruff: noqa: E501
"""US-11 — Center on a marked embryo. As an operator, I click a registered embryo and the stage drives it to the middle of the field."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-11",
    "title": "Center on a marked embryo",
    "cluster": "5 Operate (mark)",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    opened = await view(page, "operate")
    on_pane = await view(page, "bottom")
    await rec.shot("operate-center-surface")

    canvas = await exists(page, "#op-mark-canvas")  # registered embryos are click targets here
    # Centering is offered whenever the sample is clear of the objective. The
    # interlock is state, not a step: if this banner is up, XY is refused.
    locked = await page.evaluate(
        "() => { const b = document.getElementById('op-lock-bottom'); return !!b && !b.hidden; }"
    )
    roster_center = await view(page, "acquire") and await dom_count(page, "#op-roster")

    if not (opened and on_pane and canvas):
        rec.gap("no click-to-center target on the bottom-cam surface")
        return
    if locked:
        rec.gap("XY interlock is engaged on a fresh load — centering would be refused")
        return
    rec.blocked(
        f"needs device: the bottom-cam canvas offers click-to-center on registered embryos and the roster carries a per-row Centre action (roster={roster_center}); "
        "both post /api/devices/stage/move, which 502s with the device layer offline"
    )
