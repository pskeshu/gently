# ruff: noqa: E501
"""US-10 — Mark embryos. As an operator, I look at the dish on the bottom camera and mark every embryo in one pass, automatically or by hand."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-10",
    "title": "Mark embryos on the bottom camera",
    "cluster": "5 Operate (mark)",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    opened = await view(page, "operate")
    on_pane = await view(page, "bottom")  # Operate → Bottom cam surface
    await rec.shot("operate-bottom-cam")

    canvas = await exists(page, "#op-mark-canvas")  # always markable — no mode to enter
    detect = await dom_count(page, "#op-detect")  # automatic detection
    confirm = await dom_count(page, "#op-confirm")  # register the marked set
    clear = await dom_count(page, "#op-clear")
    cam = await dom_count(page, "#op-cam-toggle")

    if not (opened and on_pane and canvas and detect and confirm):
        rec.gap("bottom-cam surface is missing the marking canvas or its detect/register controls")
        return
    rec.blocked(
        f"needs device: marking canvas + detect/register/clear are present and live from load (cam toggle={cam}, clear={clear}); "
        "capturing a FOV and running detection need the bottom camera (offline -> 502)"
    )
