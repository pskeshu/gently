# ruff: noqa: E501
"""US-10 — Mark embryos. As an operator, I mark each embryo on the bottom-cam frame so they land in one canonical worklist to image."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-10",
    "title": "Mark embryos (bottom-cam detect → single list)",
    "cluster": "5 Operate (mark)",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    opened = await view(page, "operate")  # Devices → Operate spine
    await rec.shot("operate-marking-surface")
    tomark = await exists(page, "#op-tomark")  # a1 group: "Mark embryos →" entry (visible)
    board = await exists(page, "#op-board")  # the single canonical worklist ("The plan")
    mark_step = await dom_count(page, '[data-node="a2"]')  # "Mark" node in the stepper
    detect = await dom_count(
        page, "#op-detect"
    )  # a2 group: Detect (SAM) — hidden until marking active
    confirm = await dom_count(page, "#op-confirm")  # a2 group: Confirm marks into the list
    if opened and tomark and board and mark_step and detect and confirm:
        rec.partial(
            "operate view exposes the marking entry (#op-tomark), SAM detect + confirm-into-list controls, and one canonical worklist (#op-board); capturing a FOV + placing/detecting markers needs the bottom-cam device (offline → 502)"
        )
    elif opened and board:
        rec.partial(
            f"operate surface present but a marking affordance is missing (tomark={tomark}, detect={detect}, confirm={confirm})"
        )
    else:
        rec.gap(
            "operate/marking surface not reachable — no bottom-cam marking UI or single embryo worklist"
        )
