# ruff: noqa: E501
"""US-11 — Center on a marked embryo. As an operator, I pick a marked embryo and drive the stage to center it in the FOV."""

from _harness import dom_count, goto, skip_landing, tab, view

META = {"id": "US-11", "title": "Center on a marked embryo",
        "cluster": "5 Operate (mark)", "mode": "rig", "needs_account": False}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    opened = await view(page, "operate")                     # Devices → Operate spine
    await rec.shot("operate-center-surface")
    center_node = await dom_count(page, '[data-node="b1"]')  # "Center" step in the stepper
    center_btn = await dom_count(page, "#op-center")         # b1 group: "Center stage on embryo" (hidden until an embryo is selected)
    if not (opened and center_node and center_btn):
        rec.gap("no per-embryo Center control on the operate surface")
        return
    rec.blocked("needs device: reaching B1/Center requires a marked embryo (bottom-cam) and #op-center posts /api/devices/stage/move — both 502 against the offline device layer")
