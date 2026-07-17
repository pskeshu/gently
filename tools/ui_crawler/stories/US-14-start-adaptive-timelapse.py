# ruff: noqa: E501
"""US-14 — Start an adaptive timelapse. As an operator with marked embryos, I want to start an adaptive timelapse, so imaging runs at cadence without me babysitting it."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-14",
    "title": "Start an adaptive timelapse",
    "cluster": "7 Timelapse",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")
    surface = await exists(page, "#devices-view-operate")  # the Operator Spine
    await page.evaluate(
        "() => { const r=document.getElementById('op-rail'); if (r) r.dataset.active='c0'; }"
    )  # reveal the gated run-chooser step for the audit shot
    await rec.shot("operate-run-chooser")
    adaptive = await dom_count(page, 'input[name="op-mode"][value="adaptive"]')  # run-mode radio
    interval = await dom_count(page, "#op-tl-interval")  # cadence field
    stopc = await dom_count(page, "#op-tl-stop")  # stop-condition select
    start = await dom_count(page, "#op-run-start")  # "Start run" button
    if surface and adaptive and interval and stopc and start:
        rec.blocked(
            f"needs device/session: adaptive-timelapse chooser present (mode radio + interval/stop select + 'Start run', start={start}) but the run-chooser step (op-rail c0) only unlocks after embryos are marked on live hardware, and /api/devices/timelapse/start 502s with the device layer offline"
        )
    else:
        rec.gap(
            f"adaptive-timelapse start control incomplete (surface={surface}, adaptive={adaptive}, interval={interval}, stop={stopc}, start={start})"
        )
