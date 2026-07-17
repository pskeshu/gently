# ruff: noqa: E501
"""US-15 — Monitor / pause / stop / resume a run. As an operator with a live run, I want to watch it and pause, resume, or stop it, so I stay in control of the acquisition."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-15",
    "title": "Monitor / pause / stop / resume a run",
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
        "() => { const r=document.getElementById('op-rail'); if (r) r.dataset.active='running'; }"
    )  # reveal the gated run-spine step for the audit shot
    await rec.shot("operate-run-spine")
    spine = await dom_count(page, "#op-runspine")  # live tactic monitor
    pause = await dom_count(page, "#op-run-pause")  # Pause <-> Resume toggle
    stop = await dom_count(page, "#op-run-stop")  # "Stop run" button
    if surface and spine and pause and stop:
        rec.blocked(
            f"needs live run: run-spine monitor + Pause/Resume toggle + Stop present (pause={pause}, stop={stop}) but the running step (op-rail running) only appears once a timelapse is live, and pause/resume/stop hit /api/devices/timelapse/{{pause,resume,stop}} which 502 with the device layer offline"
        )
    else:
        rec.gap(
            f"run-spine controls incomplete (surface={surface}, spine={spine}, pause={pause}, stop={stop})"
        )
