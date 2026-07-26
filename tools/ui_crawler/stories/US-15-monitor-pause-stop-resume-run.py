# ruff: noqa: E501
"""US-15 — Monitor, pause, resume and stop a run. As an operator, I watch what is running and can hold or end it."""

from _harness import dom_count, goto, skip_landing, tab, view

META = {
    "id": "US-15",
    "title": "Monitor, pause and stop a run",
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
    await rec.shot("operate-run-monitor")

    # Derived from /api/operation_plan, not a client flag.
    spine = await dom_count(page, "#op-runspine")
    pause = await dom_count(page, "#op-run-pause")
    stop = await dom_count(page, "#op-run-stop")
    # Pause/Stop are hidden until something is actually running — the panel is
    # derived from the server, so a reload mid-run does not lose it.
    actions_hidden = await page.evaluate(
        "() => { const a = document.getElementById('op-run-actions'); return !!a && a.hidden; }"
    )
    idle_msg = await dom_count(page, "#op-runspine .op-empty")

    if not (on_pane and spine and pause and stop):
        rec.gap("run monitor is missing the spine or its pause/stop controls")
        return
    if not actions_hidden:
        rec.gap("pause/stop are offered while nothing is running")
        return
    rec.blocked(
        f"needs live run: the run panel reads /api/operation_plan and correctly shows an idle state ({idle_msg}) with pause/stop withheld; "
        "exercising them needs a live timelapse, and pause/resume/stop 502 with the device layer offline"
    )
