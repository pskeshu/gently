# ruff: noqa: E501
"""US-14 — Start an adaptive timelapse. As an operator, I set a cadence and a stop condition and let the run adapt."""

from _harness import dom_count, goto, skip_landing, tab, view

META = {
    "id": "US-14",
    "title": "Start an adaptive timelapse",
    "cluster": "6 Operate (acquire)",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")
    await view(page, "acquire")
    # Segmented control, not a radio behind progressive disclosure.
    await page.click('[data-mode="adaptive"]', timeout=4000)
    await rec.shot("operate-adaptive-timelapse")

    interval = await dom_count(page, "#op-tl-interval")
    stopc = await dom_count(page, "#op-tl-stop")
    monitor = await dom_count(page, "#op-tl-monitor")
    start = await dom_count(page, "#op-run-start")
    panel_shown = await page.evaluate(
        "() => { const p = document.getElementById('op-panel-adaptive'); return !!p && !p.hidden; }"
    )

    if not (interval and stopc and start and panel_shown):
        rec.gap("adaptive-timelapse controls did not appear when the mode was selected")
        return
    rec.blocked(
        f"needs device/session: cadence + stop-condition + monitoring ({monitor}) and Start are present and reachable without marking anything first; "
        "/api/devices/timelapse/start 502s with the device layer offline"
    )
