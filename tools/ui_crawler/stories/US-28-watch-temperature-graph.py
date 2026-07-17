# ruff: noqa: E501
"""US-28 — Watch the temperature graph. As an operator, I want a live water-temp vs
setpoint trace on the Devices tab, so I can watch the controller track during a run."""

from _harness import exists, goto, skip_landing, tab

META = {
    "id": "US-28",
    "title": "Watch the temperature graph",
    "cluster": "9 Temperature",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    graph = await exists(
        page, "#devices-temp-graph"
    )  # static SVG-chart surface, min-height keeps it visible
    empty = await exists(
        page, ".temp-graph-empty"
    )  # calm "No temperature data yet" (init ran, no data)
    await rec.shot("temp-graph")
    if graph and empty:
        rec.partial(
            "temperature graph surface present on the Devices tab but showing its empty state ('No temperature data yet') — the live water/setpoint trace needs a running session feeding TEMPERATURE_UPDATE events + /api/temperature/{session}/history backfill"
        )
    elif graph:
        rec.partial(
            "temperature graph section present on the Devices tab; the live trace needs a running session"
        )
    else:
        rec.gap("no temperature graph surface on the Devices tab")
