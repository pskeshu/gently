# ruff: noqa: E501
"""US-30 — Inspect a perception trace / VLM reasoning. As a researcher, I want to open an embryo's evaluation and read the VLM's stage reasoning behind a call, so I can trust or audit it."""

from _harness import exists, goto, present, skip_landing, tab, view

META = {
    "id": "US-30",
    "title": "Inspect a perception trace / VLM reasoning",
    "cluster": "10 Perception & results",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "embryos")
    await view(page, "default")
    panel = await exists(page, "#reasoning-panel")  # right-hand VLM analysis / trace detail
    rail = await present(page, "#embryo-list")  # per-embryo rail to select a timepoint's trace
    await rec.shot("reasoning-panel")
    if panel and rail:
        rec.partial(
            "Embryos → Default shows the per-embryo rail + VLM reasoning panel; detection cards render 'Show VLM reasoning' with classifier/perceiver text, raw trace JSON and 'View All Projections' — all populated by a live perception run (empty state 'Select an embryo to view analysis history' without one)"
        )
    elif panel:
        rec.partial("reasoning/trace panel present but the embryo-selection rail is missing")
    else:
        rec.gap("no perception-trace / VLM-reasoning inspection surface in the embryos tab")
