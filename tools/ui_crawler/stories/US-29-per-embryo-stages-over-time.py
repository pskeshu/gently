# ruff: noqa: E501
"""US-29 — See per-embryo predicted stages over time. As a researcher, I want each embryo's predicted developmental stage tracked across timepoints, so I can watch progression."""

from _harness import count_text, goto, present, skip_landing, tab, view

META = {
    "id": "US-29",
    "title": "Per-embryo predicted stages over time",
    "cluster": "10 Perception & results",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "embryos")
    board_btn = await count_text(page, r"\bboard\b")  # dense status table w/ stage sparkline column
    vitals_btn = await count_text(page, r"\bvitals\b")  # per-embryo stage strip charts over time
    await view(page, "board")
    board = await present(page, "#view-board")
    await view(page, "vitals")
    vitals = await present(page, "#view-vitals")
    await rec.shot("vitals-view")
    if vitals_btn and board_btn and vitals and board:
        rec.partial(
            "Embryos tab exposes Board (per-embryo stage sparkline column) + Vitals (per-embryo stage strip charts across timepoints); the stage-over-time data populates from a live perception run — empty state ('No embryos to display') without one"
        )
    elif vitals or board:
        rec.partial("only one of the Board/Vitals stage-timeline views is present")
    else:
        rec.gap("no per-embryo stage-over-time view (Board/Vitals) in the embryos tab")
