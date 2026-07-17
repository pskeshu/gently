# ruff: noqa: E501
"""US-31 — Correct / annotate ground truth. As a researcher, I want to set the true developmental stage for a timepoint, so my correction becomes ground truth the system can learn from."""

from _harness import count_text, goto, skip_landing, tab, view

META = {
    "id": "US-31",
    "title": "Correct / annotate ground truth",
    "cluster": "10 Perception & results",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "embryos")
    await view(page, "default")
    correct = await count_text(
        page,
        r"correct stage|set stage|true stage|ground truth|annotate|edit stage|fix stage|override stage",
    )
    feedback = await count_text(
        page, r"i agree|i disagree"
    )  # the only feedback affordance (localStorage-only)
    await rec.shot("no-ground-truth-control")
    if correct:
        rec.ok(f"a set-correct-stage / annotate-ground-truth control is present ({correct})")
    else:
        rec.gap(
            f"no way to set/annotate the correct ground-truth stage — the only feedback is binary I Agree / I Disagree ({feedback} visible), which is saved to localStorage (not persisted as ground truth) and only appears on detection cards produced by a live perception run"
        )
