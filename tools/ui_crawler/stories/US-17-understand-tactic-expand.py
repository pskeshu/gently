# ruff: noqa: E501
"""US-17 — Understand a tactic (expand its card). As an operator, I click a tactic card to reveal its rationale/scope/structure/relations so I understand why and how it runs."""

from _harness import dom_count, goto, skip_landing, tab

META = {
    "id": "US-17",
    "title": "Understand a tactic (expand its card)",
    "cluster": "8 Operations & tactics",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url, "/?scenario=temp_strain")  # fixture with done/planned expandable tactics
    await skip_landing(page)
    await tab(page, "experiment")
    cards = await dom_count(
        page, ".ops-card[data-tactic-expand-id]"
    )  # queued/done/paused are expandable
    before = await dom_count(page, ".ops-expand-body:not(.hidden)")  # 0 — all collapsed on render
    await page.evaluate(
        "() => { const c=document.querySelector('.ops-card[data-tactic-expand-id]'); if (c) c.click(); }"
    )
    await page.wait_for_timeout(300)
    after = await dom_count(page, ".ops-expand-body:not(.hidden)")
    rows = await dom_count(
        page, ".ops-expand-body:not(.hidden) .ops-expand-row"
    )  # rationale/scope/structure/relations rows
    await rec.shot("tactic-expanded")
    if cards and after > before and rows >= 2:
        rec.ok(
            f"clicking a tactic card expands it → {rows} detail rows (rationale/scope/structure/relations); {cards} expandable cards"
        )
    elif cards:
        rec.partial(
            f"expandable cards render but expansion revealed little (before={before}, after={after}, rows={rows})"
        )
    else:
        rec.gap("no expandable tactic cards to open")
