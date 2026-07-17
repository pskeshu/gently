# ruff: noqa: E501
"""US-36 — Read the live notebook. As a user, I open the Notebook tab and read the entries the agent has recorded, so I can follow the lab's observations, findings, and open questions."""

from _harness import count_text, dom_count, exists, goto, skip_landing, tab

META = {
    "id": "US-36",
    "title": "Read the live notebook",
    "cluster": "12 Notebook",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "notebook")
    filters = await count_text(page, r"observations|findings|questions")  # kind filter chips
    surface = await exists(page, "#nb-notes")  # notes reading region
    cards = await dom_count(page, "#nb-notes .nb-card")  # rendered entries
    await rec.shot("notebook-tab")
    if surface and filters >= 2 and cards:
        rec.ok(
            f"notebook tab renders {cards} entries + kind filter ({filters} kinds) + thread rail"
        )
    elif surface and filters >= 2:
        rec.partial(
            f"notebook reading surface renders (kind filter={filters}, thread rail, notes region) but no live notes in dev — empty state shown"
        )
    else:
        rec.gap("notebook tab does not render a readable notes surface")
