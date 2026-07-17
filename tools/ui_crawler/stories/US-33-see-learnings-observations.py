# ruff: noqa: E501
"""US-33 — See learnings / observations. As a user, I want to read what the agent
has learned and observed, so I can follow the inquiry without reading raw logs."""

from _harness import click_text, count_text, exists, goto, skip_landing, tab

META = {
    "id": "US-33",
    "title": "See learnings / observations",
    "cluster": "11 Memory & campaigns",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    opened = await tab(page, "notebook")  # learnings surface = the shared Notebook
    obs = await count_text(page, r"\bobservations\b")  # kind filter buttons
    find = await count_text(page, r"\bfindings\b")
    notes = await exists(page, "#nb-notes")  # the notes reading pane
    await click_text(page, r"\bfindings\b")  # apply the Findings filter
    await rec.shot("notebook-learnings")
    if opened and obs and find and notes:
        rec.partial(
            "Notebook tab surfaces learnings: Observation/Finding/Question filters + notes pane present, but the pane fills only as the agent records notes (no live data headless)"
        )
    elif opened and notes:
        rec.partial(
            f"notebook notes pane present but kind filters incomplete (observations={obs}, findings={find})"
        )
    else:
        rec.gap(
            "no notebook/learnings surface reachable — observations & findings have nowhere to be read"
        )
