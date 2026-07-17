# ruff: noqa: E501
"""US-37 — Ask the notebook a question. As a user, I type a question into the notebook's ask box and get a grounded answer, so I can query the lab's memory in plain language."""

from _harness import exists, goto, skip_landing, tab

META = {
    "id": "US-37",
    "title": "Ask the notebook a question",
    "cluster": "12 Notebook",
    "mode": "agent",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "notebook")
    box = await exists(page, "#nb-ask-input")  # question input
    go = await exists(page, "#nb-ask-go")  # Ask button
    if not (box and go):
        await rec.shot("notebook-no-ask")
        rec.gap(f"no ask affordance in the notebook (input={box}, ask-button={go})")
        return
    await page.fill("#nb-ask-input", "what should I try next on pioneer guidance?")
    await page.click("#nb-ask-go")  # POST /api/notebook/ask → Claude
    await page.wait_for_timeout(900)
    await rec.shot("ask-submitted")
    rec.partial(
        "ask box + Ask button present and accept a question (result panel shows a thinking/answer state); a grounded answer needs a live agent turn (POST /api/notebook/ask → Claude)"
    )
