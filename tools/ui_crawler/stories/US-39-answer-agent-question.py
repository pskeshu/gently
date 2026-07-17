# ruff: noqa: E501
"""US-39 — Answer an agent question (ask flow). As an operator, I want to answer the agent's pending question, so it can proceed with my choice."""

from _harness import dom_count, exists, goto, skip_landing

META = {
    "id": "US-39",
    "title": "Answer an agent question (ask flow)",
    "cluster": "13 Agent chat & ask",
    "mode": "agent",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    if await exists(
        page, "#agent-chat-toggle"
    ):  # asks are dual-rendered on the stage AND in the chat log
        await page.click("#agent-chat-toggle")
        await page.wait_for_timeout(300)
    stage = await dom_count(page, "#ask-stage")  # ux_v2 ask surface, 'hidden' until an ask arrives
    live_ask = await exists(
        page, "#ask-stage .ac-choice"
    )  # a rendered pending-ask card with choice buttons
    await rec.shot("ask-surface")
    if not stage:
        rec.gap("no #ask-stage surface — the agent cannot pose a stage-level question (ux_v2 off?)")
    elif live_ask:
        rec.ok("a pending ask is rendered on the stage with answerable choice controls")
    else:
        rec.blocked(
            "needs live agent: no pending ask — #ask-stage stays hidden until the agent asks a question"
        )
