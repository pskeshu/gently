# ruff: noqa: E501
"""US-38 — Chat / delegate to the agent. As an operator, I want to open the agent chat and hand it a task, so the agent can act on my behalf."""

from _harness import exists, goto, skip_landing

META = {
    "id": "US-38",
    "title": "Chat / delegate to the agent",
    "cluster": "13 Agent chat & ask",
    "mode": "agent",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    if not await exists(page, "#agent-chat-toggle"):
        await rec.shot("no-toggle")
        rec.gap("no agent-chat toggle in the header — the chat panel is unreachable")
        return
    await page.click("#agent-chat-toggle")  # header 'Agent' toggle → slides the docked panel in
    await page.wait_for_timeout(400)
    opened = await exists(page, "#agent-chat.open")
    text = await exists(page, "#agent-chat-text")
    send = await exists(page, "#agent-chat-send")
    await rec.shot("chat-open")
    if opened and text and send:
        rec.partial(
            "toggle opens the docked chat panel with composer (textarea + Send); delegating a task + getting a reply needs a live agent turn"
        )
    elif opened:
        rec.partial(f"chat panel opens but composer incomplete (text={text}, send={send})")
    else:
        rec.gap("toggle present but clicking it does not open the panel (#agent-chat.open)")
