# ruff: noqa: E501
"""US-40 — Steer / interrupt the agent. As an operator, I want to stop the current turn or queue messages while it works, so I can redirect it mid-task."""

from _harness import dom_count, exists, goto, skip_landing

META = {
    "id": "US-40",
    "title": "Steer / interrupt the agent",
    "cluster": "13 Agent chat & ask",
    "mode": "agent",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    if await exists(page, "#agent-chat-toggle"):
        await page.click("#agent-chat-toggle")
        await page.wait_for_timeout(400)
    stop = await dom_count(
        page, ".ac-stop"
    )  # explicit Stop button (hidden until a cancellable user turn)
    queue = await dom_count(
        page, ".ac-queue"
    )  # queued-message panel (hidden until type-while-busy)
    await rec.shot("chat-controls")
    if stop and queue:
        rec.partial(
            f"Stop button + queued-message panel exist in the chat composer (stop={stop}, queue={queue}); they surface during a live turn (Stop on cancellable turns, queue on type-while-busy)"
        )
    elif stop or queue:
        rec.partial(
            f"only one steer control is present (stop={stop}, queue={queue}) — the other is missing"
        )
    else:
        rec.gap(
            "no Stop or queue controls in the chat panel — a running agent turn cannot be interrupted/steered from the UI"
        )
