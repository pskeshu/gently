# ruff: noqa: E501
"""US-04 — Design a plan with the agent. As a user, I pick 'Plan an experiment' so the agent guides me through designing a run."""

from _harness import exists, goto, present

META = {
    "id": "US-04",
    "title": "Design a plan with the agent",
    "cluster": "2 Planning (guided)",
    "mode": "agent",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)  # start on the landing (do NOT skip)
    if not await exists(page, '[data-landing="plan"]'):
        rec.gap("no 'Plan an experiment' entry card on the landing")
        return
    await page.click(
        '[data-landing="plan"]'
    )  # → startPlan(): setScreen('plan') + kicks off the /plan agent turn
    await page.wait_for_timeout(1200)
    on_plan = await page.evaluate(
        "() => (document.getElementById('v2-landing')||{}).dataset && document.getElementById('v2-landing').dataset.screen === 'plan'"
    )
    ask = await present(
        page, "#v2-plan-ask"
    )  # where the agent's ask_user_choice design cards render
    summary = await exists(
        page, "#v2-plan-summary"
    )  # 'The plan' panel assembles as the agent designs
    thinking = await present(page, "#v2-plan-thinking")  # informational: agent-working indicator
    await rec.shot("plan-wizard")
    if on_plan and ask and summary:
        rec.blocked(
            f"needs live agent: plan wizard scaffold renders (ask-card mount + 'The plan' panel, thinking={thinking}) but the run is authored by a live /plan agent turn"
        )
    elif on_plan:
        rec.partial(
            f"plan screen opened but wizard scaffold incomplete (ask={ask}, summary={summary}, thinking={thinking})"
        )
    else:
        rec.gap("clicking 'Plan an experiment' did not switch to the in-place plan wizard screen")
