# ruff: noqa: E501
"""US-01 — First-run landing. As a new user, I land and see the entry choices,
so I understand how to start."""

from _harness import count_text, exists, goto

META = {
    "id": "US-01",
    "title": "First-run landing shows the entry choices",
    "cluster": "1 Onboarding",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    landing = await exists(page, "#v2-landing")
    plan = await count_text(page, r"plan an experiment")
    look = await count_text(page, r"take a quick look")
    if landing and plan and look:
        rec.ok(f"landing visible with entry choices (plan={plan}, quick-look={look})")
    elif landing:
        rec.partial(f"landing visible but a choice is missing (plan={plan}, quick-look={look})")
    else:
        rec.gap("landing overlay not shown on load")
