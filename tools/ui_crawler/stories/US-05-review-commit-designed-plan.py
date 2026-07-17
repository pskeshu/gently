# ruff: noqa: E501
"""US-05 — Review & commit a designed plan. As a user, I review the assembled plan and commit it (continue into the workspace / export)."""

from _harness import dom_count, exists, goto

META = {
    "id": "US-05",
    "title": "Review & commit a designed plan",
    "cluster": "2 Planning (guided)",
    "mode": "agent",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)  # landing → open the plan wizard where the plan + commit controls live
    if not await exists(page, '[data-landing="plan"]'):
        rec.gap("no 'Plan an experiment' entry card to reach the plan/commit surface")
        return
    await page.click('[data-landing="plan"]')
    await page.wait_for_timeout(1200)
    side = await exists(page, "#v2-plan-side")  # 'The plan' review side-panel
    empty = await exists(page, ".v2-plan-side-empty")  # empty until a live plan is authored
    cont = await exists(page, "#v2-plan-continue")  # commit → continue in workspace
    export = await dom_count(
        page, "#v2-plan-export"
    )  # export the finished plan (wired but hidden until plan-ready)
    await rec.shot("plan-review-panel")
    if side and cont and export:
        rec.blocked(
            f"needs live agent: plan review panel + commit controls present (continue={cont}, export-wired={export > 0}) but the panel is {'empty' if empty else 'unpopulated'} — no plan to review/commit until a /plan turn authors one"
        )
    else:
        rec.gap(
            f"plan review/commit surface incomplete (side={side}, continue={cont}, export={export})"
        )
