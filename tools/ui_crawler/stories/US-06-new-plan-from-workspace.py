# ruff: noqa: E501
"""US-06 — Start a new plan from the workspace. As a user already in the workspace
(after 'Take a quick look' or Skip), I want to start a new experiment plan without
restarting."""

from _harness import count_text, goto, skip_landing, tab

META = {
    "id": "US-06",
    "title": "Start a new plan from the workspace",
    "cluster": "3 Planning (access)",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)  # simulate landing dismissed → workspace
    await tab(page, "plans")
    newplan = await count_text(
        page, r"new plan|create plan|\+\s*plan|new campaign|create campaign|start planning"
    )
    if newplan:
        rec.ok(f"workspace exposes a labelled plan-creation control ({newplan})")
    else:
        rec.gap(
            "no labelled 'New plan' control in the workspace/plans tab — planning is only reachable via the header-logo→landing reset or agent chat"
        )
