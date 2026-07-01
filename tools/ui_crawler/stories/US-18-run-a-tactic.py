# ruff: noqa: E501
"""US-18 — Run a tactic. As an operator, I pick how to image the marked set and press Start run, launching one tactic on the microscope."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {"id": "US-18", "title": "Run a tactic",
        "cluster": "8 Operations & tactics", "mode": "rig", "needs_account": False}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")                          # RUN chooser lives in the Operate surface
    surface = await exists(page, ".op-stepper")           # operate surface rendered
    runstart = await dom_count(page, "#op-run-start")     # "Start run" — in DOM, gated behind the run phase
    modes = await dom_count(page, 'input[name="op-mode"]')  # manual/adaptive/library/plan/agent
    await rec.shot("operate-run-chooser")
    if surface and runstart and modes:
        rec.blocked(f"needs device+agent: the RUN chooser (#op-run-start + {modes} run modes incl. library/plan/agent) is reached only after a live bottom-cam survey marks embryos; Start run POSTs /api/operate/run-tactic which 502s with the device layer offline")
    else:
        rec.gap(f"run-tactic control not found on the Operate surface (surface={surface}, start={runstart}, modes={modes})")
