# ruff: noqa: E501
"""US-18 — Run a saved tactic. As an operator, I pick a tactic from the library and run it on the marked embryos."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-18",
    "title": "Run a saved tactic",
    "cluster": "6 Operate (acquire)",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")
    on_pane = await view(page, "acquire")
    await page.click('[data-mode="library"]', timeout=4000)
    await rec.shot("operate-saved-tactic")

    lib = await exists(page, "#op-lib-list")  # populated from /api/tactic_library
    start = await dom_count(page, "#op-run-start")
    panel_shown = await page.evaluate(
        "() => { const p = document.getElementById('op-panel-library'); return !!p && !p.hidden; }"
    )
    items = await dom_count(page, ".op-libitem")

    if not (on_pane and lib and start and panel_shown):
        rec.gap("saved-tactic mode does not expose a library list plus Start")
        return
    rec.blocked(
        f"needs device+library: the tactic library panel and Start are reachable directly, with no survey required first (items={items}); "
        "Start POSTs /api/operate/run-tactic, which needs a live device layer"
    )
