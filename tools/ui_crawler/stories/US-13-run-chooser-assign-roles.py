# ruff: noqa: E501
"""US-13 — Run chooser + assign roles after marking. As an operator, once embryos are marked I get a Run chooser to assign subject/reference roles and pick how to image them."""

from _harness import dom_count, exists, goto, skip_landing, tab, view

META = {
    "id": "US-13",
    "title": "Run chooser + assign roles after marking",
    "cluster": "6 Operate (acquire)",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")
    # Marking normally reveals the chooser (Phase C). Drive the same real render
    # path headlessly via the app's own event bus — no device needed.
    await page.evaluate(
        "() => { if (typeof ClientEventBus !== 'undefined') ClientEventBus.emit('EMBRYOS_UPDATE', {embryos:[{id:'embryo_1'},{id:'embryo_2'},{id:'embryo_3'}]}); }"
    )
    chooser = await exists(page, '.op-group[data-step="c0"]')  # c0 group now the active rail step
    chips = await dom_count(page, ".op-rolechip")  # per-embryo subject/reference chips
    modes = await dom_count(
        page, 'input[name="op-mode"]'
    )  # run modes (manual/adaptive/library/plan/agent)
    start = await exists(page, "#op-run-start")
    await rec.shot("run-chooser")
    if chooser and chips >= 2 and modes >= 2 and start:
        rec.ok(
            f"Run chooser appears after marking: {chips} role chips (toggle subject/reference) + {modes} run modes + Start run"
        )
    elif chooser:
        rec.partial(f"chooser shown but incomplete (chips={chips}, modes={modes}, start={start})")
    else:
        rec.gap("Run chooser (c0) does not appear after embryos are marked")
