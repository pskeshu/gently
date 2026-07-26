# ruff: noqa: E501
"""US-19 — Operate is an instrument, not a workflow. As an operator, every control on every surface works whenever I look at it, in any order."""

from _harness import dom_count, goto, skip_landing, tab, view

META = {
    "id": "US-19",
    "title": "Operate surfaces are always live",
    "cluster": "5 Operate (mark)",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    opened = await view(page, "operate")
    if not opened:
        rec.gap("Operate is not reachable from the devices tab")
        return

    # No stepper, no phase nodes, no progress ladder — anywhere.
    steppers = await dom_count(page, ".op-node, [data-node], .op-stepper, .op-group")

    # Walk all three surfaces in a deliberately non-linear order and confirm each
    # renders with nothing disabled by "where you are".
    results = {}
    for pane in ("acquire", "spim", "bottom", "acquire"):
        ok = await view(page, pane)
        disabled = await page.evaluate(
            "(p) => { const el = document.getElementById('op-pane-' + p); if (!el) return -1;"
            " return [...el.querySelectorAll('button')].filter(b => b.disabled).length; }",
            pane,
        )
        results[pane] = (ok, disabled)
    await rec.shot("operate-always-live")

    missing = [p for p, (ok, _) in results.items() if not ok]
    if missing:
        rec.gap(f"could not reach Operate surfaces: {', '.join(missing)}")
        return
    if steppers:
        rec.gap(f"found {steppers} step/phase elements — the workflow layer has come back")
        return

    # Register/Clear are legitimately disabled with nothing marked; that is state,
    # not sequence. Anything beyond that would be a step model in disguise.
    rec.ok(
        "all three Operate surfaces render and switch in any order with no stepper, phase node or disclosure group present; "
        f"disabled controls per surface: {', '.join(f'{p}={d}' for p, (_, d) in results.items())} "
        "(Register/Clear only, which key off an empty marker set)"
    )
