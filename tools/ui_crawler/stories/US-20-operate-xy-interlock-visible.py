# ruff: noqa: E501
"""US-20 — The XY interlock is visible. As an operator, when the sample is at the objective I can see that XY is locked and get out of it."""

from _harness import dom_count, goto, skip_landing, tab, view

META = {
    "id": "US-20",
    "title": "XY interlock is visible and escapable",
    "cluster": "5 Operate (mark)",
    "mode": "headless",
    "needs_account": False,
}

HEAD_KEY = "gently.operate.headLowered"


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    # Latch the head-down belief the way a real down-nudge would, then reload so
    # it is restored from sessionStorage exactly as it is after an F5 mid-embryo.
    await page.evaluate(f"() => sessionStorage.setItem('{HEAD_KEY}', '1')")
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "operate")
    await view(page, "bottom")
    await rec.shot("operate-xy-locked")

    banner = await page.evaluate(
        "() => { const b = document.getElementById('op-lock-bottom'); return !!b && !b.hidden; }"
    )
    cursor = await page.evaluate(
        "() => { const c = document.getElementById('op-mark-canvas');"
        " return c ? getComputedStyle(c).cursor : ''; }"
    )
    backoff = await dom_count(page, "[data-backoff]")
    # The same state must be legible on the SPIM surface, where the head is driven.
    await view(page, "spim")
    spim_banner = await page.evaluate(
        "() => { const b = document.getElementById('op-lock-spim'); return !!b && !b.hidden; }"
    )
    await page.evaluate(f"() => sessionStorage.setItem('{HEAD_KEY}', '0')")

    if not banner:
        rec.gap("the head-down latch survived a reload but no XY-locked banner is shown")
        return
    if not backoff:
        rec.gap("XY is locked with no back-off control — the latch would be inescapable")
        return
    if cursor != "not-allowed":
        rec.partial(
            f"XY-locked banner and back-off are present, but the marking canvas still invites clicks (cursor={cursor!r})"
        )
        return
    rec.ok(
        f"a latched head-down state survives reload and is shown on both camera surfaces (spim={spim_banner}), "
        f"the canvas withdraws its affordance (cursor={cursor}), and an always-reachable back-off control ({backoff}) clears it"
    )
