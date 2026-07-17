# ruff: noqa: E501
"""US-08 — Take a quick look → scope. As a user who just wants to peek at what's on the scope, I want to jump straight to the devices surface with no plan or commitment."""

from _harness import exists, goto, present

META = {
    "id": "US-08",
    "title": "Take a quick look jumps straight to the scope",
    "cluster": "4 Standalone",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)  # land on #v2-landing (do NOT skip)
    card = await exists(page, '[data-landing="standalone"]')  # the "Take a quick look" choice
    try:
        await page.click('[data-landing="standalone"]', timeout=6000)
    except Exception:
        pass
    await page.wait_for_timeout(700)
    await rec.shot("scope-open")
    tab_active = await exists(page, '.v2-nav-item[data-tab="devices"].active')
    content_active = await present(page, "#devices-content.active")
    if tab_active or content_active:
        rec.ok(
            "'Take a quick look' dismisses the landing and activates the Devices (scope) tab — no plan required"
        )
    elif card:
        rec.gap("standalone card present but clicking it did not activate the Devices tab")
    else:
        rec.gap("no 'Take a quick look' / standalone card on the landing")
