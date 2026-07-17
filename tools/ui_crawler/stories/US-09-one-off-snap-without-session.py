# ruff: noqa: E501
"""US-09 — One-off snap/volume without a session. As a user at the scope, I want to acquire a single volume without starting a plan or session."""

from _harness import exists, goto, skip_landing, tab, view

META = {
    "id": "US-09",
    "title": "One-off snap/volume without a session",
    "cluster": "4 Standalone",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    await view(page, "manual")  # Devices → Manual control
    snap = await exists(page, "#devices-ls-snap-volume")  # Acquire group: Snap Volume
    burst = await exists(page, "#devices-ls-burst")
    if not snap:
        await rec.shot("no-acquire-control")
        rec.gap("no one-off Snap Volume control in Devices → Manual")
        return
    try:
        await page.click("#devices-ls-snap-volume", timeout=4000)  # exercise the acquire path
    except Exception:
        pass
    await page.wait_for_timeout(900)
    toast = await exists(page, ".gently-toast")  # any success/error feedback?
    await rec.shot("manual-acquire-attempt")
    rec.blocked(
        f"needs device: Snap Volume{'/Burst' if burst else ''} present but POST /api/devices/acquire/volume 502s with the device layer offline; the failure is SILENT — only console.error, no toast/UI feedback (toast_visible={toast})"
    )
