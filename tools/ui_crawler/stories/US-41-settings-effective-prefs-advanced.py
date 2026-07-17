# ruff: noqa: E501
"""US-41 — Settings: effective config / prefs / advanced. As an operator I want to
inspect the server's live config, manage my dashboard prefs, and edit advanced
tunables from Settings, without hand-editing YAML."""

from _harness import exists, goto

META = {
    "id": "US-41",
    "title": "Settings: effective config / prefs / advanced",
    "cluster": "14 Config, session & mesh",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url, "/settings")
    effective = await exists(page, "#section-effective") and await exists(page, "#effective-config")
    prefs = (
        await exists(page, "#pref-export")
        and await exists(page, "#pref-import")
        and await exists(page, "#pref-save-defaults")
    )
    advanced = await exists(page, "#section-advanced") and await exists(page, "#adv-save")
    await rec.shot("settings-config")
    if effective and prefs and advanced:
        rec.ok(
            "Settings exposes Effective config (read-only server config), dashboard prefs (export/import/save-as-rig-defaults/reset), and Advanced (restart-required) tunables"
        )
    elif effective or prefs or advanced:
        rec.partial(
            f"only some config surfaces present (effective={effective}, prefs={prefs}, advanced={advanced})"
        )
    else:
        rec.gap("no effective-config / dashboard-prefs / advanced surfaces on /settings")
