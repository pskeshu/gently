# ruff: noqa: E501
"""US-25 — Configure the thermalizer connection (serial/MQTT) from Settings, so I
can point gently at the temperature controller without editing YAML."""

from _harness import dom_count, exists, goto

META = {
    "id": "US-25",
    "title": "Configure the thermalizer (serial/MQTT) in Settings",
    "cluster": "9 Temperature",
    "mode": "headless",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url, "/settings")
    section = await exists(page, "#section-thermalizer")
    backends = await dom_count(
        page, 'input[name="th-backend"]'
    )  # radios are display:none, styled labels
    apply = await exists(page, "#th-apply")
    test = await exists(page, "#th-test")
    if section and backends >= 2 and apply and test:
        rec.ok(
            f"Settings → Hardware → Thermalizer: {backends} backend options + Test + Apply present"
        )
    elif section:
        rec.partial(
            f"thermalizer section present but controls incomplete (backends={backends}, apply={apply}, test={test})"
        )
    else:
        rec.gap("no thermalizer configuration UI in Settings")
