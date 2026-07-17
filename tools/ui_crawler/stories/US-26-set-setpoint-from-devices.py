# ruff: noqa: E501
"""US-26 — Set a setpoint from Devices. As an operator at the rig, I want to type a
target temperature in the Devices header and hit Set, so the controller ramps to it."""

from _harness import dom_count, exists, goto, skip_landing, tab

META = {
    "id": "US-26",
    "title": "Set a temperature setpoint from Devices",
    "cluster": "9 Temperature",
    "mode": "rig",
    "needs_account": False,
}


async def flow(page, url, rec):
    await goto(page, url)
    await skip_landing(page)
    await tab(page, "devices")
    tinput = await dom_count(page, "#devices-temp-input")  # header setpoint field
    tset = await dom_count(page, "#devices-temp-set")  # "Set" button
    revealed = await exists(page, "#devices-temp")  # pill unhides only when a controller is online
    await rec.shot("devices-temp-control")
    if tinput and tset:
        rec.blocked(
            f"needs device: setpoint input + Set button wired in the Devices header (input={tinput}, set={tset}), but #devices-temp stays hidden until the ACUITYnano controller is online (revealed={revealed}) and /api/devices/temperature/set 502s with the device layer offline"
        )
    else:
        rec.gap("no temperature setpoint control in the Devices header")
