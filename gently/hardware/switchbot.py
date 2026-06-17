"""
SwitchBot Bot (WoHand) control as a Bluesky/Ophyd-protocol device.

The SwitchBot Bot is a Bluetooth-LE button pusher. In "Switch mode" it supports
explicit on/off; in "Press mode" it does a momentary press. This module talks to
it directly over BLE via ``bleak`` using the documented GATT command protocol —
no SwitchBot cloud, no hub.

The device follows the same duck-typed Bluesky protocol as the diSPIM devices
(see ``gently.hardware.dispim.devices.optical.DiSPIMLED``): ``set(state)`` returns
an ophyd ``Status``, plus ``read()``/``describe()``. So it drops into plans via
``yield from bps.mv(bot, 'on')``.

BLE I/O is async (``bleak``); ``set()`` runs a fresh connect→write→disconnect
cycle in a worker thread and resolves the Status when done. Connecting per command
keeps the implementation robust (no stale-connection handling) at the cost of
~1-2 s latency, which is fine for a low-frequency accessory. For lower latency or
encrypted/password-protected Bots, swap the ``_send_command`` body for PySwitchbot.

Self-test (drives a real Bot)::

    python gently/hardware/switchbot.py EC:6F:04:06:5B:23 on off
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

# SwitchBot Bot GATT. Note the UUID group is 9fb8 — the widely-copied 9fb9 is wrong.
_CTRL_CHAR = "cba20002-224d-11e6-9fb8-0002a5d5c51b"  # write / write-without-response
_NOTIFY_CHAR = "cba20003-224d-11e6-9fb8-0002a5d5c51b"  # notify (command response)

_COMMANDS = {
    "on": bytes([0x57, 0x01, 0x01]),
    "off": bytes([0x57, 0x01, 0x02]),
    "press": bytes([0x57, 0x01, 0x00]),
}
# Dedicated status query: returns battery %, firmware version, mode flags.
# This is the only reliable source of battery — action-command responses
# also include status bytes but in a different format (byte 1 there isn't
# battery despite what's documented for older firmware).
_QUERY_STATUS = bytes([0x57, 0x02])
# Status-query response format (firmware ≥ 6.x):
#   byte 0 = 0x01 success
#   byte 1 = battery %
#   byte 2 = firmware version (BCD-ish: high nibble.low nibble — 0x42 = v4.2)
#   bytes 3+ = mode flags / timer count / counters (firmware-dependent)
_STATUS_BATTERY_IDX = 1
_STATUS_FIRMWARE_IDX = 2
# First byte of the success response. Older Bot firmware returns 0x01 alone;
# modern firmware (≥ Bot v4.x) returns 0x05 for action commands followed by
# action-status bytes. Both are "command landed" — the action payload format
# differs from the status-query payload format, so don't reuse parsers.
_RESP_OK = (0x01, 0x05)


class SwitchBotError(RuntimeError):
    """BLE I/O failed, timed out, or the Bot reported a non-OK response."""


async def _send_command(address: str, command: bytes, timeout: float) -> bytes:
    """Connect, send one command, await the response notification, disconnect.

    Returns the raw response bytes; raises SwitchBotError on timeout or non-OK.
    """
    from bleak import BleakClient  # lazy import keeps module import cheap

    response: dict[str, bytes] = {}
    got = asyncio.Event()

    def _on_notify(_char, data: bytearray) -> None:
        response["data"] = bytes(data)
        got.set()

    async with BleakClient(address, timeout=timeout) as client:
        await client.start_notify(_NOTIFY_CHAR, _on_notify)
        await client.write_gatt_char(_CTRL_CHAR, command, response=True)
        try:
            await asyncio.wait_for(got.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise SwitchBotError("no response notification from SwitchBot") from exc
        finally:
            try:
                await client.stop_notify(_NOTIFY_CHAR)
            except Exception:  # disconnect cleanup is best-effort
                pass

    data = response["data"]
    if not data or data[0] not in _RESP_OK:
        raise SwitchBotError(f"SwitchBot returned non-OK response: {data.hex()}")
    return data


class SwitchBot:
    """Bluesky-protocol device for a SwitchBot Bot button pusher.

    Parameters
    ----------
    address : str
        BLE MAC address, e.g. "EC:6F:04:06:5B:23".
    name : str
        Device name used as the key in plans and read() output.
    timeout : float
        Per-command BLE connect/response timeout in seconds.

    Valid states for ``set``: 'on', 'off', 'press'.
    """

    def __init__(self, address: str, name: str = "switchbot", *, timeout: float = 20.0):
        self.address = address
        self.name = name
        self.timeout = timeout
        self.parent = None  # required for Bluesky bps.mv()
        self._state = "unknown"  # last commanded on/off state
        # Status fields populated only by read_status(). Left as None until
        # first contact — action commands deliberately don't update these,
        # see note on _STATUS_BATTERY_IDX above.
        self._battery_pct: int | None = None
        self._firmware: int | None = None
        self._status_ts: float | None = None
        self._lock = threading.Lock()  # serialize BLE access (one radio, one bot)

    # -- Bluesky settable protocol -------------------------------------------
    def set(self, state: str):
        """Send on/off/press. Returns an ophyd Status that finishes when done."""
        from ophyd.status import Status

        state = str(state).lower()
        if state not in _COMMANDS:
            raise ValueError(f"state {state!r} not in {list(_COMMANDS)}")

        status = Status(obj=self, timeout=self.timeout + 5)

        def worker():
            with self._lock:
                try:
                    data = asyncio.run(_send_command(self.address, _COMMANDS[state], self.timeout))
                except Exception as exc:
                    logger.warning("SwitchBot %s set(%s) failed: %s", self.name, state, exc)
                    status.set_exception(exc)
                    return
            if state in ("on", "off"):
                self._state = state
            logger.info("SwitchBot %s -> %s (resp %s)", self.name, state, data.hex())
            status.set_finished()

        threading.Thread(target=worker, name=f"{self.name}-set", daemon=True).start()
        return status

    # -- Dedicated status query (no actuation) -------------------------------
    def read_status(self) -> dict:
        """Query battery / firmware / mode without touching the switch arm.

        Synchronous: runs its own BLE connect → query → disconnect on the
        caller's thread. Updates the cached status fields on success so
        read() surfaces fresh values to the device-state stream. Use this
        for periodic polls (~hourly is fine; battery doesn't move quickly).

        Returns a dict ``{battery_pct, firmware, raw_hex}``; raises
        SwitchBotError on BLE / protocol failure.
        """
        with self._lock:
            data = asyncio.run(_send_command(self.address, _QUERY_STATUS, self.timeout))
        info: dict[str, Any] = {
            "raw_hex": data.hex(),
            "battery_pct": data[_STATUS_BATTERY_IDX] if len(data) > _STATUS_BATTERY_IDX else None,
            "firmware": data[_STATUS_FIRMWARE_IDX] if len(data) > _STATUS_FIRMWARE_IDX else None,
        }
        if info["battery_pct"] is not None:
            self._battery_pct = info["battery_pct"]
        if info["firmware"] is not None:
            self._firmware = info["firmware"]
        self._status_ts = time.time()
        logger.info("SwitchBot %s status: %s", self.name, info)
        return info

    # -- Bluesky readable protocol -------------------------------------------
    def read(self):
        ts = time.time()
        out = OrderedDict({self.name: {"value": self._state, "timestamp": ts}})
        if self._battery_pct is not None:
            out[f"{self.name}_battery_pct"] = {
                "value": self._battery_pct,
                "timestamp": self._status_ts or ts,
            }
        if self._firmware is not None:
            out[f"{self.name}_firmware"] = {
                "value": self._firmware,
                "timestamp": self._status_ts or ts,
            }
        return out

    def describe(self):
        return OrderedDict(
            {
                self.name: {
                    "source": f"switchbot:{self.address}",
                    "dtype": "string",
                    "shape": [],
                },
                f"{self.name}_battery_pct": {
                    "source": f"switchbot:{self.address}",
                    "dtype": "integer",
                    "shape": [],
                },
                f"{self.name}_firmware": {
                    "source": f"switchbot:{self.address}",
                    "dtype": "integer",
                    "shape": [],
                },
            }
        )

    def read_configuration(self):
        return OrderedDict()

    def describe_configuration(self):
        return OrderedDict()


if __name__ == "__main__":
    # Standalone self-test, e.g.:  python gently/hardware/switchbot.py AA:BB:.. on off
    import sys

    address = "EC:6F:04:06:5B:23"
    cmds = []
    for arg in sys.argv[1:]:
        if ":" in arg and len(arg) >= 17:  # looks like a MAC address
            address = arg
        else:
            cmds.append(arg.lower())
    cmds = cmds or ["on", "off"]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bot = SwitchBot(address)
    print(f"SwitchBot {address} — sequence: {cmds}\n")
    for i, cmd in enumerate(cmds):
        print(f"set({cmd!r}) ...")
        st = bot.set(cmd)
        st.wait(30)  # blocks; raises if the command failed
        print(f"  done; read() -> {bot.read()[bot.name]['value']}")
        if i != len(cmds) - 1:
            time.sleep(1.5)
    print("\nOK")
