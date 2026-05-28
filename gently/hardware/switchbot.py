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

logger = logging.getLogger(__name__)

# SwitchBot Bot GATT. Note the UUID group is 9fb8 — the widely-copied 9fb9 is wrong.
_CTRL_CHAR = "cba20002-224d-11e6-9fb8-0002a5d5c51b"    # write / write-without-response
_NOTIFY_CHAR = "cba20003-224d-11e6-9fb8-0002a5d5c51b"  # notify (command response)

_COMMANDS = {
    "on":    bytes([0x57, 0x01, 0x01]),
    "off":   bytes([0x57, 0x01, 0x02]),
    "press": bytes([0x57, 0x01, 0x00]),
}
# First byte of the success response. Older Bot firmware returns 0x01 alone;
# modern firmware (≥ Bot v4.x) returns 0x05 followed by a status frame —
# byte 1 = battery %, byte 2 = flag bits. Both mean "press landed."
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
        self.parent = None              # required for Bluesky bps.mv()
        self._state = "unknown"         # last commanded on/off state
        self._lock = threading.Lock()   # serialize BLE access (one radio, one bot)

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
                    data = asyncio.run(
                        _send_command(self.address, _COMMANDS[state], self.timeout)
                    )
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

    # -- Bluesky readable protocol -------------------------------------------
    def read(self):
        return OrderedDict({
            self.name: {"value": self._state, "timestamp": time.time()}
        })

    def describe(self):
        return OrderedDict({
            self.name: {"source": f"switchbot:{self.address}", "dtype": "string", "shape": []}
        })

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
