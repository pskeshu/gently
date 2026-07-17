"""Live position dashboard for the device-layer state stream.

Connects to /api/devices/stream and overwrites a single terminal line with the
current XY / piezo / galvo positions. Also shows the effective update rate and
the wall-clock latency between the payload timestamp and now — so you can feel
the responsiveness while jogging the joystick.

Usage:
    python scripts/watch_position.py
    python scripts/watch_position.py --host 127.0.0.1 --port 60610
    python scripts/watch_position.py --log    # one line per update (no in-place overwrite)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from typing import Any

import aiohttp


def _split_positions(
    positions: dict[str, Any],
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(x, y, z, galvo_a, galvo_b) from a positions payload, any missing as None."""
    x = y = z = a = b = None
    for entry in (positions or {}).values():
        kind = entry.get("kind")
        if kind == "xy_stage":
            x, y = entry.get("X"), entry.get("Y")
        elif kind == "piezo":
            z = entry.get("Position")
        elif kind == "galvo":
            a, b = entry.get("A"), entry.get("B")
    return x, y, z, a, b


def _fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return f"{'—':>10}"
    return f"{v:>10.{digits}f}"


async def watch(host: str, port: int, log_mode: bool) -> None:
    url = f"http://{host}:{port}/api/devices/stream"
    print(f"Connecting to {url}…", file=sys.stderr)

    # Track last 20 arrivals for FPS calculation.
    arrivals: deque[float] = deque(maxlen=20)
    last_print = 0.0
    count = 0

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status}: {await resp.text()}", file=sys.stderr)
                sys.exit(1)
            print("Connected. Move the joystick. Press Ctrl+C to stop.\n", file=sys.stderr)

            buf = b""
            async for chunk in resp.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    data_lines: list[str] = []
                    for raw_line in frame.splitlines():
                        s = raw_line.decode("utf-8", errors="replace")
                        if not s or s.startswith(":") or s.startswith("event:"):
                            continue
                        if s.startswith("data:"):
                            data_lines.append(s[5:].lstrip())
                    if not data_lines:
                        continue
                    try:
                        payload = json.loads("".join(data_lines))
                    except json.JSONDecodeError:
                        continue

                    now = time.time()
                    arrivals.append(now)
                    count += 1

                    payload_t = payload.get("t", now)
                    latency_ms = (now - payload_t) * 1000.0

                    fps = 0.0
                    if len(arrivals) >= 2:
                        span = arrivals[-1] - arrivals[0]
                        if span > 0:
                            fps = (len(arrivals) - 1) / span

                    x, y, z, a, b = _split_positions(payload.get("positions"))

                    paused = payload.get("paused") or payload.get("heartbeat")
                    status = "PAUSED" if paused else "  LIVE"

                    line = (
                        f"[{status}] {fps:5.1f} Hz  lat {latency_ms:5.0f} ms   "
                        f"X={_fmt(x, 2)}  Y={_fmt(y, 2)}   "
                        f"Z={_fmt(z, 3)}   A={_fmt(a, 4)}  B={_fmt(b, 4)}   "
                        f"n={count}"
                    )

                    if log_mode:
                        print(line, flush=True)
                    else:
                        # Rate-limit overwrites to ~30 Hz so the terminal can keep up.
                        if now - last_print >= 1.0 / 30:
                            sys.stdout.write("\r" + line + " " * 4)
                            sys.stdout.flush()
                            last_print = now


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=60610)
    p.add_argument(
        "--log",
        action="store_true",
        help="Print one line per update instead of overwriting",
    )
    args = p.parse_args()

    try:
        asyncio.run(watch(args.host, args.port, args.log))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
