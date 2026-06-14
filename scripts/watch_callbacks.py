"""Watch raw MMCore push callbacks from the device layer.

Connects to /api/devices/callbacks/stream and prints every event the MMCore
adapter emits (property changes, stage moves, exposure changes, config-group
changes, etc.) — including events triggered outside the host process (e.g.
joystick moves) if and only if the adapter chooses to fire callbacks for
them.

Use this to figure out what we can move off the polling path. Run the device
layer in one terminal, this script in another, then drive the hardware
(joystick, MM GUI, agent plans, manual setProperty calls) and watch what
fires.

Usage:
    python scripts/watch_callbacks.py
    python scripts/watch_callbacks.py --host 127.0.0.1 --port 60610
    python scripts/watch_callbacks.py --raw
    python scripts/watch_callbacks.py --grep property_changed
    python scripts/watch_callbacks.py --tally   # summarize counts at end
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from typing import Any

import aiohttp


def _fmt_event(idx: int, payload: dict[str, Any]) -> str:
    kind = payload.get("kind", "?")
    t = payload.get("t", 0.0)
    extras = {k: v for k, v in payload.items() if k not in ("kind", "t")}
    extras_str = " ".join(f"{k}={v!r}" for k, v in extras.items())
    return f"[{idx:>5}] t={t:.3f}  {kind:<32} {extras_str}"


async def watch(host: str, port: int, raw: bool, grep: str | None, tally: bool) -> None:
    url = f"http://{host}:{port}/api/devices/callbacks/stream"
    pattern = re.compile(grep, re.IGNORECASE) if grep else None
    tally_counts: Counter[str] = Counter()
    idx = 0
    t0 = time.monotonic()

    print(f"Connecting to {url}…", file=sys.stderr)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status}: {await resp.text()}", file=sys.stderr)
                sys.exit(1)
            print("Connected. Press Ctrl+C to stop.\n", file=sys.stderr)

            buf = b""
            async for chunk in resp.content.iter_any():
                buf += chunk
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    data_lines = []
                    for line in frame.splitlines():
                        s = line.decode("utf-8", errors="replace")
                        if not s or s.startswith(":") or s.startswith("event:"):
                            continue
                        if s.startswith("data:"):
                            data_lines.append(s[5:].lstrip())
                    if not data_lines:
                        continue
                    try:
                        payload = json.loads("".join(data_lines))
                    except json.JSONDecodeError as e:
                        print(f"  ! parse error: {e}", file=sys.stderr)
                        continue
                    idx += 1
                    if tally:
                        tally_counts[payload.get("kind", "?")] += 1
                    if pattern and not pattern.search(payload.get("kind", "")):
                        continue
                    if raw:
                        print(json.dumps(payload), flush=True)
                    else:
                        print(_fmt_event(idx, payload), flush=True)

    if tally:
        elapsed = time.monotonic() - t0
        print(f"\n-- tally over {elapsed:.1f}s --", file=sys.stderr)
        for kind, count in tally_counts.most_common():
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  {kind:<36} {count:>6}  ({rate:.2f}/s)", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=60610)
    p.add_argument("--raw", action="store_true", help="Print full JSON per line")
    p.add_argument("--grep", help="Only print events whose `kind` matches this regex")
    p.add_argument(
        "--tally",
        action="store_true",
        help="At exit, print a count of each event kind seen",
    )
    args = p.parse_args()

    try:
        asyncio.run(watch(args.host, args.port, args.raw, args.grep, args.tally))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
