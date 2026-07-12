"""Smoke test for the device-layer state stream.

Opens an SSE connection to /api/devices/stream and prints each event as it
arrives. Useful for verifying the poller is alive without spinning up the
agent or the browser UI.

Usage:
    python scripts/watch_devices.py
    python scripts/watch_devices.py --host 127.0.0.1 --port 60610
    python scripts/watch_devices.py --once         # one-shot via /api/devices/state
    python scripts/watch_devices.py --raw          # don't pretty-print, dump JSON lines
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import aiohttp


def _fmt_positions(positions: dict[str, Any]) -> str:
    if not positions:
        return "(no positions)"
    parts = []
    for name, entry in positions.items():
        kind = entry.get("kind", "?")
        if kind == "xy_stage":
            parts.append(f"{name}: X={entry.get('X', '?'):.2f} Y={entry.get('Y', '?'):.2f}")
        elif kind == "piezo":
            parts.append(f"{name}: Z={entry.get('Position', '?'):.3f}")
        elif kind == "galvo":
            parts.append(f"{name}: A={entry.get('A', '?'):.4f} B={entry.get('B', '?'):.4f}")
        else:
            parts.append(f"{name}: {entry}")
    return " | ".join(parts)


def _print_event(idx: int, payload: dict[str, Any], raw: bool) -> None:
    if raw:
        print(json.dumps(payload), flush=True)
        return

    t = payload.get("t", 0.0)
    flags = []
    if payload.get("heartbeat"):
        flags.append("HEARTBEAT")
    if payload.get("paused"):
        flags.append("PAUSED")
    flag_str = f" [{', '.join(flags)}]" if flags else ""

    positions = payload.get("positions") or {}
    props = payload.get("properties") or {}
    n_devices = len(props)
    n_props = sum(len(v) - (1 if "__type__" in v else 0) for v in props.values())

    print(
        f"[{idx:>5}] t={t:.2f}{flag_str}  {_fmt_positions(positions)}"
        f"  ({n_devices} devs, {n_props} props)",
        flush=True,
    )


async def watch_stream(host: str, port: int, raw: bool) -> None:
    url = f"http://{host}:{port}/api/devices/stream"
    print(f"Connecting to {url}…", file=sys.stderr)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status}: {await resp.text()}", file=sys.stderr)
                sys.exit(1)
            print("Connected. Press Ctrl+C to stop.\n", file=sys.stderr)

            buffer = b""
            idx = 0
            t0 = time.monotonic()
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    data_lines: list[str] = []
                    is_snapshot = False
                    for raw_line in frame.splitlines():
                        line = raw_line.decode("utf-8", errors="replace")
                        if line.startswith(":"):
                            # comment / keepalive
                            continue
                        if line.startswith("event:"):
                            is_snapshot = line.split(":", 1)[1].strip() == "snapshot"
                            continue
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                    if not data_lines:
                        continue
                    try:
                        payload = json.loads("".join(data_lines))
                    except json.JSONDecodeError as e:
                        print(f"  ! parse error: {e}", file=sys.stderr)
                        continue
                    idx += 1
                    if is_snapshot and not raw:
                        print("-- snapshot --", flush=True)
                    _print_event(idx, payload, raw)

            elapsed = time.monotonic() - t0
            print(
                f"\nStream ended after {elapsed:.1f}s, {idx} events received.",
                file=sys.stderr,
            )


async def fetch_once(host: str, port: int, raw: bool, refresh: bool) -> None:
    url = f"http://{host}:{port}/api/devices/state"
    if refresh:
        url += "?refresh=1"
    print(f"GET {url}", file=sys.stderr)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status}: {await resp.text()}", file=sys.stderr)
                sys.exit(1)
            payload = await resp.json()
    _print_event(1, payload, raw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=60610)
    p.add_argument(
        "--once",
        action="store_true",
        help="Fetch one-shot via /api/devices/state and exit",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="With --once, force a fresh read (?refresh=1)",
    )
    p.add_argument("--raw", action="store_true", help="Print raw JSON (one object per line)")
    args = p.parse_args()

    try:
        if args.once:
            asyncio.run(fetch_once(args.host, args.port, args.raw, args.refresh))
        else:
            asyncio.run(watch_stream(args.host, args.port, args.raw))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)


if __name__ == "__main__":
    main()
