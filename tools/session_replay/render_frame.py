"""Render a session-replay frame to PNG at a chosen moment.

The agent-postmortem bridge (spec: docs/superpowers/specs/2026-07-13-session-replay-design.md):
an agent reads the semantic action log as text, then renders only the moments
that matter as pixels. Replays the recorded rrweb stream headlessly (Playwright
chromium + the repo's vendored rrweb bundle) and screenshots one instant.

Usage:
    python tools/session_replay/render_frame.py --session 81865db3 --t 0:52 \
        --out /tmp/frame.png
    python tools/session_replay/render_frame.py --session 81865db3 \
        --t 2026-07-13T07:05:38Z --tab 04dce1bd --url http://localhost:8080 \
        --out /tmp/frame.png

--t accepts an offset (ms, "52s", "mm:ss") or an absolute ISO timestamp
(matched against rrweb event timestamps, e.g. straight from actions.jsonl).
--url (a running gently server) makes same-origin asset URLs in the recorded
DOM resolve; without it the frame renders from the inlined CSS alone.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RRWEB_JS = REPO_ROOT / "gently" / "ui" / "web" / "static" / "vendor" / "rrweb" / "rrweb.umd.min.js"


def storage_root(cli: str | None) -> Path:
    if cli:
        return Path(cli)
    env = os.environ.get("GENTLY_STORAGE_PATH")
    if env:
        return Path(env)
    # settings.py default: Path("D:/Gently3") — relative on non-Windows, so it
    # lands under the cwd the server was launched from.
    return Path("D:/Gently3")


def find_replay_dir(root: Path, session: str) -> Path:
    if re.match(r"^unassigned-\d{8}$", session):
        d = root / "ui-replay" / session
        if d.is_dir():
            return d
        sys.exit(f"error: no unassigned bucket at {d}")
    index = root / "sessions" / "_index.yaml"
    folder = None
    if index.exists():
        import yaml

        mapping = yaml.safe_load(index.read_text(encoding="utf-8")) or {}
        folder = mapping.get(session)
    if folder is None:
        # allow a raw folder name too
        cand = root / "sessions" / session
        if cand.is_dir():
            folder = session
    if folder is None:
        sys.exit(f"error: session {session!r} not in {index}")
    d = root / "sessions" / folder / "ui-replay"
    if not d.is_dir():
        sys.exit(f"error: session {session} has no ui-replay data ({d})")
    return d


def load_events(replay_dir: Path, tab: str | None) -> tuple[str, list]:
    streams = sorted(replay_dir.glob("rrweb-*.jsonl"))
    if not streams:
        sys.exit(f"error: no rrweb streams in {replay_dir}")
    if tab:
        path = replay_dir / f"rrweb-{tab}.jsonl"
        if not path.exists():
            names = ", ".join(p.stem.replace("rrweb-", "") for p in streams)
            sys.exit(f"error: tab {tab!r} not found (have: {names})")
    else:
        # default: the largest stream (the main working tab)
        path = max(streams, key=lambda p: p.stat().st_size)
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if len(events) < 2:
        sys.exit(f"error: {path.name} has too few events to replay")
    return path.stem.replace("rrweb-", ""), events


def parse_moment(spec: str, first_ts: int, last_ts: int) -> int:
    """Offset in ms from the first event. Accepts ms, '52s', 'mm:ss', ISO."""
    s = spec.strip()
    if re.match(r"^\d+$", s):
        return int(s)
    if re.match(r"^\d+(\.\d+)?s$", s):
        return int(float(s[:-1]) * 1000)
    m = re.match(r"^(\d+):(\d{1,2})$", s)
    if m:
        return (int(m.group(1)) * 60 + int(m.group(2))) * 1000
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        abs_ms = int(dt.timestamp() * 1000)
        off = abs_ms - first_ts
        if off < 0 or abs_ms > last_ts + 60_000:
            print(
                f"warning: {spec} is outside the recording "
                f"({datetime.fromtimestamp(first_ts / 1000)} → "
                f"{datetime.fromtimestamp(last_ts / 1000)} local)",
                file=sys.stderr,
            )
        return max(0, off)
    except ValueError:
        sys.exit(f"error: cannot parse --t {spec!r}")


def render(events: list, offset_ms: int, out: Path, base_url: str | None) -> None:
    from playwright.sync_api import sync_playwright

    rrweb_src = RRWEB_JS.read_text(encoding="utf-8")
    base_tag = f'<base href="{html.escape(base_url, quote=True)}/">' if base_url else ""
    page_html = f"""<!doctype html><html><head><meta charset="utf-8">{base_tag}
<style>body{{margin:0;background:#0a0d13}} iframe{{border:none}}</style></head>
<body><div id="stage"></div></body></html>"""

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.set_content(page_html, wait_until="domcontentloaded")
        page.add_script_tag(content=rrweb_src)
        page.evaluate(
            """([events, offset]) => {
                window.__rep = new rrweb.Replayer(events, {
                    root: document.getElementById('stage'),
                    skipInactive: true,
                    // Disable CSS animations/transitions so static (= final)
                    // styles apply: paused replays otherwise freeze entrance
                    // animations at their from-state (v2-rise: opacity 0),
                    // leaving whole panels invisible.
                    insertStyleRules: [
                        '*, *::before, *::after { animation: none !important; ' +
                        'transition: none !important; }',
                    ],
                });
                window.__rep.pause(offset);
            }""",
            [events, offset_ms],
        )
        # let stylesheets/images inside the replay iframe settle
        page.wait_for_timeout(1500)
        iframe = page.locator("#stage iframe")
        out.parent.mkdir(parents=True, exist_ok=True)
        iframe.screenshot(path=str(out))
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--session", required=True, help="session id, folder name, or unassigned-YYYYMMDD"
    )
    ap.add_argument(
        "--t", required=True, help="moment: ms offset, '52s', 'mm:ss', or ISO timestamp"
    )
    ap.add_argument("--tab", help="tab id (default: largest stream)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument(
        "--storage", help="storage root (default: $GENTLY_STORAGE_PATH or ./D:/Gently3)"
    )
    ap.add_argument(
        "--url", help="running gently server for asset resolution, e.g. http://localhost:8080"
    )
    args = ap.parse_args()

    replay_dir = find_replay_dir(storage_root(args.storage), args.session)
    tab, events = load_events(replay_dir, args.tab)
    first_ts = events[0].get("timestamp", 0)
    last_ts = events[-1].get("timestamp", first_ts)
    offset = parse_moment(args.t, first_ts, last_ts)
    total = last_ts - first_ts
    print(
        f"tab {tab}: {len(events)} events, {total / 1000:.0f}s total — "
        f"rendering at +{offset / 1000:.1f}s"
    )
    render(events, offset, Path(args.out), args.url)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
