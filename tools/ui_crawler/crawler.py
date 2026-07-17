#!/usr/bin/env python3
# Embedded JS (FINGERPRINT_JS / ENUMERATE_JS) makes line-length impractical here.
# ruff: noqa: E501
"""Gently UI crawler / simulator — the dynamic complement to the static
user-story audit.

It *walks the app like a user*: from a seed state it fingerprints the UI,
enumerates every interactive element, clicks each in an isolated page, diffs the
resulting state, and builds an empirical state-transition graph. Because it
observes real runtime behaviour (not just wiring), it surfaces things static
tracing misses — e.g. "reloading returns you to the landing", dead controls that
do nothing, clicks that throw console errors or 4xx/5xx, and infinite spinners.

Parallel by design: N headless browser contexts probe (state, element) jobs
concurrently, each reaching its target state by replaying the click-path from
root, so probes are isolated and deterministic.

Usage:
    uv run python tools/ui_crawler/crawler.py --url http://localhost:8080 \
        --workers 4 --max-depth 3 --out tools/ui_crawler/out

Requires the dev group (`playwright` in pyproject [dependency-groups].dev) plus a
one-time `uv run playwright install chromium` (or firefox/webkit).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from collections import deque
from pathlib import Path

from playwright.async_api import async_playwright

# --- injected page scripts -------------------------------------------------

# Canonical STRUCTURAL fingerprint of the current UI state. Deliberately ignores
# data (embryo names/counts) so states dedupe on structure, not content.
FINGERPRINT_JS = r"""
() => {
  const vis = el => {
    if (!el) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') === 0) return false;
    const r = el.getBoundingClientRect();   // rect works for position:fixed (offsetParent does not)
    return r.width > 1 && r.height > 1;
  };
  const landing = document.getElementById('v2-landing');
  const landingVisible = !!(landing && !landing.classList.contains('dismissed') && vis(landing));
  const at = document.querySelector('[data-tab].active, .tab.active, .nav-tab.active');
  const activeTab = at ? (at.dataset.tab || (at.textContent || '').trim().slice(0, 24)) : null;
  const activeViews = [...document.querySelectorAll('[data-view].active')].map(e => e.dataset.view).sort();
  const panels = [...document.querySelectorAll('section[id], [id^="tab-"], [id^="panel-"], [id^="section-"]')]
    .filter(vis).map(e => e.id).sort().slice(0, 24);
  const modal = [...document.querySelectorAll('.modal, [role="dialog"]')].filter(vis)
    .map(e => e.id || (e.className || '').toString().slice(0, 30)).sort().slice(0, 5);
  const spinner = [...document.querySelectorAll('.spinner, .loading, [aria-busy="true"], .v2-thinking:not(.hidden)')]
    .some(vis);
  const toast = [...document.querySelectorAll('.gently-toast, .embryo-toast, .toast')].filter(vis)
    .map(e => (e.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 60)).slice(0, 3);
  return { path: location.pathname, landingVisible, activeTab, activeViews, panels, modal, spinner, toast };
}
"""

# Interactive elements + a stable selector for each.
ENUMERATE_JS = r"""
() => {
  const vis = el => {
    if (!el) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') === 0) return false;
    const r = el.getBoundingClientRect();
    return r.width > 1 && r.height > 1;
  };
  const sel = (el) => {
    if (el.id) return '#' + CSS.escape(el.id);
    for (const a of ['data-tab','data-view','data-landing','data-go-tab','data-node','name']) {
      const v = el.getAttribute(a);
      if (v) return `${el.tagName.toLowerCase()}[${a}="${CSS.escape(v)}"]`;
    }
    const parts = []; let node = el, depth = 0;
    while (node && node.nodeType === 1 && depth < 6) {
      if (node.id) { parts.unshift('#' + CSS.escape(node.id)); break; }
      let s = node.tagName.toLowerCase();
      const p = node.parentElement;
      if (p) { const sib = [...p.children].filter(c => c.tagName === node.tagName);
               if (sib.length > 1) s += `:nth-of-type(${sib.indexOf(node) + 1})`; }
      parts.unshift(s); node = p; depth++;
    }
    return parts.join(' > ');
  };
  const origin = location.origin;
  const skip = (el) => {
    if (!vis(el)) return true;
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') return true;
    if (el.tagName === 'A') {
      const href = el.getAttribute('href') || '';
      if (el.target === '_blank') return true;
      if (/^(mailto:|tel:|javascript:)/.test(href)) return true;
      if (/^https?:\/\//.test(href) && !href.startsWith(origin)) return true;
      if (/\/logout/.test(href)) return true;  // never sign ourselves out
    }
    return false;
  };
  const nodes = [...document.querySelectorAll(
    'button, a[href], [role="button"], [data-tab], [data-view], [data-landing], ' +
    '[data-go-tab], [onclick], .btn, input[type="submit"], input[type="button"]')];
  const seen = new Set(); const out = [];
  for (const el of nodes) {
    if (skip(el)) continue;
    const s = sel(el); if (!s || seen.has(s)) continue; seen.add(s);
    const kind = el.dataset.tab ? 'tab' : el.dataset.view ? 'view'
      : el.dataset.landing ? 'landing' : el.tagName.toLowerCase();
    out.push({ selector: s, text: (el.innerText || el.value || '').replace(/\s+/g, ' ').trim().slice(0, 50), kind });
  }
  return out;
}
"""


def fp_hash(fp: dict) -> str:
    key = json.dumps(
        {
            k: fp.get(k)
            for k in ("path", "landingVisible", "activeTab", "activeViews", "panels", "modal")
        },
        sort_keys=True,
    )
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def fp_label(fp: dict) -> str:
    if fp.get("landingVisible"):
        return "landing"
    tab = fp.get("activeTab") or "?"
    views = fp.get("activeViews") or []
    return f"{tab}/{views[0]}" if views else tab


class Crawler:
    def __init__(self, args):
        self.url = args.url.rstrip("/")
        self.browser_name = args.browser
        self.workers = args.workers
        self.max_depth = args.max_depth
        self.max_states = args.max_states
        self.max_elems = args.max_elements
        self.timeout = args.timeout
        self.headed = args.headed
        self.slow_mo = args.slow_mo
        self.trace = args.trace
        self.trace_findings = args.trace_findings
        self.video = args.video
        self.out = Path(args.out)
        self.states: dict[str, dict] = {}  # hash -> {hash, path, fp, elements}
        self.edges: list[dict] = []
        self._sem = asyncio.Semaphore(self.workers)

    async def _new_page(self, context):
        page = await context.new_page()
        page.set_default_timeout(self.timeout)
        return page

    async def _settle(self, page):
        try:
            await page.wait_for_load_state("networkidle", timeout=2500)
        except Exception:
            pass
        await asyncio.sleep(0.35)

    # Synthetic (non-DOM) actions probed from every state, so browser-level
    # transitions — the real way back to the landing is a reload — are explored.
    SYNTHETIC = [
        {"selector": "__reload__", "text": "(reload page)", "kind": "nav"},
        {"selector": "__goto_root__", "text": "(go to /)", "kind": "nav"},
    ]

    async def _do_action(self, page, selector):
        if selector == "__reload__":
            await page.reload(wait_until="domcontentloaded", timeout=self.timeout)
        elif selector == "__goto_root__":
            await page.goto(self.url + "/", wait_until="domcontentloaded", timeout=self.timeout)
        else:
            await page.click(selector, timeout=self.timeout)
        await self._settle(page)

    async def _reach(self, page, path):
        """Navigate to root and replay the action-path. Returns True on success."""
        await page.goto(self.url + "/", wait_until="domcontentloaded", timeout=self.timeout)
        await asyncio.sleep(0.5)  # let landing.js / app.js init
        for step in path:
            try:
                await self._do_action(page, step)
            except Exception:
                return False
        return True

    async def _fingerprint(self, page):
        return await page.evaluate(FINGERPRINT_JS)

    async def _enumerate(self, page):
        return await page.evaluate(ENUMERATE_JS)

    async def _probe(self, context, state, element):
        """Reach `state`, click `element`, capture the resulting state + errors."""
        async with self._sem:
            page = await self._new_page(context)
            console_errors: list[str] = []
            http_errors: list[str] = []
            page.on(
                "console",
                lambda m: console_errors.append(m.text[:160]) if m.type == "error" else None,
            )
            page.on(
                "response",
                lambda r: (
                    http_errors.append(f"{r.status} {r.url.split('?')[0]}")
                    if r.status >= 400
                    else None
                ),
            )
            edge = {
                "from": state["hash"],
                "from_label": fp_label(state["fp"]),
                "via": element["selector"],
                "via_text": element["text"],
                "via_kind": element["kind"],
            }
            try:
                if not await self._reach(page, state["path"]):
                    edge["result"] = "reach_failed"
                    return edge, None
                console_errors.clear()
                http_errors.clear()
                before = await self._fingerprint(page)
                try:
                    await self._do_action(page, element["selector"])
                except Exception as exc:
                    edge["result"] = "click_failed"
                    edge["detail"] = str(exc)[:120]
                    return edge, None
                after = await self._fingerprint(page)
                changed = fp_hash(before) != fp_hash(after)
                new_toast = [
                    t for t in (after.get("toast") or []) if t not in (before.get("toast") or [])
                ]
                edge.update(
                    {
                        "result": "ok",
                        "to": fp_hash(after),
                        "to_label": fp_label(after),
                        "changed": changed,
                        "toast": new_toast,
                        "console_errors": sorted(set(console_errors))[:5],
                        "http_errors": sorted(set(http_errors))[:5],
                        "spinner_after": bool(after.get("spinner")),
                        "returned_to_landing": bool(after.get("landingVisible"))
                        and not before.get("landingVisible"),
                    }
                )
                new_state = None
                if changed and len(state["path"]) + 1 <= self.max_depth:
                    elems = await self._enumerate(page)
                    new_state = {
                        "hash": edge["to"],
                        "path": state["path"] + [element["selector"]],
                        "fp": after,
                        "elements": elems,
                    }
                return edge, new_state
            except Exception as exc:
                edge["result"] = "error"
                edge["detail"] = str(exc)[:140]
                return edge, None
            finally:
                await page.close()

    async def run(self):
        async with async_playwright() as p:
            engine = {"chromium": p.chromium, "firefox": p.firefox, "webkit": p.webkit}[
                self.browser_name
            ]
            # --disable-dev-shm-usage avoids the /dev/shm exhaustion (EPIPE / tab
            # crashes) that headless Chromium hits under concurrent pages.
            launch_args = (
                ["--disable-dev-shm-usage", "--no-sandbox"]
                if self.browser_name == "chromium"
                else []
            )
            browser = await engine.launch(
                headless=not self.headed, slow_mo=self.slow_mo, args=launch_args
            )
            ctx_kwargs = {"viewport": {"width": 1440, "height": 900}}
            if self.video:
                (self.out / "videos").mkdir(parents=True, exist_ok=True)
                ctx_kwargs["record_video_dir"] = str(self.out / "videos")
            context = await browser.new_context(**ctx_kwargs)
            if self.trace:
                await context.tracing.start(screenshots=True, snapshots=True, sources=True)

            # seed state
            page = await self._new_page(context)
            if not await self._reach(page, []):
                raise RuntimeError(f"could not load {self.url}")
            root_fp = await self._fingerprint(page)
            root_elems = await self._enumerate(page)
            await page.close()
            root = {"hash": fp_hash(root_fp), "path": [], "fp": root_fp, "elements": root_elems}
            self.states[root["hash"]] = root

            frontier = deque([root["hash"]])
            while frontier and len(self.states) < self.max_states:
                sh = frontier.popleft()
                state = self.states[sh]
                if len(state["path"]) >= self.max_depth:
                    continue
                elems = state["elements"][: self.max_elems] + self.SYNTHETIC
                print(
                    f"[crawl] state {fp_label(state['fp'])} ({sh}) depth={len(state['path'])} "
                    f"probing {len(elems)} elements  [{len(self.states)} states so far]",
                    flush=True,
                )
                results = await asyncio.gather(*[self._probe(context, state, el) for el in elems])
                for edge, new_state in results:
                    self.edges.append(edge)
                    if (
                        new_state
                        and new_state["hash"] not in self.states
                        and len(self.states) < self.max_states
                    ):
                        self.states[new_state["hash"]] = new_state
                        frontier.append(new_state["hash"])
            if self.trace:
                self.out.mkdir(parents=True, exist_ok=True)
                await context.tracing.stop(path=str(self.out / "trace.zip"))
                print(
                    f"[trace] {self.out}/trace.zip — view with: uv run playwright show-trace {self.out}/trace.zip",
                    flush=True,
                )
            await context.close()  # flush videos
            await browser.close()

    # --- outputs -----------------------------------------------------------

    def analyse(self) -> dict:
        ok = [e for e in self.edges if e.get("result") == "ok"]
        dead = [
            e
            for e in ok
            if not e["changed"]
            and not e["toast"]
            and not e["console_errors"]
            and not e["http_errors"]
        ]
        console_err = [e for e in ok if e["console_errors"]]
        http_err = [e for e in ok if e["http_errors"]]
        spinners = [e for e in ok if e["spinner_after"]]
        to_landing = [e for e in ok if e["returned_to_landing"]]
        # unreachable tabs: data-tab values enumerated but never an activeTab
        seen_tabs, active_tabs = set(), set()
        for s in self.states.values():
            if s["fp"].get("activeTab"):
                active_tabs.add(s["fp"]["activeTab"])
            for el in s["elements"]:
                if el["kind"] == "tab":
                    t = el["selector"].split('"')[1] if '"' in el["selector"] else el["text"]
                    seen_tabs.add(t)
        unreachable = sorted(seen_tabs - active_tabs)
        return {
            "dead": dead,
            "console_err": console_err,
            "http_err": http_err,
            "spinners": spinners,
            "to_landing": to_landing,
            "unreachable_tabs": unreachable,
        }

    def write(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "graph.json").write_text(
            json.dumps({"states": self.states, "edges": self.edges}, indent=2)
        )
        a = self.analyse()

        # Mermaid transition graph (notable edges only, to stay readable)
        lines = ["```mermaid", "flowchart LR"]
        for h, s in self.states.items():
            lines.append(f'  {h}["{fp_label(s["fp"])}<br/><small>{h}</small>"]')
        for e in self.edges:
            if e.get("result") != "ok" or not e.get("changed"):
                continue
            lbl = (e["via_text"] or e["via_kind"] or "click").replace('"', "'")[:22]
            mark = (
                " ⚑" if e["returned_to_landing"] or e["http_errors"] or e["console_errors"] else ""
            )
            lines.append(f"  {e['from']} -->|{lbl}{mark}| {e['to']}")
        lines.append("```")
        (self.out / "graph.mmd").write_text("\n".join(lines))

        # Human report
        def block(title, items, fmt):
            r = [f"## {title} ({len(items)})", ""]
            r += [fmt(x) for x in items] or ["_none_"]
            r.append("")
            return r

        rep = [
            f"# Gently UI crawl — {self.url}",
            "",
            f"States discovered: **{len(self.states)}** · edges probed: **{len(self.edges)}** · "
            f"engine: {self.browser_name}",
            "",
        ]
        rep += block(
            "↩ Transitions that return to the landing",
            a["to_landing"],
            lambda e: (
                f"- from **{e['from_label']}** via `{e['via_text'] or e['via_kind']}` "
                f"(`{e['via']}`) → landing"
            ),
        )
        rep += block(
            "💥 Clicks that triggered console errors",
            a["console_err"],
            lambda e: f"- **{e['from_label']}** · `{e['via_text']}` → {e['console_errors']}",
        )
        rep += block(
            "🌐 Clicks that triggered HTTP 4xx/5xx",
            a["http_err"],
            lambda e: f"- **{e['from_label']}** · `{e['via_text']}` → {e['http_errors']}",
        )
        rep += block(
            "⏳ Clicks that left a spinner running",
            a["spinners"],
            lambda e: f"- **{e['from_label']}** · `{e['via_text']}` → spinner still visible",
        )
        rep += block(
            "🚫 Dead controls (no state change / toast / error / nav)",
            a["dead"],
            lambda e: (
                f"- **{e['from_label']}** · `{e['via_text'] or e['via_kind']}` (`{e['via']}`)"
            ),
        )
        rep += [
            "## Tabs enumerated but never reachable as active",
            "",
            (", ".join(a["unreachable_tabs"]) or "_none_"),
            "",
        ]
        (self.out / "report.md").write_text("\n".join(rep))
        return a

    async def replay_findings(self):
        """Replay each notable finding edge in its OWN Playwright trace, so every
        deficiency can be scrubbed action-by-action in `playwright show-trace`."""

        def kind_of(e):
            if e.get("returned_to_landing"):
                return "return-to-landing"
            if e.get("console_errors"):
                return "console-error"
            if e.get("http_errors"):
                return "http-error"
            if e.get("spinner_after"):
                return "spinner"
            return "dead-control"

        def notable(e):
            return e.get("result") == "ok" and (
                e.get("returned_to_landing")
                or e.get("console_errors")
                or e.get("http_errors")
                or e.get("spinner_after")
            )

        picks = [e for e in self.edges if notable(e)]
        dead = [
            e
            for e in self.edges
            if e.get("result") == "ok"
            and not e.get("changed")
            and not e.get("toast")
            and not e.get("console_errors")
            and not e.get("http_errors")
        ]
        picks += dead[:4]
        seen, uniq = set(), []
        for e in picks:
            k = (e["from"], e["via"])
            if k not in seen:
                seen.add(k)
                uniq.append(e)

        tdir = self.out / "traces"
        tdir.mkdir(parents=True, exist_ok=True)
        manifest = []
        async with async_playwright() as p:
            engine = {"chromium": p.chromium, "firefox": p.firefox, "webkit": p.webkit}[
                self.browser_name
            ]
            args = (
                ["--disable-dev-shm-usage", "--no-sandbox"]
                if self.browser_name == "chromium"
                else []
            )
            browser = await engine.launch(headless=not self.headed, slow_mo=self.slow_mo, args=args)
            context = await browser.new_context(viewport={"width": 1440, "height": 900})
            for i, e in enumerate(uniq):
                state = self.states.get(e["from"])
                if not state:
                    continue
                kind = kind_of(e)
                label = e.get("via_text") or e.get("via_kind") or "click"
                slug = f"{i:02d}-{kind}-" + (
                    re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:28] or "action"
                )
                await context.tracing.start(
                    screenshots=True, snapshots=True, sources=True, title=f"{kind}: {label}"
                )
                page = await self._new_page(context)
                try:
                    if await self._reach(page, state["path"]):
                        try:
                            await self._do_action(page, e["via"])
                        except Exception:
                            pass
                finally:
                    await page.close()
                await context.tracing.stop(path=str(tdir / f"{slug}.zip"))
                manifest.append(
                    {
                        "trace": f"{slug}.zip",
                        "kind": kind,
                        "from": e["from_label"],
                        "action": label,
                        "via": e["via"],
                        "detail": e.get("console_errors") or e.get("http_errors") or "",
                    }
                )
                print(
                    f"  [trace] {kind:17} from {e['from_label']:15} '{label[:28]}' → {slug}.zip",
                    flush=True,
                )
            await context.close()
            await browser.close()
        (tdir / "index.json").write_text(json.dumps(manifest, indent=2))
        print(f"\n[traces] {len(manifest)} finding traces → {tdir}/")
        print(f"  scrub one:  uv run playwright show-trace {tdir}/<name>.zip")
        return manifest


def main():
    ap = argparse.ArgumentParser(description="Gently UI crawler / simulator")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--browser", default="chromium", choices=["chromium", "firefox", "webkit"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=50)
    ap.add_argument("--max-elements", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=8000, help="per-action timeout (ms)")
    ap.add_argument(
        "--headed", action="store_true", help="show a real browser window (watch it live)"
    )
    ap.add_argument("--slow-mo", type=int, default=0, help="delay each action by N ms (watchable)")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="record a Playwright trace (screenshots+DOM+network) -> out/trace.zip",
    )
    ap.add_argument(
        "--video", action="store_true", help="record .webm video of each page -> out/videos/"
    )
    ap.add_argument(
        "--trace-findings",
        action="store_true",
        help="after crawling, replay each found deficiency into its own out/traces/<name>.zip",
    )
    ap.add_argument("--out", default="tools/ui_crawler/out")
    args = ap.parse_args()

    c = Crawler(args)
    t0 = time.time()
    asyncio.run(c.run())
    a = c.write()
    if args.trace_findings:
        print("\n[trace-findings] replaying each deficiency into its own trace...", flush=True)
        asyncio.run(c.replay_findings())
    dt = time.time() - t0
    print(f"\n[done] {len(c.states)} states, {len(c.edges)} edges in {dt:.0f}s → {c.out}/")
    print(
        f"  ↩ returns-to-landing: {len(a['to_landing'])} | 💥 console-err: {len(a['console_err'])} "
        f"| 🌐 http-err: {len(a['http_err'])} | ⏳ spinners: {len(a['spinners'])} "
        f"| 🚫 dead: {len(a['dead'])} | 🚫 unreachable tabs: {len(a['unreachable_tabs'])}"
    )


if __name__ == "__main__":
    main()
