#!/usr/bin/env python3
# ruff: noqa: E501
"""Per-story UX audit + regression spine.

Discovers tools/ui_crawler/stories/US-*.py (each defines META + `async flow(page, url, rec)`),
runs every flow in its own Playwright trace, and writes a 4-state status report
(works / partial / gap / blocked) + STATUS.md. That report IS the audit.

Committed as tools/ui_crawler/baseline/status.json, diffing a fresh run against it
IS the regression signal: a story that FLIPS (e.g. works→gap) is a break to triage —
fix the UX, or, if the change is a deliberate paradigm shift, update the story doc +
re-baseline (`--update-baseline`). Exit code is non-zero if any story got worse, so
this can gate a pipeline.

    uv run python tools/ui_crawler/run_stories.py --url http://localhost:8080 \
        [--account-url http://localhost:8081] [--update-baseline] [--docs-status]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).parent
STORIES_DIR = HERE / "stories"
BASELINE = HERE / "baseline" / "status.json"
ICON = {"works": "✅", "partial": "◑", "gap": "⚠", "blocked": "⏳", "unknown": "❔"}
RANK = {"works": 3, "partial": 2, "gap": 1, "blocked": 0, "unknown": 0}


def _load(pyfile):
    spec = importlib.util.spec_from_file_location(pyfile.stem, pyfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _slim(meta):
    return {
        "id": meta["id"],
        "title": meta["title"],
        "cluster": meta.get("cluster", ""),
        "mode": meta.get("mode", "headless"),
    }


async def run(args):
    sys.path.insert(0, str(STORIES_DIR))
    from _harness import Rec

    files = sorted(STORIES_DIR.glob("US-*.py"))
    out = Path(args.out)
    tdir = out / "traces"
    sdir = out / "shots"
    tdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not args.headed,
            slow_mo=args.slow_mo,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        for f in files:
            mod = _load(f)
            meta = mod.META
            rec = Rec()
            target = args.account_url if meta.get("needs_account") else args.url
            if meta.get("needs_account") and not args.account_url:
                results.append(
                    {
                        **_slim(meta),
                        "status": "blocked",
                        "observed": "needs --account-url",
                        "trace": None,
                        "shots": [],
                        "console_errors": [],
                        "screen_text": "",
                    }
                )
                print(
                    f"  [blocked] {meta['id']:6} {meta['title'][:44]:44} (needs account server)",
                    flush=True,
                )
                continue
            context = await browser.new_context(viewport={"width": 1440, "height": 900})
            await context.tracing.start(
                screenshots=True, snapshots=True, sources=True, title=meta["id"]
            )
            page = await context.new_page()
            page.set_default_timeout(12000)
            page.on(
                "console",
                lambda m, _r=rec: _r.console.append(m.text[:160]) if m.type == "error" else None,
            )
            rec._page, rec._dir, rec._id = page, sdir, meta["id"]
            try:
                await mod.flow(page, target, rec)
            except Exception as e:
                rec.blocked(f"flow error: {e}")
            try:
                await rec.shot("final")  # always capture where the story ended (agent-readable PNG)
            except Exception:
                pass
            try:
                screen_text = await page.evaluate(
                    "() => (document.body.innerText || '').replace(/\\n{2,}/g, '\\n').trim().slice(0, 1200)"
                )
            except Exception:
                screen_text = ""
            zip_name = f"{f.stem}.zip"
            await context.tracing.stop(path=str(tdir / zip_name))
            await context.close()
            st = rec.status or "unknown"
            results.append(
                {
                    **_slim(meta),
                    "status": st,
                    "observed": rec.observed,
                    "trace": f"traces/{zip_name}",
                    "shots": [f"shots/{s}" for s in rec.shots],
                    "console_errors": sorted(set(rec.console))[:8],
                    "screen_text": screen_text,
                }
            )
            print(
                f"  [{st:7}] {meta['id']:6} {meta['title'][:42]:42} {ICON.get(st, '')}  ({len(rec.shots)} shot)",
                flush=True,
            )
        await browser.close()

    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps({"url": args.url, "stories": results}, indent=2))
    md = _status_md(results)
    (out / "STATUS.md").write_text(md)
    if args.docs_status:
        Path("docs/user-stories/STATUS.md").write_text(md)
        print("[docs] wrote docs/user-stories/STATUS.md")

    flips = _diff(results)
    if args.update_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps(
                {r["id"]: {"status": r["status"], "title": r["title"]} for r in results}, indent=2
            )
        )
        print(f"[baseline] updated {BASELINE}")
    return results, flips


def _diff(results):
    if not BASELINE.exists():
        print("\n[baseline] none yet — run with --update-baseline to create one")
        return []
    base = json.loads(BASELINE.read_text())
    flips = [
        (r["id"], base[r["id"]]["status"], r["status"], r["title"])
        for r in results
        if r["id"] in base and base[r["id"]]["status"] != r["status"]
    ]
    new = [r["id"] for r in results if r["id"] not in base]
    if flips:
        print("\n[regression-diff] status FLIPS vs baseline:")
        for sid, was, now, title in flips:
            tag = "⬇ REGRESSION" if RANK[now] < RANK[was] else "⬆ improved"
            print(f"  {sid} {was} → {now}  {tag}  ({title})")
    else:
        print("\n[regression-diff] no status flips vs baseline ✅")
    if new:
        print(f"  (+{len(new)} new stories not in baseline: {', '.join(new)})")
    return flips


def _status_md(results):
    n = {
        k: sum(1 for r in results if r["status"] == k)
        for k in ("works", "partial", "gap", "blocked")
    }
    lines = [
        "# Gently — UX audit status",
        "",
        f"Generated by `tools/ui_crawler/run_stories.py`. "
        f"✅ {n['works']} works · ◑ {n['partial']} partial · ⚠ {n['gap']} gap · ⏳ {n['blocked']} blocked (rig/agent).",
        "",
    ]
    by_cluster = {}
    for r in results:
        by_cluster.setdefault(r["cluster"] or "—", []).append(r)
    for cluster, rows in by_cluster.items():
        lines.append(f"### {cluster}")
        lines.append("")
        lines.append("| Story | Status | Observed | Screenshot | Console |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            shot = f"`{r['shots'][0]}`" if r.get("shots") else "—"
            cons = f"{len(r['console_errors'])} err" if r.get("console_errors") else "—"
            lines.append(
                f"| {r['id']} {r['title']} | {ICON.get(r['status'], '')} {r['status']} | {r['observed']} | {shot} | {cons} |"
            )
        lines.append("")
    lines.append(
        "_Per-story PNG screenshots in `out/stories/shots/`; full per-step trace in `out/stories/traces/` "
        "(`playwright show-trace`); final screen text + console in `status.json`._"
    )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Per-story UX audit + baseline-diff regression")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument(
        "--account-url", default=None, help="account-mode (logged-out) server for view-only stories"
    )
    ap.add_argument("--out", default="tools/ui_crawler/out/stories")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--slow-mo", type=int, default=0)
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="write the current run as the committed baseline",
    )
    ap.add_argument(
        "--docs-status", action="store_true", help="also write docs/user-stories/STATUS.md"
    )
    args = ap.parse_args()
    _, flips = asyncio.run(run(args))
    regressions = [f for f in flips if RANK[f[2]] < RANK[f[1]]]
    print(
        f"\n[done] {'REGRESSIONS' if regressions else 'no regressions'} "
        f"({len(regressions)} worse) → {args.out}/STATUS.md"
    )
    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
