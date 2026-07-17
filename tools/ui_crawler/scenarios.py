#!/usr/bin/env python3
# ruff: noqa: E501
"""Scripted reproductions of the STATIC-audit deficiencies, each captured as a
Playwright trace you can scrub in `playwright show-trace`.

crawler.py finds deficiencies by *walking* the app (trace them with
`crawler.py --trace-findings`). The static code-audit findings — missing export,
no ground-truth annotation, the dead notebook Questions tab, the view-only plan
wizard that spins — aren't reachable by blind crawling. This harness navigates to
each surface deliberately and records, in a trace, exactly what is absent or
broken. Rig/agent-only findings (LED force-close, scripted_protocol false
success, autonomous-turn stop) are listed as NOT headless-reproducible.

Run (control shim on 8080; optional account-mode server on 8081 for view-only):
    python launch_gently.py --no-api --no-auth --no-browser
    GENTLY_VIZ_PORT=8081 GENTLY_STORAGE_PATH=/tmp/gently_acct python launch_gently.py --no-api --no-browser
    uv run python tools/ui_crawler/scenarios.py --url http://localhost:8080 \
        --account-url http://localhost:8081 --out tools/ui_crawler/out
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

TIMEOUT = 12000


async def _goto(page, url, path="/"):
    await page.goto(url + path, wait_until="domcontentloaded", timeout=TIMEOUT)
    await asyncio.sleep(0.7)


async def _skip_landing(page):
    """Dismiss the landing overlay to reach the workspace."""
    try:
        await page.evaluate(
            "() => { const s=document.getElementById('v2-landing-skip'); if (s) s.click(); }"
        )
        await asyncio.sleep(0.4)
    except Exception:
        pass


async def _tab(page, tab):
    try:
        await page.click(f'[data-tab="{tab}"]', timeout=6000)
        await asyncio.sleep(0.7)
    except Exception:
        pass


async def _count_text(page, regex):
    """Count visible clickable controls whose text matches regex (case-insensitive)."""
    return await page.evaluate(
        """(q) => {
          const vis = el => { const cs=getComputedStyle(el); if (cs.display==='none'||cs.visibility==='hidden') return false;
            const r=el.getBoundingClientRect(); return r.width>1 && r.height>1; };
          const re = new RegExp(q, 'i');
          return [...document.querySelectorAll('button, a, [role=button], .btn')].filter(e => vis(e) && re.test(e.textContent||'')).length;
        }""",
        regex,
    )


# --- scenarios: each returns an "observed" string proving the finding ---


async def sc_no_export(page, url):
    seen = 0
    for tab in ("embryos", "gallery", "plans"):
        await _tab(page, tab)
        seen += await _count_text(page, r"export|download\s*(csv|data|predictions|results)")
    return f"export/download-data controls found across embryos/gallery/plans: {seen} (expected >0 for a scientific tool)"


async def sc_no_ground_truth(page, url):
    await _tab(page, "embryos")
    controls = await _count_text(page, r"correct stage|set (ground )?truth|annotate|fix stage")
    return f"ground-truth 'set correct stage' controls on embryos surface: {controls} (only Agree/Disagree exists, which is localStorage-only)"


async def sc_notebook_questions(page, url):
    await _tab(page, "notebook")
    q_tab = await _count_text(page, r"^questions$")
    # click a Questions tab if present, then measure content
    try:
        await page.click("text=/^Questions$/i", timeout=4000)
        await asyncio.sleep(0.6)
    except Exception:
        pass
    body = await page.evaluate(
        "() => (document.querySelector('#tab-notebook, .notebook, main')||document.body).innerText.slice(0,200)"
    )
    return f"notebook Questions tab present={q_tab > 0}; content after click: {body!r}"


async def sc_snap_silent_503(page, url):
    http, console = [], []
    page.on("response", lambda r: http.append(r.status) if r.status >= 400 else None)
    page.on("console", lambda m: console.append(m.text[:80]) if m.type == "error" else None)
    await _tab(page, "devices")
    try:
        await page.click('[data-view="manual"]', timeout=5000)
        await asyncio.sleep(0.6)
    except Exception:
        pass
    http.clear()
    console.clear()
    clicked = False
    for label in ("Snap", "Acquire", "Capture"):
        try:
            await page.click(f"text=/^{label}$/i", timeout=3000)
            clicked = True
            break
        except Exception:
            continue
    await asyncio.sleep(1.2)
    toast = await _count_text(page, r"error|failed|not connected|offline")
    return f"snap/acquire clicked={clicked}; HTTP≥400 seen={sorted(set(http))}; error-toasts shown={toast} (device offline → silent failure)"


async def sc_no_create_campaign(page, url):
    await _tab(page, "plans")
    btn = await _count_text(
        page, r"new campaign|create campaign|\+\s*campaign|new plan|create plan"
    )
    return f"create-campaign/new-plan buttons on plans tab: {btn} (creation is agent-tool only)"


async def sc_temperature_alerts(page, url):
    await _goto(page, url, "/settings")
    alerts = await _count_text(page, r"alert|threshold.*(alarm|notify)|out.?of.?range")
    return f"temperature-alert controls in Settings: {alerts} (no drift/out-of-range/fault alerting exists)"


async def sc_mesh_invisible(page, url):
    await _skip_landing(page)
    mesh = await _count_text(page, r"mesh|instances|peers|import (session|from)")
    return f"mesh/instances/session-import controls in the web UI: {mesh} (mesh backend exists but is unexposed)"


async def sc_view_only_plan_spins(page, url):
    # account-mode server, logged out → 'Plan an experiment' should spin forever
    await _goto(page, url, "/")
    try:
        await page.click('[data-landing="plan"]', timeout=6000)
    except Exception:
        return "could not click 'Plan an experiment' (is this an account-mode server?)"
    await asyncio.sleep(3.0)
    spinning = await page.evaluate(
        "() => { const t=document.querySelector('.v2-thinking:not(.hidden), .spinner, [aria-busy=\"true\"]'); return !!t; }"
    )
    plan_items = await page.evaluate(
        "() => document.querySelectorAll('.v2-plan-item, [data-plan-item]').length"
    )
    return f"view-only 'Plan an experiment' → spinner present={spinning}, plan items rendered={plan_items} (wizard spins, no take-control prompt)"


SCENARIOS = [
    ("no-export", "No perception/experiment data export", "high", sc_no_export, False),
    ("no-ground-truth", "No ground-truth annotation UI", "high", sc_no_ground_truth, False),
    (
        "judge-notebook-questions",
        "Notebook Questions tab dead / no note detail",
        "medium",
        sc_notebook_questions,
        False,
    ),
    (
        "snap-silent-503",
        "One-off snap silently fails when device offline",
        "medium",
        sc_snap_silent_503,
        False,
    ),
    (
        "no-create-campaign",
        "No UI to create a campaign / new plan",
        "medium",
        sc_no_create_campaign,
        False,
    ),
    (
        "temperature-alerts",
        "Temperature alerts do not exist",
        "medium",
        sc_temperature_alerts,
        False,
    ),
    (
        "mesh-invisible",
        "Mesh + session import invisible from web",
        "medium",
        sc_mesh_invisible,
        False,
    ),
    (
        "view-only-plan-spins",
        "View-only user gets a permanently spinning plan wizard",
        "medium",
        sc_view_only_plan_spins,
        True,
    ),
]

# Findings that need real hardware or a live agent turn — not headless-reproducible:
RIG_ONLY = [
    (
        "led-force-close",
        "LED force-close safety violated on step-leave — needs device LED state + Operate focus step",
    ),
    (
        "scripted-protocol-false-success",
        "scripted_protocol reports success while nothing runs — needs device + Run flow",
    ),
    (
        "answer-silently-queued",
        "Typed answer to agent question silently queued — needs a pending agent ask",
    ),
    (
        "autonomous-not-stoppable",
        "Autonomous wake turns can't be stopped from web — needs an autonomous turn",
    ),
]


async def main_async(args):
    out = Path(args.out) / "scenarios"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    async with async_playwright() as p:
        launch_args = ["--disable-dev-shm-usage", "--no-sandbox"]
        browser = await p.chromium.launch(
            headless=not args.headed, slow_mo=args.slow_mo, args=launch_args
        )
        for name, title, sev, fn, needs_account in SCENARIOS:
            target = args.account_url if needs_account else args.url
            if needs_account and not args.account_url:
                print(f"  [skip] {name}: needs --account-url (account-mode server)", flush=True)
                manifest.append(
                    {
                        "scenario": name,
                        "title": title,
                        "severity": sev,
                        "trace": None,
                        "observed": "skipped — no account-mode server provided",
                    }
                )
                continue
            context = await browser.new_context(viewport={"width": 1440, "height": 900})
            await context.tracing.start(screenshots=True, snapshots=True, sources=True, title=title)
            page = await context.new_page()
            page.set_default_timeout(TIMEOUT)
            try:
                await _goto(page, target, "/")
                if not needs_account:
                    await _skip_landing(page)
                observed = await fn(page, target)
            except Exception as exc:
                observed = f"scenario error: {exc}"
            await context.tracing.stop(path=str(out / f"{name}.zip"))
            await context.close()
            manifest.append(
                {
                    "scenario": name,
                    "title": title,
                    "severity": sev,
                    "trace": f"{name}.zip",
                    "observed": observed,
                }
            )
            print(f"  [trace] {sev:6} {name:26} → {name}.zip\n          {observed}", flush=True)
        await browser.close()
    (out / "index.json").write_text(
        json.dumps({"scenarios": manifest, "rig_only": RIG_ONLY}, indent=2)
    )
    print(f"\n[scenarios] {len([m for m in manifest if m['trace']])} traced → {out}/")
    print("  rig/agent-only (not headless-reproducible):")
    for n, why in RIG_ONLY:
        print(f"    - {n}: {why}")
    print(f"\n  scrub one:  uv run playwright show-trace {out}/<name>.zip")


def main():
    ap = argparse.ArgumentParser(
        description="Scripted static-audit deficiency reproductions → traces"
    )
    ap.add_argument("--url", default="http://localhost:8080", help="control-mode viz server")
    ap.add_argument(
        "--account-url",
        default=None,
        help="account-mode (logged-out) server for view-only scenarios",
    )
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--slow-mo", type=int, default=0)
    ap.add_argument("--out", default="tools/ui_crawler/out")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
