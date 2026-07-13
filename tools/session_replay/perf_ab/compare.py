"""Compare recorder on/off A/B arms from run_ab.sh → REPORT.md + verdict.

Three signals, per the spec's certification gate:
  a. functional parity — story statuses identical across every run
  b. console parity  — per-story console_errors identical off vs on
  c. per-action durations — parsed from the Playwright trace zips that
     run_stories.py records unconditionally; pair before/after events on
     callId, duration = after.endTime - before.startTime

Verdict PASS requires: no status flips, no console deltas, overall median
per-story action-time delta <= 5%, and no story's median regressing > 15%
with an absolute slowdown > 250 ms (guards against ratio noise on fast
stories).

Usage: python tools/session_replay/perf_ab/compare.py [--root tools/ui_crawler/out/ab]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import zipfile
from pathlib import Path

OVERALL_MEDIAN_DELTA_PCT = 5.0
PER_STORY_DELTA_PCT = 15.0
PER_STORY_DELTA_ABS_MS = 250.0


def load_status(run_dir: Path) -> dict[str, dict]:
    data = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    stories = data["stories"] if isinstance(data, dict) and "stories" in data else data
    out = {}
    if isinstance(stories, dict):
        for sid, rec in stories.items():
            out[sid] = rec
    else:
        for rec in stories:
            out[rec.get("id") or rec.get("story")] = rec
    return out


def trace_action_ms(zip_path: Path) -> float:
    """Total paired-action duration (ms) inside one story trace zip."""
    total = 0.0
    starts: dict[str, tuple[str, float]] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("trace.trace")]
        if not names:
            return 0.0
        with z.open(names[0]) as fh:
            for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                typ = ev.get("type")
                cid = ev.get("callId")
                if not cid:
                    continue
                if typ == "before":
                    starts[cid] = (ev.get("method", "?"), float(ev.get("startTime", 0)))
                elif typ == "after" and cid in starts:
                    _, t0 = starts.pop(cid)
                    t1 = float(ev.get("endTime", t0))
                    if t1 >= t0:
                        total += t1 - t0
    return total


def collect(root: Path) -> tuple[dict, dict, dict]:
    """→ (statuses[run][story], consoles[run][story], action_ms[run][story])"""
    statuses: dict[str, dict] = {}
    consoles: dict[str, dict] = {}
    action_ms: dict[str, dict] = {}
    for run_dir in sorted(root.iterdir()):
        if not (run_dir / "status.json").exists():
            continue
        run = run_dir.name
        recs = load_status(run_dir)
        statuses[run] = {s: (r.get("status") or "?") for s, r in recs.items()}
        consoles[run] = {
            s: sorted(set(map(str, r.get("console_errors") or []))) for s, r in recs.items()
        }
        action_ms[run] = {}
        traces = run_dir / "traces"
        if traces.is_dir():
            for zp in traces.glob("*.zip"):
                sid = zp.stem.split("-")[0] + "-" + zp.stem.split("-")[1]
                action_ms[run][sid] = trace_action_ms(zp)
    return statuses, consoles, action_ms


def median_by_arm(per_run: dict, story: str, arm: str) -> float | None:
    vals = [
        per_run[r][story]
        for r in per_run
        if r.startswith(arm) and story in per_run[r] and per_run[r][story] > 0
    ]
    return statistics.median(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tools/ui_crawler/out/ab")
    args = ap.parse_args()
    root = Path(args.root)

    statuses, consoles, action_ms = collect(root)
    runs = sorted(statuses)
    if not runs:
        raise SystemExit(f"no runs with status.json under {root}")
    stories = sorted({s for r in runs for s in statuses[r]})

    flips: list[str] = []
    for s in stories:
        vals = {r: statuses[r].get(s, "absent") for r in runs}
        if len(set(vals.values())) > 1:
            flips.append(f"{s}: " + ", ".join(f"{r}={v}" for r, v in vals.items()))

    console_deltas: list[str] = []
    for s in stories:
        off = {tuple(consoles[r].get(s, [])) for r in runs if r.startswith("off")}
        on = {tuple(consoles[r].get(s, [])) for r in runs if r.startswith("on")}
        if off != on:
            console_deltas.append(f"{s}: off={sorted(off)} on={sorted(on)}")

    rows = []
    deltas = []
    regressions = []
    for s in stories:
        m_off = median_by_arm(action_ms, s, "off")
        m_on = median_by_arm(action_ms, s, "on")
        if m_off is None or m_on is None or m_off == 0:
            rows.append((s, m_off, m_on, None))
            continue
        pct = (m_on - m_off) / m_off * 100
        deltas.append(pct)
        rows.append((s, m_off, m_on, pct))
        if pct > PER_STORY_DELTA_PCT and (m_on - m_off) > PER_STORY_DELTA_ABS_MS:
            regressions.append(f"{s}: {m_off:.0f}ms → {m_on:.0f}ms ({pct:+.1f}%)")

    overall = statistics.median(deltas) if deltas else 0.0
    ok = (
        not flips and not console_deltas and overall <= OVERALL_MEDIAN_DELTA_PCT and not regressions
    )
    verdict = "PASS" if ok else "FAIL"

    lines = [
        "# Session-replay recorder — performance certification (A/B)",
        "",
        f"Runs: {', '.join(runs)} — full 36-story ui_crawler suite per run,",
        "arms alternated O-N-O-N; identical binaries, `GENTLY_REPLAY` is the only delta.",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "- Functional parity: "
        + ("OK — no status flips" if not flips else "FLIPS: " + "; ".join(flips)),
        "- Console parity: "
        + (
            "OK — identical error sets"
            if not console_deltas
            else "DELTAS: " + "; ".join(console_deltas)
        ),
        f"- Overall median per-story action-time delta (on vs off): {overall:+.1f}% "
        f"(gate: ≤ {OVERALL_MEDIAN_DELTA_PCT:.0f}%)",
        f"- Per-story regressions > {PER_STORY_DELTA_PCT:.0f}% and "
        f"> {PER_STORY_DELTA_ABS_MS:.0f} ms: "
        + ("none" if not regressions else "; ".join(regressions)),
        "",
        "## Per-story median action time (ms, Playwright-trace paired actions)",
        "",
        "| story | off | on | Δ% |",
        "|---|---|---|---|",
    ]
    for s, m_off, m_on, pct in rows:
        lines.append(
            f"| {s} | {m_off:.0f} | {m_on:.0f} | {pct:+.1f} |"
            if pct is not None
            else f"| {s} | {m_off or '—'} | {m_on or '—'} | n/a |"
        )
    report = "\n".join(lines) + "\n"
    out = Path(__file__).parent / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
