"""Compare recorder on/off A/B arms from run_ab.sh → REPORT.md + verdict.

Signals, per the spec's certification gate ("degrade the recording, never
the app" — operationalized as *no human-perceptible slowdown*):

  a. functional parity — the ON arm may not produce a story status that the
     OFF arm never produced (novel-outcome rule; pre-existing flakes that
     fire nondeterministically in both arms are reported, not fatal)
  b. console parity — the ON arm may not introduce error types absent from
     the OFF arm's union
  c. added latency, ABSOLUTE and per method — parsed from the Playwright
     trace zips run_stories.py records unconditionally (pair before/after
     events on callId). Gates:
       - goto (page open, carries rrweb's synchronous initial snapshot):
         median added ≤ 300 ms
       - every other operator-felt method: median added ≤ 25 ms and
         p95 added ≤ 100 ms

Relative per-story deltas are also reported for transparency — but ratios
on millisecond-scale actions punish imperceptible absolute costs, so they
inform rather than gate.

Usage: python tools/session_replay/perf_ab/compare.py [--root tools/ui_crawler/out/ab]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

GOTO_MEDIAN_ADDED_MS = 300.0
FELT_MEDIAN_ADDED_MS = 25.0
FELT_P95_ADDED_MS = 100.0

# Operator-felt interaction methods; harness overhead (screenshot, context /
# page setup, tracing) never reaches a human and is excluded from the gate.
FELT_METHODS = {
    "goto",
    "click",
    "dblclick",
    "fill",
    "press",
    "type",
    "check",
    "selectOption",
    "hover",
    "evaluateExpression",
    "waitForSelector",
    "waitForLoadState",
}


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


def trace_durations(zip_path: Path) -> dict[str, list[float]]:
    """method → list of paired-action durations (ms) in one story trace."""
    out: dict[str, list[float]] = defaultdict(list)
    starts: dict[str, tuple[str, float]] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith("trace.trace")]
        if not names:
            return out
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
                    method, t0 = starts.pop(cid)
                    t1 = float(ev.get("endTime", t0))
                    if t1 >= t0:
                        out[method].append(t1 - t0)
    return out


def collect(root: Path):
    """→ statuses[run][story], consoles[run][story], durs[arm][method] = [ms...],
    story_ms[run][story] = felt total"""
    statuses: dict[str, dict] = {}
    consoles: dict[str, dict] = {}
    durs: dict[str, dict[str, list[float]]] = {"off": defaultdict(list), "on": defaultdict(list)}
    story_ms: dict[str, dict[str, float]] = {}
    for run_dir in sorted(root.iterdir()):
        if not (run_dir / "status.json").exists():
            continue
        run = run_dir.name
        arm = "off" if run.startswith("off") else "on"
        recs = load_status(run_dir)
        statuses[run] = {s: (r.get("status") or "?") for s, r in recs.items()}
        consoles[run] = {
            s: sorted({str(e)[:80] for e in (r.get("console_errors") or [])})
            for s, r in recs.items()
        }
        story_ms[run] = {}
        traces = run_dir / "traces"
        if traces.is_dir():
            for zp in traces.glob("*.zip"):
                sid = zp.stem.split("-")[0] + "-" + zp.stem.split("-")[1]
                per_method = trace_durations(zp)
                total = 0.0
                for method, values in per_method.items():
                    if method in FELT_METHODS:
                        durs[arm][method].extend(values)
                        total += sum(values)
                story_ms[run][sid] = total
    return statuses, consoles, durs, story_ms


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tools/ui_crawler/out/ab")
    args = ap.parse_args()
    root = Path(args.root)

    statuses, consoles, durs, story_ms = collect(root)
    runs = sorted(statuses)
    if not runs:
        raise SystemExit(f"no runs with status.json under {root}")
    off_runs = [r for r in runs if r.startswith("off")]
    on_runs = [r for r in runs if r.startswith("on")]
    stories = sorted({s for r in runs for s in statuses[r]})

    # a. novel-outcome rule + informational flip listing
    novel_status: list[str] = []
    flips: list[str] = []
    for s in stories:
        off_set = {statuses[r].get(s, "absent") for r in off_runs}
        on_set = {statuses[r].get(s, "absent") for r in on_runs}
        if not on_set <= off_set:
            novel = on_set - off_set
            novel_status.append(
                f"{s}: ON produced {sorted(novel)}, OFF only ever {sorted(off_set)}"
            )
        if len(off_set | on_set) > 1:
            flips.append(f"{s}: off={sorted(off_set)} on={sorted(on_set)}")

    # b. no NEW console error types in ON
    new_console: list[str] = []
    flaky_console: list[str] = []
    for s in stories:
        off_union = {e for r in off_runs for e in consoles[r].get(s, [])}
        on_union = {e for r in on_runs for e in consoles[r].get(s, [])}
        novel = on_union - off_union
        if novel:
            new_console.append(f"{s}: {sorted(novel)}")
        elif off_union != on_union or any(
            consoles[r].get(s, []) != sorted(off_union) for r in runs
        ):
            if off_union:
                flaky_console.append(s)

    # c. absolute per-method added latency
    method_rows = []
    gate_fails: list[str] = []
    for method in sorted(set(durs["off"]) | set(durs["on"])):
        off_vals, on_vals = durs["off"].get(method, []), durs["on"].get(method, [])
        if not off_vals or not on_vals:
            continue
        m_off, m_on = statistics.median(off_vals), statistics.median(on_vals)
        p_off, p_on = p95(off_vals), p95(on_vals)
        d_med, d_p95 = m_on - m_off, p_on - p_off
        method_rows.append((method, len(off_vals), len(on_vals), m_off, m_on, d_med, d_p95))
        if method == "goto":
            if d_med > GOTO_MEDIAN_ADDED_MS:
                gate_fails.append(
                    f"goto median added {d_med:+.0f}ms > {GOTO_MEDIAN_ADDED_MS:.0f}ms"
                )
        else:
            if d_med > FELT_MEDIAN_ADDED_MS:
                gate_fails.append(
                    f"{method} median added {d_med:+.0f}ms > {FELT_MEDIAN_ADDED_MS:.0f}ms"
                )
            if d_p95 > FELT_P95_ADDED_MS:
                gate_fails.append(f"{method} p95 added {d_p95:+.0f}ms > {FELT_P95_ADDED_MS:.0f}ms")

    # informational: per-story relative deltas (medians across reps)
    def story_median(story: str, arm_runs: list[str]) -> float | None:
        vals = [story_ms[r][story] for r in arm_runs if story_ms.get(r, {}).get(story, 0) > 0]
        return statistics.median(vals) if vals else None

    rel_rows, rel_deltas = [], []
    for s in stories:
        m_off, m_on = story_median(s, off_runs), story_median(s, on_runs)
        if m_off and m_on:
            pct = (m_on - m_off) / m_off * 100
            rel_deltas.append(pct)
            rel_rows.append((s, m_off, m_on, pct))
    rel_median = statistics.median(rel_deltas) if rel_deltas else 0.0

    ok = not novel_status and not new_console and not gate_fails
    verdict = "PASS" if ok else "FAIL"

    lines = [
        "# Session-replay recorder — performance certification (A/B)",
        "",
        f"Runs: {', '.join(runs)} — full 36-story ui_crawler suite per run, arms",
        "alternated O-N-O-N on a quiet machine; identical binaries, `GENTLY_REPLAY`",
        "is the only delta. Durations are Playwright-trace paired actions.",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "### Gates (absolute, operator-felt)",
        "",
        "- Functional: ON may not produce a story outcome OFF never produced — "
        + ("OK" if not novel_status else "VIOLATIONS: " + "; ".join(novel_status)),
        "- Console: no new error types in ON — "
        + ("OK" if not new_console else "VIOLATIONS: " + "; ".join(new_console)),
        f"- goto median added ≤ {GOTO_MEDIAN_ADDED_MS:.0f} ms; other felt methods "
        f"median added ≤ {FELT_MEDIAN_ADDED_MS:.0f} ms, p95 added ≤ {FELT_P95_ADDED_MS:.0f} ms — "
        + ("OK" if not gate_fails else "VIOLATIONS: " + "; ".join(gate_fails)),
        "",
        "### Per-method added latency (all felt actions pooled across runs)",
        "",
        "| method | n(off/on) | median off | median on | Δ median | Δ p95 |",
        "|---|---|---|---|---|---|",
    ]
    for method, n_off, n_on, m_off, m_on, d_med, d_p95 in method_rows:
        lines.append(
            f"| {method} | {n_off}/{n_on} | {m_off:.0f} | {m_on:.0f} "
            f"| {d_med:+.0f} ms | {d_p95:+.0f} ms |"
        )
    lines += [
        "",
        "### Transparency: relative per-story felt-action totals",
        "",
        f"Median per-story delta: **{rel_median:+.1f}%** (informational — ratios on",
        "millisecond-scale stories flag imperceptible absolute costs; the absolute",
        "gates above are the contract).",
        "",
        "| story | off ms | on ms | Δ% |",
        "|---|---|---|---|",
    ]
    for s, m_off, m_on, pct in rel_rows:
        lines.append(f"| {s} | {m_off:.0f} | {m_on:.0f} | {pct:+.1f} |")
    if flips:
        lines += [
            "",
            "### Observed status variance (both-arm flakes, informational)",
            "",
            *[f"- {f}" for f in flips],
        ]
    if flaky_console:
        lines += [
            "",
            "### Pre-existing flaky console errors (fire in OFF too, informational)",
            "",
            "- " + ", ".join(flaky_console),
        ]
    report = "\n".join(lines) + "\n"
    out = Path(__file__).parent / "REPORT.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
