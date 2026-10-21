#!/usr/bin/env python3
"""Seed two projects with sessions, a plan, a ledger and references, so the
Projects tab, the Sessions tab and Home can be judged against real content.

Everything is written through the real stores, so the result is what an
acquired night plus a planned project would leave behind: sessions with
embryos and projections (``tools/seed_demo_session.py`` does those), a
checkpoint that says how each run ended, an event capture with a few errors
and warnings, an interaction log with turns and corrections, campaigns with
phases and items in every state, expectations, questions, findings and the
references a project keeps.

    GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python tools/seed_project_layer.py

Not a test fixture and not for CI. It adds; it never touches what is there.

ponytail: the embryo data is the demo seeder's synthetic blobs, enough for
holdings, thumbnails and layout. The checkpoints and logs are written as
files, the way the orchestrator and the loggers leave them.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import seed_demo_session as demo  # noqa: E402

from gently.core.file_store import FileStore  # noqa: E402
from gently.harness.memory.file_store import FileContextStore  # noqa: E402
from gently.harness.memory.model import Expectation, Question  # noqa: E402
from gently.harness.memory.notebook import Author, Note, NoteKind, NoteStatus  # noqa: E402
from gently.settings import settings  # noqa: E402


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _dir(store: FileStore, sid: str) -> Path:
    folder = store._session_dir(sid)
    assert folder is not None, sid
    return Path(folder)


def _checkpoint(
    store: FileStore,
    sid: str,
    *,
    status: str,
    started: datetime,
    interval: int,
    timepoints: int,
    going: int,
) -> None:
    """A timelapse.yaml the way save_state leaves it, for how the run ended."""
    eids = store.list_embryo_ids(sid)
    rows = {e: {"is_complete": i >= going} for i, e in enumerate(eids)}
    doc = {
        "status": status,
        "started_at": _iso(started),
        "saved_at": _iso(started + timedelta(seconds=interval * timepoints)),
        "current_round": timepoints,
        "total_timepoints": timepoints,
        "base_interval_seconds": interval,
        "embryos": rows,
    }
    (_dir(store, sid) / "timelapse.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


def _events(
    store: FileStore,
    sid: str,
    started: datetime,
    eids: list[str],
    *,
    errors: int,
    warnings: int,
    ended: str,
) -> None:
    out = [
        {
            "event_type": "ACQUISITION_STARTED",
            "data": {"embryo_ids": eids},
            "timestamp": _iso(started),
        },
        {
            "event_type": "DETECTION_TRIGGERED",
            "data": {"detector_name": "comma", "embryo_id": eids[0]},
            "timestamp": _iso(started + timedelta(minutes=7)),
        },
    ]
    for i in range(warnings):
        out.append(
            {
                "event_type": "WARNING_ISSUED",
                "data": {"message": f"561 nm power {3 + i} % above the plan's 3 %"},
                "timestamp": _iso(started + timedelta(minutes=3 + 40 * i)),
            }
        )
    for i in range(errors):
        out.append(
            {
                "event_type": "ERROR_OCCURRED",
                "data": {"message": "stage move timed out"},
                "timestamp": _iso(started + timedelta(hours=6 + i)),
            }
        )
    end = {
        "complete": "ACQUISITION_COMPLETED",
        "cut": "ACQUISITION_STOPPED",
        "failed": "ACQUISITION_FAILED",
    }
    if ended in end:
        data = (
            {"reason": "stopped by operator"}
            if ended == "cut"
            else ({"error": "stage timeout"} if ended == "failed" else {})
        )
        out.append(
            {
                "event_type": end[ended],
                "data": data,
                "timestamp": _iso(started + timedelta(hours=8)),
            }
        )
    path = _dir(store, sid) / "events.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in out) + "\n", encoding="utf-8")


def _interactions(store: FileStore, sid: str, turns: int, corrections: int) -> None:
    rows = []
    for i in range(turns):
        rows.append(
            {
                "interaction_id": uuid.uuid4().hex[:8],
                "session_id": sid,
                "user_prompt": "…",
                "tool_calls": [{"tool_name": "get_status"}] if i % 2 else [],
                "was_corrected": i < corrections,
                "error": None,
            }
        )
    path = _dir(store, sid) / "interaction_log.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def seed(root: Path, *, timepoints: int) -> None:
    store = FileStore(root)
    cs = FileContextStore(root / "agent")
    nb = cs.notebook
    now = datetime.now().replace(second=0, microsecond=0)

    # ---- project one: the one the mockup shows -------------------------------
    p1 = cs.create_campaign(
        description="Does a 20 → 25 °C shift change the twitching-to-hatching interval?",
        shorthand="hatching-temp",
        target="Interval at 25 °C against the 20 °C baseline, six embryos each",
        strains=[
            {
                "source": "cgc",
                "key": "OH10689",
                "title": "OH10689",
                "subtitle": "otIs355 [rab-3p(prom1)::2xNLS::TagRFP] IV",
                "url": "https://cgc.umn.edu/strain/OH10689",
                "added_by": "human",
                "added_at": _iso(now),
            },
            {
                "source": "cgc",
                "key": "N2",
                "title": "N2",
                "subtitle": "wild type",
                "url": "https://cgc.umn.edu/strain/N2",
                "added_by": "human",
                "added_at": _iso(now),
            },
        ],
        references=[
            {
                "source": "alliance",
                "key": "WBGene00004267",
                "title": "rab-3",
                "subtitle": "C. elegans",
                "url": "https://wormbase.org/species/c_elegans/gene/WBGene00004267",
                "added_by": "agent",
                "added_at": _iso(now),
            },
            {
                "source": "ols",
                "key": "WBls:0000017",
                "title": "comma embryo Ce",
                "url": "https://wormbase.org/species/all/life_stage/WBls:0000017",
                "added_by": "agent",
                "added_at": _iso(now),
            },
            {
                "source": "europepmc",
                "key": "PMID:26291158",
                "title": "Regulatory logic of pan-neuronal gene expression in C. elegans",
                "subtitle": "Stefanakis N, Carrera I, Hobert O · Neuron · 2015",
                "url": "https://europepmc.org/article/MED/26291158",
                "added_by": "human",
                "added_at": _iso(now),
            },
        ],
    )
    ph1 = cs.create_campaign(description="Baseline at 20 °C", parent_id=p1)
    ph2 = cs.create_campaign(description="Shift to 25 °C", parent_id=p1)
    base_spec = {
        "strain": "OH10689",
        "genotype": "otIs355 [rab-3p(prom1)::2xNLS::TagRFP] IV",
        "temperature_c": 20.0,
        "num_embryos": 6,
        "num_slices": 80,
        "exposure_ms": 10.0,
        "laser_wavelength_nm": 488,
        "laser_power_pct": 3.0,
        "interval_s": 90,
        "stop_condition": "hatching",
    }
    i11 = cs.create_plan_item(
        campaign_id=ph1,
        type="imaging",
        title="Six embryos until hatching",
        spec=base_spec,
        origin="agent",
    )
    i12 = cs.create_plan_item(
        campaign_id=ph1,
        type="imaging",
        title="Repeat",
        spec=base_spec,
        inherit_from=i11,
        depends_on=[i11],
        origin="agent",
    )
    i21 = cs.create_plan_item(
        campaign_id=ph2,
        type="imaging",
        title="Six embryos at 25 °C",
        spec=dict(base_spec, temperature_c=25.0, interval_s=60),
        origin="agent",
    )
    i22 = cs.create_plan_item(
        campaign_id=ph2,
        type="imaging",
        title="Quick look at 25 °C",
        spec=dict(base_spec, temperature_c=25.0, interval_s=30, num_embryos=2),
        origin="manual",
    )
    cs.create_plan_item(
        campaign_id=ph2,
        type="imaging",
        title="Repeat 25 °C with 488 only",
        description="So 561 is not confounded with temperature.",
        spec=dict(base_spec, temperature_c=25.0, interval_s=60),
        status="proposed",
        origin="agent",
    )
    i3 = cs.create_plan_item(
        campaign_id=p1,
        type="decision_point",
        title="Decide: keep 561?",
        depends_on=[i21],
        origin="agent",
    )
    del i3

    # Nights. Each is a real seeded session, then filed and given its records.
    nights = [
        # (title, item, days_ago, embryos, status, errors, warnings, turns, corrections, ended)
        ("baseline, first night", i11, 38, 5, "failed", 2, 0, 22, 1, "failed"),
        ("baseline, second night", i11, 32, 6, "completed", 0, 2, 12, 0, "complete"),
        ("repeat at 20 °C", i12, 26, 6, "completed", 0, 0, 6, 0, "complete"),
        ("baseline, third night", i11, 21, 6, "running", 1, 3, 31, 2, "cut"),
        ("quick look at 25 °C", i22, 20, 2, "completed", 0, 1, 14, 1, "complete"),
    ]
    sids: dict[str, str] = {}
    for n, (title, item, days_ago, n_emb, status, errs, warns, turns, corr, ended) in enumerate(
        nights
    ):
        sid = demo.seed(store, n_embryos=n_emb, n_timepoints=timepoints, seed_val=100 + n)
        store.update_session(sid, name=title)
        started = now - timedelta(days=days_ago, hours=3)
        eids = store.list_embryo_ids(sid)
        going = 2 if status == "running" else 0
        _checkpoint(
            store,
            sid,
            status=status,
            started=started,
            interval=90 if n_emb > 2 else 30,
            timepoints=timepoints,
            going=going,
        )
        _events(store, sid, started, eids, errors=errs, warnings=warns, ended=ended)
        _interactions(store, sid, turns, corr)
        from gently.app.filing import file_session_under

        planned = cs.get_plan_item(item)
        assert planned is not None
        file_session_under(store, cs, sid, planned.campaign_id, item)
        sids[title] = sid
    cs.complete_plan_item(i11, "Twitching-to-hatching 715 ± 22 min (n = 14) at 20 °C.")
    cs.complete_plan_item(i12, "702 ± 31 min (n = 6); consistent with 1.1.")
    cs.update_plan_item(
        i22,
        status=__import__(
            "gently.harness.memory.model", fromlist=["PlanItemStatus"]
        ).PlanItemStatus.COMPLETED,
        outcome="Two embryos hatched at 588 and 601 min.",
    )

    # The ledger's other columns.
    s3 = sids["baseline, third night"]
    s5 = sids["quick look at 25 °C"]
    cs.add_expectation(
        Expectation(
            id="exp-e3cm",
            target="e03",
            prediction="reaches comma",
            expected_time=now + timedelta(hours=1, minutes=20),
            uncertainty="±30 min",
            basis="715 min baseline",
            session_id=s5,
        )
    )
    met = Expectation(
        id="exp-e2ht",
        target="e02",
        prediction="hatches",
        expected_time=now - timedelta(days=20, hours=1),
        basis="baseline",
        session_id=s3,
    )
    cs.add_expectation(met)
    cs.resolve_expectation(
        met.id,
        __import__(
            "gently.harness.memory.model", fromlist=["ExpectationStatus"]
        ).ExpectationStatus.CONFIRMED,
    )
    missed = Expectation(
        id="exp-e5ht",
        target="e05",
        prediction="hatches",
        expected_time=now - timedelta(days=20),
        basis="baseline",
        session_id=s3,
    )
    cs.add_expectation(missed)
    cs.resolve_expectation(
        missed.id,
        __import__(
            "gently.harness.memory.model", fromlist=["ExpectationStatus"]
        ).ExpectationStatus.SURPRISED,
    )
    cs.add_question(
        Question(id="q-561", content="Does 561 add phototoxicity at 60 s cadence?", project_id=p1)
    )
    cs.add_question(
        Question(
            id="q-e5", content="Why did e05 arrest at 2-fold on the third night?", session_id=s3
        )
    )
    answered = Question(id="q-led", content="Is 1 % LED enough for the overview?", project_id=p1)
    cs.add_question(answered)
    cs.resolve_question(answered.id, "Yes, since the dark/flat references.")

    n1 = nb.write_note(
        Note(
            id="",
            kind=NoteKind.OBSERVATION,
            author=Author.HUMAN,
            body="Both 25 °C embryos hatched well before the baseline: 588 and 601 min.",
            sessions=[s5],
            projects=[p1],
            strains=["OH10689"],
        )
    )
    n2 = nb.write_note(
        Note(
            id="",
            kind=NoteKind.OBSERVATION,
            author=Author.AGENT,
            body="The 25 °C run used 561 as well as 488; the baseline used 488 only.",
            sessions=[s5],
            projects=[p1],
        )
    )
    nb.write_note(
        Note(
            id="",
            kind=NoteKind.FINDING,
            author=Author.AGENT,
            status=NoteStatus.PROPOSED,
            body=(
                "At 25 °C the twitching-to-hatching interval shortens by about 18 % "
                "on two embryos; too few to call."
            ),
            basis=[n1, n2],
            sessions=[s5],
            projects=[p1],
        )
    )
    nb.write_note(
        Note(
            id="",
            kind=NoteKind.FINDING,
            author=Author.HUMAN,
            status=NoteStatus.CONFIRMED,
            body="The 20 °C baseline interval is 715 ± 22 min over three nights.",
            sessions=[sids["baseline, second night"], sids["repeat at 20 °C"]],
            projects=[p1],
        )
    )

    # ---- project two, and a session in no project --------------------------
    p2 = cs.create_campaign(
        description="Lineage reporters: which markers survive 60 s cadence?", shorthand="lineage"
    )
    i_pilot = cs.create_plan_item(
        campaign_id=p2,
        type="imaging",
        title="lin-12 reporter pilot",
        spec=dict(base_spec, strain="JK2868", interval_s=120),
        origin="agent",
    )
    sid = demo.seed(store, n_embryos=4, n_timepoints=timepoints, seed_val=200)
    store.update_session(sid, name="lin-12 reporter, pilot")
    started = now - timedelta(days=24, hours=2)
    _checkpoint(
        store,
        sid,
        status="completed",
        started=started,
        interval=120,
        timepoints=timepoints,
        going=0,
    )
    _events(store, sid, started, store.list_embryo_ids(sid), errors=0, warnings=0, ended="complete")
    _interactions(store, sid, 8, 0)
    from gently.app.filing import file_session_under

    file_session_under(store, cs, sid, p2, i_pilot)
    # the same session also served the first project's quick look (one home, many items)
    cs.link_plan_item_session(i22, sid)

    sid = demo.seed(store, n_embryos=3, n_timepoints=max(4, timepoints // 3), seed_val=300)
    store.update_session(sid, name="unnamed, a Tuesday")
    started = now - timedelta(days=28, hours=5)
    _checkpoint(
        store,
        sid,
        status="running",
        started=started,
        interval=90,
        timepoints=max(4, timepoints // 3),
        going=3,
    )
    _events(store, sid, started, store.list_embryo_ids(sid), errors=0, warnings=1, ended="cut")
    _interactions(store, sid, 4, 0)

    print(f"projects: {p1} (hatching-temp), {p2} (lineage); sessions: {len(store.list_sessions())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--timepoints", type=int, default=12)
    args = ap.parse_args()
    root = Path(settings.storage.base_path)
    root.mkdir(parents=True, exist_ok=True)
    seed(root, timepoints=args.timepoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
