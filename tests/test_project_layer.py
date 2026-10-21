"""The project layer: a session has one home project, the project holds the
ledger, references and its sessions, and one search answers in one shape.

Each test is the smallest thing that fails if the logic breaks.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.filing import backfill_session_projects, file_session_under, unfile_session
from gently.harness.memory.model import Expectation, PlanItemStatus, Question
from gently.harness.memory.notebook import Note, NoteKind, NoteStatus
from gently.ui.web.auth import require_control

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _project(cs, desc="Hatching timing under temperature stress", shorthand="hatching-temp"):
    root = cs.create_campaign(description=desc, shorthand=shorthand)
    phase = cs.create_campaign(description="Baseline at 20 °C", parent_id=root)
    return root, phase


def _app(file_store, cs, agent=None, claude=None):
    from gently.ui.web.routes.campaigns import create_router as campaigns
    from gently.ui.web.routes.context import create_router as context
    from gently.ui.web.routes.notebook import create_router as notebook
    from gently.ui.web.routes.sessions import create_router as sessions

    class _Bridge:
        def __init__(self, a):
            self.agent = a

    class _Server:
        context_store = cs
        gently_store = file_store
        agent_bridge = _Bridge(agent) if agent is not None else None
        claude_async = claude

    app = FastAPI()
    server = _Server()
    for factory in (campaigns, sessions, context, notebook):
        app.include_router(factory(server))
    app.dependency_overrides[require_control] = lambda: None
    return TestClient(app)


# ---------------------------------------------------------------------------
# sessions belong to a project
# ---------------------------------------------------------------------------


def test_a_session_has_one_home_and_many_items(file_store, file_context_store):
    cs = file_context_store
    root, phase = _project(cs)
    other, _ = _project(cs, "Lineage reporters", "lineage")
    item = cs.create_plan_item(campaign_id=phase, type="imaging", title="six embryos")
    other_item = cs.create_plan_item(campaign_id=other, type="imaging", title="pilot")
    file_store.create_session("s1", name="night one")

    first = file_session_under(file_store, cs, "s1", phase, item)
    assert first["home_set"] and first["project_id"] == root
    assert file_store.get_session("s1")["project_id"] == root
    assert "s1" in cs.get_plan_item(item).session_ids

    # a plan item from another project links the session but does not re-home it
    second = file_session_under(file_store, cs, "s1", other, other_item)
    assert second["home_kept"] == root and not second["home_set"]
    assert file_store.get_session("s1")["project_id"] == root
    assert "s1" in cs.get_plan_item(other_item).session_ids

    # unless the person moves it
    moved = file_session_under(file_store, cs, "s1", other, move=True)
    assert moved["home_set"] and file_store.get_session("s1")["project_id"] == other

    assert unfile_session(file_store, cs, "s1")
    assert file_store.get_session("s1")["project_id"] is None


def test_list_sessions_filters_by_project_and_unfiled(file_store, file_context_store):
    cs = file_context_store
    root, _ = _project(cs)
    for sid in ("a", "b", "c"):
        file_store.create_session(sid)
    file_store.set_session_project("a", root)
    file_store.set_session_project("b", root)
    assert {s["session_id"] for s in file_store.list_sessions(project_id=root)} == {"a", "b"}
    assert {s["session_id"] for s in file_store.list_sessions(unfiled=True)} == {"c"}
    assert len(file_store.list_sessions()) == 3


def test_backfill_gives_linked_sessions_their_home(file_store, file_context_store):
    cs = file_context_store
    root, phase = _project(cs)
    file_store.create_session("old")
    file_store.create_session("lonely")
    cs.link_session_campaign("old", phase)  # the pre-project way of linking
    assert backfill_session_projects(file_store, cs) == 1
    assert file_store.get_session("old")["project_id"] == root
    assert file_store.get_session("lonely")["project_id"] is None
    assert backfill_session_projects(file_store, cs) == 0  # idempotent


# ---------------------------------------------------------------------------
# the plan: proposed items are not runnable
# ---------------------------------------------------------------------------


def test_a_proposed_item_is_not_next_until_accepted(file_context_store):
    cs = file_context_store
    root, _ = _project(cs)
    planned = cs.create_plan_item(campaign_id=root, type="imaging", title="planned one")
    suggested = cs.create_plan_item(
        campaign_id=root, type="imaging", title="a suggestion", status="proposed"
    )
    assert cs.get_plan_item(suggested).status is PlanItemStatus.PROPOSED
    assert cs.get_plan_item(suggested).origin == "agent"
    assert {i.id for i in cs.get_unblocked_plan_items(root)} == {planned}
    assert cs.get_plan_status(root)["proposed"] == 1
    cs.update_plan_item(suggested, status=PlanItemStatus.PLANNED)
    assert {i.id for i in cs.get_unblocked_plan_items(root)} == {planned, suggested}


# ---------------------------------------------------------------------------
# references: one shape, kept once
# ---------------------------------------------------------------------------


def test_references_and_strains_are_kept_once_on_the_root(file_context_store):
    cs = file_context_store
    root, phase = _project(cs)
    ref = {
        "source": "alliance",
        "key": "WBGene00004267",
        "title": "rab-3",
        "url": "https://wormbase.org/x",
    }
    cs.add_campaign_reference(root, ref)
    cs.add_campaign_reference(root, ref)  # a repeat is not a second entry
    cs.add_campaign_reference(
        root, {"source": "cgc", "key": "OH10689", "subtitle": "otIs355"}, as_strain=True
    )
    c = cs.get_campaign(root)
    assert [r["key"] for r in c.references] == ["WBGene00004267"]
    assert [s["key"] for s in c.strains] == ["OH10689"]
    assert cs.root_campaign_id(phase) == root
    assert cs.remove_campaign_reference(root, "cgc", "OH10689")
    assert cs.get_campaign(root).strains == []


# ---------------------------------------------------------------------------
# the notebook knows projects
# ---------------------------------------------------------------------------


def test_notes_are_found_by_project(file_context_store):
    nb = file_context_store.notebook
    nb.write_note(
        Note(id="", kind=NoteKind.FINDING, body="in P", projects=["P"], status=NoteStatus.PROPOSED)
    )
    nb.write_note(Note(id="", kind=NoteKind.FINDING, body="in Q", projects=["Q"]))
    assert [n.body for n in nb.query_notes(project="P")] == ["in P"]
    assert [n.body for n in nb.query_notes(kind=NoteKind.FINDING, status=NoteStatus.PROPOSED)] == [
        "in P"
    ]


# ---------------------------------------------------------------------------
# the ledger over HTTP
# ---------------------------------------------------------------------------


def test_ledger_scopes_to_the_project_through_its_sessions(file_store, file_context_store):
    cs = file_context_store
    root, phase = _project(cs)
    other, _ = _project(cs, "Lineage reporters", "lineage")
    done = cs.create_plan_item(campaign_id=phase, type="imaging", title="baseline")
    cs.complete_plan_item(done, "715 ± 22 min")
    nxt = cs.create_plan_item(campaign_id=phase, type="imaging", title="shift to 25 °C")
    cs.create_plan_item(campaign_id=phase, type="imaging", title="a suggestion", status="proposed")
    file_store.create_session("s1")
    file_session_under(file_store, cs, "s1", phase, done)
    file_store.create_session("elsewhere")
    file_store.set_session_project("elsewhere", other)
    soon = datetime.now() + timedelta(hours=1)
    cs.add_expectation(
        Expectation(id="e1", target="e3", prediction="comma", expected_time=soon, session_id="s1")
    )
    cs.add_expectation(
        Expectation(
            id="e2", target="e9", prediction="hatch", expected_time=soon, session_id="elsewhere"
        )
    )
    cs.add_question(Question(id="q1", content="why did e5 arrest?", project_id=root))
    cs.add_question(Question(id="q2", content="not ours", session_id="elsewhere"))
    cs.notebook.write_note(
        Note(id="", kind=NoteKind.FINDING, body="ours", projects=[root], status=NoteStatus.PROPOSED)
    )
    cs.notebook.write_note(Note(id="", kind=NoteKind.FINDING, body="via session", sessions=["s1"]))
    cs.notebook.write_note(Note(id="", kind=NoteKind.FINDING, body="theirs", projects=[other]))

    client = _app(file_store, cs)
    led = client.get(f"/api/campaigns/{root}/ledger").json()
    assert led["project_id"] == root and led["members"] == ["s1"]
    assert [i["id"] for i in led["next"]] == [nxt]
    assert [i["title"] for i in led["proposed"]] == ["a suggestion"]
    assert [i["outcome"] for i in led["measured"]["outcomes"]] == ["715 ± 22 min"]
    assert {f["body"] for f in led["measured"]["findings"]} == {"ours", "via session"}
    assert [e["id"] for e in led["predicted"]] == ["e1"]
    assert [q["id"] for q in led["open"]] == ["q1"]

    # the project list carries what Home shows
    roots = client.get("/api/campaigns").json()["campaigns"]
    ours = next(t for t in roots if t["campaign"]["id"] == root)
    assert ours["project"]["sessions"] == 1
    assert ours["project"]["proposed_items"] == 1
    assert ours["project"]["proposed_findings"] == 1


def test_the_writes_a_person_makes(file_store, file_context_store):
    cs = file_context_store
    client = _app(file_store, cs)
    made = client.post(
        "/api/campaigns", json={"description": "Lineage reporters", "shorthand": "lineage"}
    ).json()
    root = made["campaign"]["id"]
    assert made["ok"] and made["campaign"]["parent_id"] is None

    item = client.post(
        f"/api/campaigns/{root}/items", json={"type": "imaging", "title": "pilot"}
    ).json()["item"]
    assert item["origin"] == "manual" and item["status"] == "planned"
    suggestion = client.post(
        f"/api/campaigns/{root}/items",
        json={"type": "imaging", "title": "idea", "status": "proposed", "origin": "agent"},
    ).json()["item"]
    accepted = client.post(
        f"/api/campaigns/{root}/items/{suggestion['id']}/status", json={"status": "planned"}
    ).json()
    assert accepted.get("ok", True)
    assert cs.get_plan_item(suggestion["id"]).status is PlanItemStatus.PLANNED
    assert client.delete(f"/api/campaigns/{root}/items/{item['id']}").json()["deleted"]

    kept = client.post(
        f"/api/campaigns/{root}/references",
        json={"source": "cgc", "key": "OH10689", "subtitle": "otIs355", "files_as": "strain"},
    ).json()
    assert kept["as_strain"] and cs.get_campaign(root).strains[0]["key"] == "OH10689"
    assert client.delete(
        f"/api/campaigns/{root}/references", params={"source": "cgc", "key": "OH10689"}
    ).json()["removed"]

    file_store.create_session("s9")
    filed = client.post("/api/sessions/s9/project", json={"campaign_id": root}).json()
    assert filed["home_set"] and file_store.get_session("s9")["project_id"] == root
    rows = client.get("/api/sessions", params={"project_id": root}).json()["sessions"]
    assert [r["session_id"] for r in rows] == ["s9"]
    assert rows[0]["project_id"] == root and "errors" in rows[0] and "turns" in rows[0]
    assert client.post("/api/sessions/s9/project", json={"campaign_id": None}).json()["unfiled"]

    q = client.post(
        "/api/context/questions", json={"content": "keep 561?", "project_id": root}
    ).json()
    assert q["ok"]
    e = client.post(
        "/api/context/expectations",
        json={
            "target": "e1",
            "prediction": "hatches",
            "expected_time": datetime.now().isoformat(),
            "project_id": root,
        },
    ).json()
    assert e["ok"]
    ctx = client.get("/api/context", params={"project_id": root}).json()
    assert len(ctx["questions"]) == 1 and len(ctx["expectations"]) == 1

    note = client.post(
        "/api/notebook/notes",
        json={"kind": "finding", "body": "the interval shortens", "projects": [root]},
    ).json()["note"]
    assert note["status"] == "confirmed" and note["projects"] == [root]
    settled = client.post(
        f"/api/notebook/notes/{note['id']}/status", json={"status": "superseded"}
    ).json()["note"]
    assert settled["status"] == "superseded"
    assert (
        client.get("/api/notebook/notes", params={"project": root}).json()["notes"][0]["id"]
        == note["id"]
    )


# ---------------------------------------------------------------------------
# one search, one shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_answers_in_one_shape_and_a_down_source_is_a_status(
    monkeypatch, file_context_store
):
    from gently.harness.plan_mode.tools import research

    cs = file_context_store
    root, _ = _project(cs)
    cs.add_campaign_reference(
        root, {"source": "cgc", "key": "OH10689", "subtitle": "otIs355 rab-3p"}, as_strain=True
    )

    async def boom(*a, **k):
        raise RuntimeError("HTTP 503")

    monkeypatch.setattr(research, "_alliance_autocomplete", boom)
    lab = await research.search("lab", "rab-3", cs=cs)
    assert lab["status"] == "ok" and lab["results"][0]["key"] == "OH10689"
    assert lab["results"][0]["files_as"] == "strain"
    assert set(lab["results"][0]) >= {
        "source",
        "kind",
        "key",
        "title",
        "subtitle",
        "url",
        "files_as",
    }
    genes = await research.search("genes", "rab-3")
    assert genes["status"] == "error" and "503" in genes["error"] and genes["results"] == []
    nowhere = await research.search("nowhere", "x")
    assert nowhere["status"] == "error" and "unknown source" in nowhere["error"]
    none = await research.search("lab", "zzz-not-here", cs=cs)
    assert none["status"] == "empty"


def test_cgc_rows_parse_the_absolute_links():
    from gently.harness.plan_mode.tools.research import _parse_cgc_rows

    html = """
    <table><tr><th>Strain</th><th>Species</th><th>Genotype</th></tr>
    <tr><td><a href="https://cgc.umn.edu/strain/NM791">NM791</a></td><td>C. elegans</td>
        <td>rab-3(js49) II. <a>Show Description</a> Ric; very slightly slow.</td></tr>
    <tr><td><a href="https://cgc.umn.edu/strain/NM791">NM791</a></td><td>dup</td><td>x</td></tr>
    </table>"""
    rows = _parse_cgc_rows(html)
    assert [r["name"] for r in rows] == ["NM791"]
    assert rows[0]["genotype"] == "rab-3(js49) II" and rows[0]["description"].startswith("Ric")


# ---------------------------------------------------------------------------
# stages carry an ontology id; a night leaves a note
# ---------------------------------------------------------------------------


def test_every_scored_stage_has_a_wormbase_term():
    from gently.organisms.celegans.stages import WBLS, DevelopmentalStage, ontology_id

    for stage in ("bean", "comma", "1.5fold", "2fold", "pretzel", "hatched"):
        assert WBLS[stage].startswith("WBls:")
    assert ontology_id(DevelopmentalStage.COMMA.value) == "WBls:0000017"
    assert ontology_id("hatching") is None and ontology_id(None) is None


def test_session_end_text_says_what_happened_and_nothing_for_an_empty_night(tmp_path):
    from gently.core.session_summary import interaction_stats, session_end_text, signal_counts

    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "event_type": "ACQUISITION_STARTED",
                    "data": {"embryo_ids": ["e1", "e2"]},
                    "timestamp": "2026-10-20T21:04:00",
                },
                {
                    "event_type": "WARNING_ISSUED",
                    "data": {"message": "561 nm power 4 % above plan"},
                    "timestamp": "2026-10-20T21:05:00",
                },
                {
                    "event_type": "ERROR_OCCURRED",
                    "data": {"message": "stage timeout"},
                    "timestamp": "2026-10-21T03:12:00",
                },
                "not json at all",
            ]
            if isinstance(r, dict)
        )
        + "\nnot json\n",
        encoding="utf-8",
    )
    log = tmp_path / "interaction_log.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(r) for r in [{"was_corrected": True, "tool_calls": [{}, {}]}, {"error": "x"}]
        ),
        encoding="utf-8",
    )
    assert signal_counts(events) == {"errors": 1, "warnings": 1}
    assert interaction_stats(log) == {
        "turns": 2,
        "corrections": 1,
        "tool_calls": 2,
        "agent_errors": 1,
    }
    text = session_end_text(
        {
            "held": {
                "embryo_count": 2,
                "timepoints": 12,
                "last_run": {"status": "completed", "interval_seconds": 30},
            },
            "events": [
                {"level": "error", "at": "2026-10-21T03:12:00", "text": "Error: stage timeout"}
            ],
            "signals": {"errors": 1, "warnings": 1},
            "agent": {"turns": 2, "corrections": 1},
        }
    )
    assert "2 embryos, 12 timepoints every 30 s." in text
    assert "1 error and 1 warning" in text and "03:12 Error: stage timeout." in text
    assert "2 agent turns, 1 correction." in text
    assert session_end_text({"held": {}, "events": [], "signals": {}, "agent": {}}) == ""


# ---------------------------------------------------------------------------
# an intention becomes a spec
# ---------------------------------------------------------------------------


def test_a_sentence_states_its_spec_fields_with_provenance():
    from gently.harness.memory.intent_spec import merge_spec, parse_intent

    read = parse_intent(
        "OH10689, six embryos every 60 s, 488+561, until hatching, at 25 °C, 80 slices, 10 ms"
    )
    assert read["spec"] == {
        "strain": "OH10689",
        "num_embryos": 6,
        "interval_s": 60,
        "laser_wavelength_nm": 488,
        "stop_condition": "hatching",
        "temperature_c": 25.0,
        "num_slices": 80,
        "exposure_ms": 10.0,
    }
    assert read["provenance"]["interval_s"] == {"source": "sentence", "text": "every 60 s"}
    assert read["provenance"]["laser_wavelength_nm"]["text"] == "488 + 561"

    intent = parse_intent(
        "i want to image an exposure ramp to check at what exposure the led saturates"
    )
    assert intent["spec"] == {"success_criteria": "at what exposure the led saturates"}

    cadence = parse_intent("quick look at comma, 12 timepoints at 30 s cadence for 2 h")
    assert cadence["spec"]["stop_condition"] == "comma"
    assert cadence["spec"]["interval_s"] == 30 and cadence["spec"]["estimated_duration_h"] == 2.0

    # what a person typed is kept; what the sentence read fills the gaps
    merged = merge_spec({"num_embryos": 4, "provenance": {}}, read)
    assert merged["num_embryos"] == 4 and merged["interval_s"] == 60
    assert (
        merged["provenance"]["interval_s"]["source"] == "sentence"
        and "num_embryos" not in merged["provenance"]
    )


def test_an_item_written_as_a_sentence_arrives_with_its_spec(file_store, file_context_store):
    cs = file_context_store
    client = _app(file_store, cs)
    root = client.post("/api/campaigns", json={"description": "Light dose"}).json()["campaign"][
        "id"
    ]
    item = client.post(
        f"/api/campaigns/{root}/items",
        json={"type": "imaging", "title": "Two embryos every 30 s with 488 until comma"},
    ).json()["item"]
    spec = item["imaging_spec"]
    assert (
        spec["num_embryos"] == 2 and spec["interval_s"] == 30 and spec["stop_condition"] == "comma"
    )
    assert spec["provenance"]["interval_s"]["source"] == "sentence"

    bare = client.post(
        f"/api/campaigns/{root}/items",
        json={"type": "imaging", "title": "exposure ramp to find where the LED saturates"},
    ).json()["item"]
    assert bare["imaging_spec"]["success_criteria"] == "where the LED saturates"
    again = client.post(
        f"/api/campaigns/{root}/items/{bare['id']}/draft", json={"model": False}
    ).json()
    assert again["ok"] and again["from_sentence"] == ["success_criteria"]
    assert again["status"] == "drafted"
    assert again["item"]["imaging_spec"]["draft"]["status"] == "drafted"
    assert client.get("/api/plan/intent", params={"text": "six embryos at 25 °C"}).json()[
        "spec"
    ] == {
        "num_embryos": 6,
        "temperature_c": 25.0,
    }


class _FakeClaude:
    """Answers the draft tool call the way the model would, from a canned draft."""

    def __init__(self, draft: dict):
        self.draft = draft
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="tool_use", input=dict(self.draft))
        return SimpleNamespace(content=[block])


def test_gently_drafts_the_spec_and_a_person_accepts_it(file_store, file_context_store):
    cs = file_context_store
    fake = _FakeClaude(
        {
            "title": "LED exposure ramp",
            "item_type": "imaging",
            "rationale": "A one-off brightfield sweep finds the exposure where 16-bit pixels clip.",
            "num_embryos": 1,
            "acquisition": {
                "cadence_s": 30,
                "volumes": False,
                "bf": {
                    "enabled": True,
                    "every_seconds": 30,
                    "exposure_ms": 1,
                    "light": "led",
                    "led_intensity_pct": 1,
                },
                "stop_condition": "timepoints:8",
            },
            "success_criteria": "the exposure at which brightfield pixels reach 65535",
            "tactics": [
                {
                    "kind": "scripted_protocol",
                    "name": "exposure sweep",
                    "scope": "brightfield",
                    "structure": {"steps": [{"exposure_ms": e} for e in (1, 2, 5, 10, 20, 50)]},
                }
            ],
            "assumptions": ["LED at the default 1 %", "an empty field is as good as an embryo"],
        }
    )
    client = _app(file_store, cs, claude=fake)
    root = client.post("/api/campaigns", json={"description": "Light dose"}).json()["campaign"][
        "id"
    ]
    sentence = "i want to image an exposure ramp to check at what exposure the led saturates"
    made = client.post(
        f"/api/campaigns/{root}/items", json={"type": "imaging", "title": sentence}
    ).json()["item"]
    # the sentence was read at once; the draft itself runs in the background
    assert made["imaging_spec"]["success_criteria"] == "at what exposure the led saturates"
    assert made["imaging_spec"]["draft"]["status"] in ("drafting", "drafted")

    drafted = client.post(f"/api/campaigns/{root}/items/{made['id']}/draft", json={}).json()
    assert drafted["status"] == "drafted" and fake.calls, "the model was asked"
    item = drafted["item"]
    assert item["title"] == "LED exposure ramp" and item["description"] == sentence
    spec = item["imaging_spec"]
    # the sentence outranks the draft; the draft fills the rest, with provenance
    assert spec["success_criteria"] == "at what exposure the led saturates"
    assert spec["provenance"]["success_criteria"]["source"] == "sentence"
    assert spec["num_embryos"] == 1 and spec["provenance"]["num_embryos"]["source"] == "model"
    assert (
        spec["acquisition"]["volumes"] is False
        and spec["acquisition"]["stop_condition"] == "timepoints:8"
    )
    assert (
        spec["acquisition_derived"] is False
        and spec["provenance"]["acquisition"]["source"] == "model"
    )
    assert spec["tactics"][0]["kind"] == "scripted_protocol"
    assert len(spec["tactics"][0]["structure"]["steps"]) == 6
    assert spec["draft"]["status"] == "drafted"
    assert "LED at the default 1 %" in spec["draft"]["assumptions"]
    assert "Instrument:" in fake.calls[-1]["messages"][0]["content"]

    # a typed field is the person's and a redraft does not touch it
    client.patch(f"/api/campaigns/{root}/items/{made['id']}", json={"spec": {"num_embryos": 2}})
    redrawn = client.post(
        f"/api/campaigns/{root}/items/{made['id']}/draft", json={"note": "use two embryos"}
    ).json()
    assert redrawn["item"]["imaging_spec"]["num_embryos"] == 2
    assert redrawn["item"]["imaging_spec"]["provenance"]["num_embryos"]["source"] == "typed"

    accepted = client.post(f"/api/campaigns/{root}/items/{made['id']}/spec/accept").json()
    acc = accepted["item"]["imaging_spec"]
    assert acc["draft"]["status"] == "accepted"
    assert acc["provenance"]["acquisition"]["source"] == "accepted"


# ---------------------------------------------------------------------------
# the acquisition is the pane's shape, everywhere
# ---------------------------------------------------------------------------


def test_the_acquisition_is_one_shape_in_the_pane_s_words():
    from gently.harness.memory.acquisition import (
        acquisition_from_legacy,
        describe_acquisition,
        effective_acquisition,
        stop_spec_from_words,
        structure_from_payload,
    )

    legacy = {
        "interval_s": 60,
        "num_slices": 80,
        "exposure_ms": 10.0,
        "laser_wavelength_nm": 488,
        "laser_power_pct": 3.0,
        "stop_condition": "hatching",
    }
    acq = acquisition_from_legacy(legacy)
    assert acq == {
        "cadence_s": 60,
        "interval": 60,
        "num_slices": 80,
        "exposure_ms": 10.0,
        "laser_powers": {"488": 3.0},
        "stop_condition": "hatching",
    }
    assert stop_spec_from_words("12 timepoints") == "timepoints:12"
    assert stop_spec_from_words("after 6 h") == "duration:6h"
    assert stop_spec_from_words("pretzel") == "stages(pretzel)"
    assert stop_spec_from_words("single pass") == "timepoints:1"
    assert stop_spec_from_words("whenever") is None

    # an explicit acquisition wins over the flat fields
    merged = effective_acquisition({**legacy, "acquisition": {"cadence_s": 90, "volumes": True}})
    assert (
        merged["cadence_s"] == 90
        and merged["interval"] == 90
        and merged["laser_powers"] == {"488": 3.0}
    )

    said = describe_acquisition(acq, 6)
    assert (
        said
        == "Every 1 min: SPIM volumes (80 slices · 10 ms · 488 at 3 %) of 6 embryos · at hatching."
    )
    bf_only = describe_acquisition(
        {
            "cadence_s": 30,
            "volumes": False,
            "bf": {"enabled": True, "exposure_ms": 1, "light": "led", "led_intensity_pct": 1},
            "stop_condition": "timepoints:8",
        }
    )
    assert bf_only.startswith(
        "Every 30 s: one brightfield frame of the field on the bottom camera (1 ms), "
        "under the LED at 1 %"
    )
    assert bf_only.endswith("after 8 frames. No SPIM volumes.")

    kept = structure_from_payload(
        {
            "interval_seconds": 90,
            "stop_condition": "duration:16h",
            "num_slices": 50,
            "exposure_ms": 10,
            "laser_config": "488+561",
            "laser_powers": {488: 4, "561": 10},
            "bf": {"enabled": True, "every_seconds": 90, "light": "led"},
        }
    )
    assert (
        kept["cadence_s"] == 90
        and kept["laser_powers"] == {"488": 4, "561": 10}
        and kept["volumes"] is True
    )
    dark = structure_from_payload(
        {
            "interval_seconds": 30,
            "volumes": False,
            "stop_condition": "timepoints:8",
            "num_slices": 50,
        }
    )
    assert dark["volumes"] is False and dark["num_slices"] is None and dark["laser_config"] is None


def test_an_item_goes_out_with_the_acquisition_it_means(file_store, file_context_store):
    cs = file_context_store
    client = _app(file_store, cs)
    root = client.post("/api/campaigns", json={"description": "Light dose"}).json()["campaign"][
        "id"
    ]
    item = client.post(
        f"/api/campaigns/{root}/items",
        json={
            "type": "imaging",
            "title": "Two embryos every 30 s with 488 at 3 % until comma",
            "draft": False,
        },
    ).json()["item"]
    spec = item["imaging_spec"]
    # written in the flat fields by the sentence reader, read out in the pane's shape
    assert spec["acquisition"] == {
        "cadence_s": 30,
        "interval": 30,
        "laser_powers": {"488": 3.0},
        "stop_condition": "comma",
    }
    assert spec["acquisition_derived"] is True
    led = client.get(f"/api/campaigns/{root}/ledger").json()
    assert led["next"][0]["imaging_spec"]["acquisition"]["stop_condition"] == "comma"
