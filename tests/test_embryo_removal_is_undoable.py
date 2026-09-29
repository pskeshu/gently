"""Removing an embryo deletes nothing, and can be undone.

"and what if i accidentally delete a embryo from the embryo list? is there an
undo button?"

There was not. The x beside an embryo is for a false positive, and it sits
one row from the embryo that has been imaged all night. It asked nothing, and
it deleted the embryo's folder: volumes, projections, traces, calibration.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.agent import MicroscopyAgent
from gently.core.file_store import FileStore
from gently.harness.state import ExperimentState
from gently.ui.web import auth
from gently.ui.web.routes.data import create_router

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
ROSTER = (WEB / "static" / "js" / "panels" / "roster.js").read_text(encoding="utf-8")
DEVICES = (WEB / "static" / "js" / "devices.js").read_text(encoding="utf-8")
STORE_SRC = (ROOT / "gently" / "core" / "file_store.py").read_text(encoding="utf-8")

FIT = {"slope_um_per_deg": 101.5, "offset_um": 3.0}


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path)
    fs.create_session("s1")
    for n, fit in ((1, FIT), (2, None), (3, FIT)):
        fs.register_embryo(
            "s1",
            f"embryo_{n}",
            position_coarse={"x": 100.0 * n, "y": -50.0},
            calibration=fit,
            role="test",
        )
    for tp in range(3):
        fs.put_volume("s1", "embryo_1", tp, np.full((2, 4, 4), tp + 1, dtype=np.uint16))
    return fs


def _embryo_dir(store, embryo_id="embryo_1") -> Path:
    return store._embryo_dir("s1", embryo_id)


def _removed_dir(store) -> Path:
    return store._session_dir("s1") / "removed"


def _files(folder: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(folder)).replace("\\", "/"): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# ── the store ────────────────────────────────────────────────────────────


class TestTheStore:
    def test_nothing_is_deleted(self, store):
        before = _files(_embryo_dir(store))
        assert any(k.startswith("volumes/") for k in before)
        record = store.set_aside_embryo("s1", "embryo_1")
        assert not _embryo_dir(store).exists()
        after = _files(_removed_dir(store) / record["folder"])
        after.pop("removed.yaml")
        assert after == before

    def test_the_record_says_what_it_held(self, store):
        record = store.set_aside_embryo("s1", "embryo_1", by={"by": "ryan", "client": "10.0.0.2"})
        assert record["embryo_id"] == "embryo_1"
        assert record["timepoints"] == 3
        assert record["calibrated"] is True
        assert record["removed_by"] == {"by": "ryan", "client": "10.0.0.2"}
        on_disk = yaml.safe_load(
            (_removed_dir(store) / record["folder"] / "removed.yaml").read_text("utf-8")
        )
        assert on_disk["timepoints"] == 3 and on_disk["embryo_id"] == "embryo_1"

    def test_it_is_out_of_every_listing(self, store):
        store.set_aside_embryo("s1", "embryo_1")
        assert "embryo_1" not in store.list_embryo_ids("s1")
        assert "embryo_1" not in [e["embryo_id"] for e in store.list_embryos("s1")]

    def test_the_old_name_deletes_nothing_either(self, store):
        before = _files(_embryo_dir(store))
        assert store.delete_embryo("s1", "embryo_1") is True
        kept = store.list_removed_embryos("s1")
        assert [r["embryo_id"] for r in kept] == ["embryo_1"]
        after = _files(_removed_dir(store) / kept[0]["folder"])
        after.pop("removed.yaml")
        assert after == before

    def test_no_rmtree_on_an_embryo(self):
        body = STORE_SRC[STORE_SRC.index("    def set_aside_embryo(") :]
        body = body[: body.index("    # ====")]
        assert "rmtree" not in body

    def test_an_embryo_without_a_folder_is_none(self, store):
        assert store.set_aside_embryo("s1", "embryo_9") is None
        assert store.delete_embryo("s1", "embryo_9") is False

    def test_it_comes_back_whole(self, store):
        before = _files(_embryo_dir(store))
        store.set_aside_embryo("s1", "embryo_1")
        record = store.restore_embryo("s1", "embryo_1")
        assert record["embryo_id"] == "embryo_1"
        assert _files(_embryo_dir(store)) == before
        assert store.list_removed_embryos("s1") == []
        assert "embryo_1" in store.list_embryo_ids("s1")

    def test_restore_does_not_overwrite_an_embryo_of_that_name(self, store):
        store.set_aside_embryo("s1", "embryo_2")
        store.register_embryo("s1", "embryo_2", position_coarse={"x": 1.0, "y": 2.0}, role="test")
        with pytest.raises(FileExistsError):
            store.restore_embryo("s1", "embryo_2")
        assert [r["embryo_id"] for r in store.list_removed_embryos("s1")] == ["embryo_2"]

    def test_the_same_name_removed_twice_keeps_both(self, store):
        store.set_aside_embryo("s1", "embryo_2")
        store.register_embryo("s1", "embryo_2", position_coarse={"x": 1.0, "y": 2.0}, role="test")
        store.set_aside_embryo("s1", "embryo_2")
        kept = store.list_removed_embryos("s1")
        assert len(kept) == 2 and len({r["folder"] for r in kept}) == 2

    def test_nothing_removed_nothing_to_restore(self, store):
        assert store.restore_embryo("s1", "embryo_1") is None
        assert store.list_removed_embryos("s1") == []

    def test_a_folder_in_use_is_left_where_it_is(self, store, monkeypatch):
        import gently.core.file_store as fs_mod

        def refuse(src, dst):
            raise PermissionError(13, "The process cannot access the file")

        monkeypatch.setattr(fs_mod.os, "rename", refuse)
        with pytest.raises(OSError):
            store.set_aside_embryo("s1", "embryo_1")
        assert _embryo_dir(store).exists()


# ── the routes ───────────────────────────────────────────────────────────


def _agent(store, run=None):
    agent = MagicMock()
    agent.session_id = "s1"
    agent.store = store
    agent.experiment = ExperimentState()
    for row in store.list_embryos("s1"):
        agent.experiment.add_embryo(
            embryo_id=row["embryo_id"],
            position=row.get("position_coarse") or {},
            calibration=row.get("calibration") or {},
            role=row.get("role") or "test",
        )
    agent.import_embryos_from_session = MicroscopyAgent.import_embryos_from_session.__get__(agent)
    agent._compute_imported_dose = MicroscopyAgent._compute_imported_dose.__get__(agent)
    orch = MagicMock()
    orch._status = SimpleNamespace(value="idle")
    orch._embryo_states = {}
    if run:
        orch._status = SimpleNamespace(value=run["status"])
        orch._embryo_states = {
            k: SimpleNamespace(is_complete=done) for k, done in run["embryos"].items()
        }
    agent.timelapse_orchestrator = orch
    return agent


def _client(agent):
    server = MagicMock()
    server.agent_bridge.agent = agent
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


class TestTheRoutes:
    def test_removing_keeps_the_files_and_says_so(self, store):
        agent = _agent(store)
        before = _files(_embryo_dir(store))
        r = _client(agent).delete("/api/embryos/embryo_1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["restorable"] is True and body["kept"]["timepoints"] == 3
        assert "embryo_1" not in agent.experiment.embryos
        after = _files(_removed_dir(store) / body["kept"]["folder"])
        after.pop("removed.yaml")
        assert after == before

    def test_the_removed_are_listed(self, store):
        agent = _agent(store)
        client = _client(agent)
        client.delete("/api/embryos/embryo_1")
        rows = client.get("/api/embryos/removed").json()["removed"]
        assert [(r["embryo_id"], r["timepoints"], r["calibrated"]) for r in rows] == [
            ("embryo_1", 3, True)
        ]
        assert rows[0]["id_taken"] is False

    def test_undo_puts_it_back_with_everything_it_had(self, store):
        agent = _agent(store)
        client = _client(agent)
        before = _files(_embryo_dir(store))
        client.delete("/api/embryos/embryo_1")
        r = client.post("/api/embryos/embryo_1/restore")
        assert r.status_code == 200, r.text
        back = agent.experiment.embryos["embryo_1"]
        assert back.calibration["slope_um_per_deg"] == FIT["slope_um_per_deg"]
        assert back.position_coarse["x"] == 100.0
        assert _files(_embryo_dir(store)) == before
        assert client.get("/api/embryos/removed").json()["removed"] == []
        # and the rest were left as they were
        # and it is back in its place, not at the end
        assert list(agent.experiment.embryos) == ["embryo_1", "embryo_2", "embryo_3"]

    def test_an_embryo_the_run_is_imaging_is_refused(self, store):
        agent = _agent(store, run={"status": "running", "embryos": {"embryo_1": False}})
        r = _client(agent).delete("/api/embryos/embryo_1")
        assert r.status_code == 409
        assert "Stop it in the run first" in r.json()["detail"]
        assert "embryo_1" in agent.experiment.embryos
        assert _embryo_dir(store).exists()

    def test_a_paused_run_is_still_imaging_it(self, store):
        agent = _agent(store, run={"status": "paused", "embryos": {"embryo_1": False}})
        assert _client(agent).delete("/api/embryos/embryo_1").status_code == 409

    def test_one_the_run_has_finished_with_can_go(self, store):
        agent = _agent(store, run={"status": "running", "embryos": {"embryo_1": True}})
        assert _client(agent).delete("/api/embryos/embryo_1").status_code == 200

    def test_one_the_run_never_had_can_go(self, store):
        agent = _agent(store, run={"status": "running", "embryos": {"embryo_1": False}})
        assert _client(agent).delete("/api/embryos/embryo_2").status_code == 200

    def test_a_folder_that_cannot_be_moved_leaves_the_embryo_in_the_list(self, store, monkeypatch):
        import gently.core.file_store as fs_mod

        agent = _agent(store)

        def refuse(src, dst):
            raise PermissionError(13, "The process cannot access the file")

        monkeypatch.setattr(fs_mod.os, "rename", refuse)
        r = _client(agent).delete("/api/embryos/embryo_1")
        assert r.status_code == 409
        assert "was not removed" in r.json()["detail"]
        assert "embryo_1" in agent.experiment.embryos

    def test_restore_is_refused_when_the_name_is_taken(self, store):
        agent = _agent(store)
        client = _client(agent)
        client.delete("/api/embryos/embryo_2")
        agent.experiment.add_embryo(embryo_id="embryo_2", position={"x": 1.0, "y": 2.0})
        r = client.post("/api/embryos/embryo_2/restore")
        assert r.status_code == 409
        rows = client.get("/api/embryos/removed").json()["removed"]
        assert rows[0]["id_taken"] is True

    def test_restoring_what_was_never_removed_is_not_found(self, store):
        agent = _agent(store)
        agent.experiment.remove_embryo("embryo_3")
        assert _client(agent).post("/api/embryos/embryo_3/restore").status_code == 404

    def test_an_unknown_embryo_is_not_found(self, store):
        assert _client(_agent(store)).delete("/api/embryos/embryo_9").status_code == 404

    def test_removed_is_not_taken_for_an_embryo_id(self, store):
        r = _client(_agent(store)).get("/api/embryos/removed")
        assert r.status_code == 200 and r.json()["session_id"] == "s1"


# ── the page ─────────────────────────────────────────────────────────────


class TestThePage:
    def _delete(self) -> str:
        return OPERATE[OPERATE.index("    async function deleteEmbryo(id) {") :][:2600]

    def test_an_embryo_that_holds_something_is_asked_about(self):
        fn = self._delete()
        assert "const holds = heldBy(emb);" in fn
        assert "if (holds && !window.confirm(" in fn
        assert fn.index("window.confirm(") < fn.index("method: 'DELETE'")

    def test_the_question_says_nothing_is_deleted(self):
        assert "Nothing is deleted: its files are kept in the session" in self._delete()

    def test_what_it_holds_is_timepoints_and_a_calibration(self):
        fn = OPERATE[OPERATE.index("    function heldBy(emb) {") :][:600]
        assert "emb.timepoints_acquired" in fn
        assert "slope_um_per_deg" in fn

    def test_removal_offers_undo(self):
        fn = self._delete()
        assert "'Undo'" in fn and "() => restoreEmbryo(id)" in fn

    def test_a_refusal_is_said_in_the_servers_words(self):
        fn = self._delete()
        assert "data: await res.json().catch(() => ({}))" in fn
        assert "toastFail(`Not removed (${why(e)})`)" in fn

    def test_restore_is_a_verb_of_the_roster(self):
        assert "restore: id => restoreEmbryo(id)," in OPERATE
        fn = OPERATE[OPERATE.index("    async function restoreEmbryo(id) {") :][:700]
        assert "/restore`, { method: 'POST' }" in fn

    def test_the_roster_lists_the_removed_where_it_can_remove(self):
        fn = ROSTER[ROSTER.index("    function removed(opts) {") :][:600]
        assert "if (!opts.actions.includes('remove') || !list.length) return '';" in fn
        assert 'data-verb="restore"' in ROSTER
        assert "+ removed(opts);" in ROSTER

    def test_the_list_follows_the_roster_whoever_changed_it(self):
        assert "SharedState.on('embryos', removedMayHaveChanged);" in ROSTER
        assert "SharedState.on('removedEmbryos', render);" in ROSTER
        # the panel asks; operate.js, which owns the endpoints, reads
        assert "v.readRemoved()" in ROSTER and "fetch(" not in ROSTER
        assert "fetch('/api/embryos/removed')" in OPERATE
        assert "readRemoved: () => readRemoved()," in OPERATE

    def test_the_shared_state_knows_the_key(self):
        # SharedState.set refuses a key it does not declare, with a warning
        # in the console and nothing on the page.
        store = (WEB / "static" / "js" / "status-store.js").read_text(encoding="utf-8")
        assert "        removedEmbryos: []," in store

    def test_a_name_that_is_taken_cannot_be_restored_from_the_list(self):
        assert "${taken ? 'disabled' : ''}" in ROSTER

    def test_the_map_says_the_same_and_offers_the_same(self):
        fn = DEVICES[DEVICES.index("    async function attemptDeleteSelected() {") :][:1500]
        assert "Nothing is deleted" in fn
        assert "OperateManager.roster.restore(id)" in fn
