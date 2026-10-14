"""A session resumes when an embryo's fit lives only in the embryo table.

The snapshot is the primary source on resume; the embryo table fills in
embryos the snapshot does not have. That path read the fit with
``json.loads`` — right for the SQLite store, which kept a JSON string, wrong
for FileStore, which keeps a dict. Resuming such a session raised
``TypeError: the JSON object must be str, bytes or bytearray, not dict`` and
nothing resumed. Seen on a seeded dev session; real whenever an embryo is
registered with a fit and the session is not saved again before a resume.
"""

from __future__ import annotations

from gently.core.file_store import FileStore
from gently.harness.session.manager import SessionManager
from gently.harness.state import ExperimentState

FIT = {"slope_um_per_deg": 98.2, "offset_um": 3.1, "r_squared_top": 0.9, "r_squared_bottom": 0.9}


def test_a_fit_kept_as_a_dict_survives_the_resume(tmp_path) -> None:
    store = FileStore(root=tmp_path)
    sid = store.create_session("fit-only", name="fit-only")
    store.register_embryo(sid, "embryo_1", position_x=1.0, position_y=2.0, calibration=FIT)

    experiment = ExperimentState()
    ok, _history = SessionManager(store, tmp_path)._resume_session(sid, experiment)

    assert ok
    assert experiment.embryos["embryo_1"].calibration["slope_um_per_deg"] == 98.2


def test_a_fit_kept_as_json_text_still_resumes(tmp_path, monkeypatch) -> None:
    """The SQLite-era shape is read the same way it always was."""
    import json
    from typing import Any

    store = FileStore(root=tmp_path)
    sid = store.create_session("fit-text", name="fit-text")
    store.register_embryo(sid, "embryo_1", position_x=1.0, position_y=2.0, calibration=FIT)
    rows: list[Any] = [dict(r) for r in store.list_embryos(sid)]
    rows[0]["calibration"] = json.dumps(FIT)
    monkeypatch.setattr(store, "list_embryos", lambda _sid: rows)

    experiment = ExperimentState()
    ok, _ = SessionManager(store, tmp_path)._resume_session(sid, experiment)
    assert ok and experiment.embryos["embryo_1"].calibration == FIT
