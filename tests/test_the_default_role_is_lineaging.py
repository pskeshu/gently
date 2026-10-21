"""The default embryo role is lineaging: the nuclear perception route.

"change default so we go to lineage i think so we go to the perception route."
Every door an embryo comes in through agrees, so a slide marked and never
labelled is staged rather than read for a reporter it may not carry.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from gently.app.tools import detection_tools
from gently.harness.roles import DEFAULT_ROLE, REGISTRY
from gently.harness.state import EmbryoState, ExperimentState

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"


def test_the_default_is_lineaging_and_lineaging_is_perception():
    assert DEFAULT_ROLE == "lineaging"
    assert REGISTRY["lineaging"].detector_name == "perception"
    assert REGISTRY["lineaging"].no_object_consecutive_terminal == 5, (
        "the default keeps the old default's patience for an embryo out of frame"
    )


def test_a_new_embryo_state_is_lineaging():
    assert EmbryoState(id="embryo_1").role == "lineaging"
    exp = ExperimentState()
    exp.add_embryo(embryo_id="embryo_1", position={"x": 1.0, "y": 2.0})
    assert exp.embryos["embryo_1"].role == "lineaging"


def test_the_detect_tools_hand_the_map_the_default():
    for name in ("detect_embryos", "mark_embryos"):
        fn = getattr(detection_tools, name, None)
        if fn is None:
            continue
        fn = getattr(fn, "__wrapped__", fn)
        assert inspect.signature(fn).parameters["default_role"].default == DEFAULT_ROLE, name


def test_the_store_files_an_unlabelled_embryo_as_lineaging(tmp_path):
    from gently.core.file_store import FileStore

    store = FileStore(root=tmp_path)
    store.create_session("s1")
    store.register_embryo("s1", "embryo_1", position_coarse={"x": 1.0, "y": 2.0})
    assert store.get_embryo("s1", "embryo_1")["role"] == "lineaging"


def test_the_pane_agrees():
    operate = (JS / "operate.js").read_text(encoding="utf-8")
    roster = (JS / "panels" / "roster.js").read_text(encoding="utf-8")
    assert "emb.role !== 'unassigned') ? emb.role : 'lineaging'" in operate
    assert "emb.role !== 'unassigned') ? emb.role : 'lineaging'" in roster
    assert "defaultRole: d.default_role || 'lineaging'" in operate
    assert "cur === 'calibration' ? 'lineaging' : 'calibration'" in operate
    assert "'test'" not in operate.split("toggleRole(id)")[1][:600], (
        "the subject/reference toggle still hands an embryo the reporter role"
    )
