"""The SPIM head's centre is measured and written down, not assumed.

`frameToStage` defines an embryo's position as the stage XY at which it sits at
the BOTTOM CAMERA's centre pixel, and `centerOnEmbryo` drove the stage to
exactly that. Nothing mapped that point to the SPIM's optical axis, so the code
assumed the two coincided — exactly, permanently, unmeasured.

When they do not, centring puts the embryo outside the light sheet and the
pre-calibration check reports "No object visible" while every instrument check
passes. That is the worst kind of bug: nothing to look at.

What must stay true:

* an instrument nobody has aligned behaves EXACTLY as before — the default is
  the old assumption, stated rather than implied;
* "unmeasured" and "measured as zero" are different, and the UI can tell;
* nothing is ever overwritten, because an alignment is a physical fact about a
  day and the record of when it changed is worth as much as the value; and
* every path that drives to an embryo uses the same correction, or centring
  from chat lands somewhere centring from the pane does not.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gently.core import spim_alignment

GENTLY = Path(__file__).resolve().parents[1] / "gently"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Never touch the real D:/Gently3/config/spim_alignment.yaml."""
    monkeypatch.setattr(
        spim_alignment, "alignment_path", lambda: tmp_path / "config" / "spim_alignment.yaml"
    )
    return tmp_path


# ---------------------------------------------------------------------------
# The default is the old assumption
# ---------------------------------------------------------------------------


def test_an_unaligned_instrument_moves_exactly_where_it_always_did() -> None:
    assert spim_alignment.centre_target(1000.0, -500.0) == (1000.0, -500.0)


def test_unmeasured_is_not_the_same_as_zero() -> None:
    """A measured (0, 0) means someone checked; unmeasured means nobody has."""
    assert spim_alignment.load().current.is_measured is False
    spim_alignment.set_offset(0.0, 0.0, session_id="s1")
    assert spim_alignment.load().current.is_measured is True


def test_a_missing_file_is_not_an_error() -> None:
    assert spim_alignment.load().current.dx_um == 0.0


def test_a_corrupt_file_falls_back_rather_than_raising() -> None:
    """The failure mode of a bad offset is a stage move to nowhere."""
    path = spim_alignment.alignment_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("current: [this is not a mapping]\n", encoding="utf-8")
    record = spim_alignment.load()
    assert record.current.dx_um == 0.0 and record.current.is_measured is False


# ---------------------------------------------------------------------------
# Nothing is overwritten
# ---------------------------------------------------------------------------


def test_setting_keeps_what_it_replaced() -> None:
    spim_alignment.set_offset(-112.4, 38.9, session_id="s1", note="first")
    spim_alignment.set_offset(-100.0, 40.0, session_id="s2", note="second")
    record = spim_alignment.load()
    assert record.current.dx_um == -100.0
    # Both the unmeasured default and the first measurement are kept.
    assert [round(h.dx_um, 1) for h in record.history] == [0.0, -112.4]


def test_the_session_is_the_provenance_not_the_embryo() -> None:
    """embryo_4 is a per-session label that means nothing next week."""
    spim_alignment.set_offset(1.0, 2.0, session_id="a4c6639e")
    current = spim_alignment.load().current
    assert current.session_id == "a4c6639e"
    assert "embryo" not in current.to_dict()


def test_restoring_is_itself_undoable() -> None:
    first = spim_alignment.set_offset(
        -112.4, 38.9, session_id="s1", now=datetime(2026, 9, 22, 1, 0, 0)
    )
    spim_alignment.set_offset(-5.0, -5.0, session_id="s2", now=datetime(2026, 9, 22, 2, 0, 0))

    restored = spim_alignment.restore(first.current.set_at, session_id="s3")
    assert restored is not None
    assert (restored.current.dx_um, restored.current.dy_um) == (-112.4, 38.9)
    # The value it replaced is in history, so the restore can be undone.
    assert any(h.dx_um == -5.0 for h in restored.history)
    assert "restored from" in restored.current.note


def test_restoring_something_that_never_existed_reports_rather_than_no_ops() -> None:
    spim_alignment.set_offset(1.0, 1.0, session_id="s1")
    assert spim_alignment.restore("2020-01-01T00:00:00") is None


def test_history_is_bounded() -> None:
    for i in range(spim_alignment.MAX_HISTORY + 10):
        spim_alignment.set_offset(float(i), 0.0, session_id="s")
    assert len(spim_alignment.load().history) <= spim_alignment.MAX_HISTORY


# ---------------------------------------------------------------------------
# One correction, every path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_moving_to_an_embryo_goes_through_the_offset() -> None:
    spim_alignment.set_offset(-112.4, 38.9, session_id="s1")
    client = MagicMock()
    client.move_to_position = AsyncMock(return_value={"success": True})
    await spim_alignment.move_to_embryo(client, {"x": 1000.0, "y": -500.0})
    args = client.move_to_position.await_args.args
    assert args == (pytest.approx(887.6), pytest.approx(-461.1))


def test_every_embryo_bound_mover_uses_the_helper() -> None:
    """A sixth call site that forgets it would drive somewhere the others do not."""
    embryo_movers = {
        "app/orchestration/timelapse.py",
        "app/orchestration/exclusive.py",
        "app/tools/calibration_tools.py",
        "app/tools/acquisition_tools.py",
        # The agent's own "move to embryo" tool. Found by this test while it
        # was being written: it was the fifth mover and it drove uncorrected,
        # so asking the agent to go to an embryo landed somewhere the pane's
        # Centre did not.
        "app/tools/stage_tools.py",
    }
    for rel in embryo_movers:
        src = (GENTLY / rel).read_text(encoding="utf-8")
        assert "spim_alignment.move_to_embryo(" in src, (
            f"{rel} drives to an embryo without the SPIM offset — centring from "
            "the pane and from the agent would land in different places"
        )
        assert 'move_to_position(pos["x"], pos["y"])' not in src, (
            f"{rel} still has a raw uncorrected move to an embryo position"
        )


def test_an_explicit_coordinate_is_never_silently_offset() -> None:
    """ "Move to x=1000, y=500" means that coordinate, not a corrected one.

    The offset belongs to "go to this EMBRYO", whose position is expressed in
    bottom-camera terms. Adding it to a coordinate the caller typed would make
    the number they asked for a lie.
    """
    src = (GENTLY / "app" / "tools" / "stage_tools.py").read_text(encoding="utf-8")
    move_stage = re.search(r"async def move_stage\(.*", src, re.S)
    assert move_stage, "move_stage is gone"
    assert "spim_alignment" not in move_stage.group(0), (
        "the arbitrary-coordinate tool now applies the SPIM offset"
    )

    route = (GENTLY / "ui" / "web" / "routes" / "data.py").read_text(encoding="utf-8")
    start = route.index('@router.post("/api/devices/stage/move"')
    assert "spim_alignment" not in route[start : start + 1200], (
        "the stage-move route now applies the SPIM offset to an explicit coordinate"
    )


def test_the_pane_and_the_server_compute_the_same_target() -> None:
    """CI runs no JS, so the pane's copy of the arithmetic is pinned here."""
    js = (GENTLY / "ui" / "web" / "static" / "js" / "operate-math.js").read_text(encoding="utf-8")
    fn = re.search(r"function centreTarget\(x, y, offset\) \{(.*?)\n    \}", js, re.S)
    assert fn, "the pane lost centreTarget"
    body = fn.group(1)
    assert "x + dx" in body and "y + dy" in body, (
        "the pane's correction is no longer a straight addition; it must match "
        "spim_alignment.centre_target or the pane and the agent disagree"
    )
    assert "Number.isFinite(offset.dx_um) ? offset.dx_um : 0" in body, (
        "an unmeasured or malformed offset must degrade to no correction"
    )


def test_centring_in_the_pane_goes_through_it() -> None:
    js = (GENTLY / "ui" / "web" / "static" / "js" / "operate.js").read_text(encoding="utf-8")
    centre = re.search(r"async function centerOnEmbryo\(emb\) \{(.*?)\n    \}", js, re.S)
    assert centre, "centerOnEmbryo is gone"
    # Through `M`, the module's guarded alias — operate.js reaches the maths
    # that way everywhere, so that a missing operate-math.js degrades to the
    # raw position instead of throwing mid-move.
    assert "M.centreTarget(" in centre.group(1), (
        "the pane centres on the bottom camera's centre pixel again, ignoring "
        "where the SPIM head actually looks"
    )
    assert "M ? M.centreTarget(" in centre.group(1), (
        "the correction is no longer guarded; if operate-math.js fails to load, "
        "centring throws instead of falling back to the uncorrected position"
    )


def test_the_control_lives_where_centring_does() -> None:
    """On the bottom camera, behind Advanced.

    The offset governs what "centre on this embryo" means on THIS pane, so the
    control belongs beside it rather than on the SPIM head. Behind a
    disclosure because it is an instrument fact that changes when someone
    re-seats the head — rarely, and never by accident.
    """
    html = (GENTLY / "ui" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    bottom = html[html.index('id="op-pane-bottom"') : html.index('id="op-pane-spim"')]
    assert 'id="op-align-set"' in bottom, "the SPIM-centre control left the bottom camera pane"
    assert 'id="op-adv"' in bottom, "it is no longer behind the Advanced disclosure"

    spim = html[html.index('id="op-pane-spim"') :]
    assert 'id="op-align-set"' not in spim, "a second copy of the control is on the SPIM pane"


def test_the_offset_in_effect_is_never_hidden_behind_the_disclosure() -> None:
    """A correction nobody can see is how people chase ghosts."""
    html = (GENTLY / "ui" / "web" / "templates" / "index.html").read_text(encoding="utf-8")
    bottom = html[html.index('id="op-pane-bottom"') : html.index('id="op-pane-spim"')]
    line_at = bottom.index('id="op-align-line"')
    adv_at = bottom.index('id="op-adv"')
    assert line_at < adv_at, (
        "the always-visible offset line moved inside the Advanced block, so an "
        "operator could be running a correction they cannot see"
    )
