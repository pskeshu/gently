"""The scan calibration recovers a known sheet from hardware-timed volumes.

A fake rig whose volumes are rendered from a known slope, offset and extent:
each slice is a textured embryo blurred by its distance from the sheet, and
dark outside the embryo's galvo range. ``scan_calibrate`` must get the line
and the extent back, and borrowing must keep the lent slope while measuring
the target's own offset.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from scipy import ndimage

from gently.calibration import atoms
from gently.calibration.scan import scan_calibrate
from gently.harness.state import CalibrationPrior, EmbryoState

SLOPE, OFFSET, TOP, BOTTOM = 92.0, 37.0, -0.18, 0.14

_rng = np.random.default_rng(7)
_TEXTURE = _rng.random((120, 120))
_YY, _XX = np.mgrid[:120, :120]
_DISK = ((_YY - 60) ** 2 + (_XX - 60) ** 2) < 40**2


def _slice(galvo: float, piezo: float) -> np.ndarray:
    defocus = abs(piezo - (SLOPE * galvo + OFFSET))
    img = ndimage.gaussian_filter(_TEXTURE, sigma=0.3 + 0.6 * defocus)
    lit = 1.0 if TOP <= galvo <= BOTTOM else 0.02
    view = (img * _DISK * lit * 4000 + _rng.random((120, 120)) * 20).astype(np.uint16)
    return np.concatenate([view, view // 3], axis=1)  # two views side by side


def _client() -> MagicMock:
    async def acquire_volume(
        num_slices, galvo_amplitude, galvo_center, piezo_amplitude, piezo_center, **_
    ):
        gal = atoms.slice_positions(galvo_center, galvo_amplitude, num_slices)
        pie = atoms.slice_positions(piezo_center, piezo_amplitude, num_slices)
        return {
            "success": True,
            "volume": np.stack([_slice(g, p) for g, p in zip(gal, pie, strict=True)]),
        }

    c = MagicMock()
    c.is_connected = True
    c.acquire_volume = AsyncMock(side_effect=acquire_volume)
    c.set_led = AsyncMock()
    c.move_to_position = AsyncMock()
    return c


def _agent(*embryos: EmbryoState) -> MagicMock:
    a = MagicMock()
    a.viz_server = None
    a.store = None
    a.experiment = MagicMock()
    a.experiment.calibration_prior = CalibrationPrior()
    a.experiment.embryos = {e.id: e for e in embryos}
    a.experiment.get_embryo_by_any_name = a.experiment.embryos.get
    return a


@pytest.fixture(autouse=True)
def _no_api(monkeypatch):
    vision = MagicMock()
    vision.detect_embryo_presence = AsyncMock(return_value=(True, 8, "embryo"))
    # The fake has no eyes: it agrees with the metric (the middle of the window).
    vision.select_best_focus = AsyncMock(
        side_effect=lambda _p, offsets, _l=None: (len(offsets) // 2, "D", "sharpest")
    )
    monkeypatch.setattr("gently.hardware.dispim.claude_client.AsyncClaudeClient", lambda: vision)


# ── pure ────────────────────────────────────────────────────────────────────


def test_a_line_through_noisy_points_is_the_line() -> None:
    g = np.array([-0.1, 0.0, 0.1, 0.2])
    slope, offset, r2 = atoms.fit_line(g, SLOPE * g + OFFSET + np.array([0.1, -0.1, 0.05, -0.05]))
    assert abs(slope - SLOPE) < 1 and abs(offset - OFFSET) < 0.2 and r2 > 0.99


def test_an_extent_that_overruns_the_sweep_is_not_measured() -> None:
    pos = np.linspace(-0.5, 0.5, 11)
    assert atoms.extent_from_signal(np.ones(11), pos) is None
    sig = np.zeros(11)
    sig[3:7] = 1
    assert atoms.extent_from_signal(sig, pos) == (pytest.approx(-0.2), pytest.approx(0.1))


def test_the_budget_refuses_before_the_rig_is_asked() -> None:
    b = atoms.Budget(10)
    b.spend(8)
    with pytest.raises(atoms.BudgetExceeded):
        b.spend(3)


# ── the routine ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_recovers_the_sheet_and_the_extent() -> None:
    emb = EmbryoState(id="embryo_1")
    emb.stage_position = {"x": 1.0, "y": 2.0}
    agent = _agent(emb)
    msg = await scan_calibrate("embryo_1", {"agent": agent, "client": _client()})
    assert msg.startswith("✓"), msg
    cal = emb.calibration
    assert cal["method"] == "scan"
    assert abs(cal["slope_um_per_deg"] - SLOPE) < 4, msg
    assert abs(cal["offset_um"] - OFFSET) < 1.0, msg
    assert (
        abs(cal["galvo_top_deg"] - TOP) < 0.02 and abs(cal["galvo_bottom_deg"] - BOTTOM) < 0.02
    ), msg
    assert cal["r_squared"] > 0.95 and len(cal["points"]) >= 3
    assert 0 < cal["exposures"] <= atoms.Budget(1).ceiling or cal["exposures"] < 600
    assert agent.experiment.calibration_prior.num_calibrations == 1


@pytest.mark.asyncio
async def test_a_borrowed_slope_is_kept_and_the_offset_is_this_embryos_own() -> None:
    emb = EmbryoState(id="embryo_2")
    agent = _agent(emb)
    lent = (SLOPE + 3.0, OFFSET - 10.0)  # a near-miss from another embryo
    msg = await scan_calibrate(
        "embryo_2", {"agent": agent, "client": _client()}, start=lent, n_focus=1
    )
    assert msg.startswith("✓"), msg
    cal = emb.calibration
    assert abs(cal["offset_um"] - OFFSET) < 1.5, msg
    assert abs(cal["piezo_center"] - (SLOPE * (TOP + BOTTOM) / 2 + OFFSET)) < 2.0, (
        "the volume is centred where the lender sits, not where this embryo is"
    )


@pytest.mark.asyncio
async def test_an_empty_field_is_refused_not_fitted() -> None:
    emb = EmbryoState(id="embryo_3")
    agent = _agent(emb)
    client = _client()
    dark = np.zeros((1, 120, 240), dtype=np.uint16)
    client.acquire_volume = AsyncMock(
        side_effect=lambda num_slices, **_: {
            "success": True,
            "volume": np.repeat(dark, num_slices, 0),
        }
    )
    msg = await scan_calibrate("embryo_3", {"agent": agent, "client": client})
    assert msg.startswith("No object visible"), msg
    assert not emb.calibration
