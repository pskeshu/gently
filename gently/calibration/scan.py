"""The ``scan`` calibration: a few hardware-timed volumes and a line through them.

1. ``extent`` — one slice-only scan at the heuristic piezo: where the embryo
   is in galvo, and one (galvo, piezo) point.
2. ``focus_at`` — one piezo-only scan at each of ``n_focus`` galvo positions
   across the extent: one point each.
3. ``fit_line`` through every point that held, ``geometry`` from the extent.

Writes the same calibration dict and the same record as ``legacy``, so the
timelapse, the gate, and the records view cannot tell them apart except by
``calibration["method"]`` — which is the point of keeping both.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np

from gently.calibration import atoms
from gently.core import spim_alignment
from gently.harness.tools.helpers import ctx_get, get_embryo_or_error

logger = logging.getLogger(__name__)

# ponytail: one ceiling for every embryo. Per-role budgets arrive with roles.
EXPOSURE_CEILING = 600
MIN_POINT_R2 = 0.5


def heuristic(agent, embryo, side: str = "A") -> tuple[float, float, str]:
    """Slope and offset to start from, and where they came from. Legacy's priority."""
    prior = agent.experiment.calibration_prior
    if prior.num_calibrations > 0 and prior.r_squared_mean >= 0.75:
        return (
            prior.slope_um_per_deg,
            prior.offset_um,
            f"session prior from {prior.num_calibrations} embryo(s)",
        )
    cal = embryo.calibration or {}
    if side == "B":
        # B is the other arm: its own previous fit or nothing; A's numbers
        # belong to A's scanner and piezo.
        cal = cal.get("B") if isinstance(cal.get("B"), dict) else {}
    if cal.get("slope_um_per_deg"):
        return cal["slope_um_per_deg"], cal.get("offset_um", 0.0), "this embryo's previous fit"
    return 100.0, 0.0, "the default heuristic"


async def _ask_visible(vision, img: np.ndarray) -> tuple[bool, int, str]:
    from PIL import Image

    norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = Path(f.name)
        Image.fromarray(norm).save(path)
    try:
        seen, score, desc = await vision.detect_embryo_presence(path)
        return bool(seen), int(score or 0), str(desc or "")
    finally:
        path.unlink(missing_ok=True)


async def scan_calibrate(
    embryo_id: str,
    context: dict | None,
    *,
    z_buffer_um: float = 25.0,
    n_focus: int = 3,
    exposure_ms: float = 10.0,
    galvo_half_range: float = 0.5,
    require_object: bool = True,
    start: tuple[float, float] | None = None,
    side: str = "A",
) -> str:
    """Calibrate one embryo with the scan atoms. Returns the same prose legacy does.

    ``side`` is the imaging path: A writes ``embryo.calibration`` as ever; B
    writes ``embryo.calibration["B"]``, measured with the B scanner, piezo and
    camera, and leaves the session prior and A's focus history alone.

    ``start`` is a borrowed (slope, offset). With it, one held point is enough:
    the slope is kept and only the offset is this embryo's own.
    """
    from gently.app.tools.calibration_tools import (
        NO_OBJECT_PREFIX,
        _recording,
        _show,
        hybrid_focus_selection,
    )
    from gently.hardware.dispim.claude_client import AsyncClaudeClient

    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")
    if not agent:
        return "Error: No agent context"
    if not client or not getattr(client, "is_connected", False):
        return f"Error: Not connected to microscope server. Cannot calibrate {embryo_id}."
    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    slope0, offset0, origin = (
        (*start, "a borrowed line") if start else heuristic(agent, embryo, side)
    )
    budget = atoms.Budget(EXPOSURE_CEILING)
    vision = AsyncClaudeClient()
    said: list[str] = [f"starting from {origin}: {slope0:.1f} µm/deg, offset {offset0:.1f} µm"]

    def show(**kw):
        if getattr(agent, "viz_server", None) or _recording(agent, embryo_id):
            _show(agent, **kw)

    async def judge(images, offsets):
        idx, how, _ = await hybrid_focus_selection(images, offsets, vision, agent, embryo_id)
        return idx if how == "vision" else None

    try:
        await client.set_led("Closed")
    except Exception as led_err:
        logger.warning("Could not close the LED before calibration (%s)", led_err)
    pos = embryo.stage_position
    if pos and pos.get("x") is not None and pos.get("y") is not None:
        await spim_alignment.move_to_embryo(client, pos)

    try:
        ext = await atoms.extent(
            client,
            piezo=offset0,
            galvo_half_range=galvo_half_range,
            exposure_ms=exposure_ms,
            budget=budget,
            side=side,
            embryo_id=embryo_id,
            show=show,
            visible=(lambda img: _ask_visible(vision, img)) if require_object else None,
        )
        said.append(ext.says)
        if not ext.ok:
            return f"{NO_OBJECT_PREFIX} for {embryo_id}: {ext.says}"
        top, bottom = ext.data["galvo_top"], ext.data["galvo_bottom"]
        points: list[tuple[float, float, float, float]] = []  # galvo, piezo, r2, max_score
        if ext.data["r_squared"] >= MIN_POINT_R2:
            points.append(
                (ext.data["galvo_at_piezo"], ext.data["piezo"], ext.data["r_squared"], 0.0)
            )

        # Centre each focus window on the line through a point just measured,
        # not on the heuristic offset: the slope guess is good to a micron over
        # the extent, the offset guess need not be good at all. When the sheet
        # did not cross the plane inside the embryo, one wide look at the
        # centre finds the line first.
        if ext.data["r_squared"] >= MIN_POINT_R2:
            g_ref, p_ref = ext.data["galvo_at_piezo"], ext.data["piezo"]
        else:
            g_mid = (top + bottom) / 2
            wide = await atoms.focus_at(
                client,
                galvo=g_mid,
                expected_piezo=slope0 * g_mid + offset0,
                half_range_um=40.0,
                step_um=1.0,
                exposure_ms=exposure_ms,
                budget=budget,
                side=side,
                embryo_id=embryo_id,
                show=show,
                judge=judge,
            )
            said.append("wide look: " + wide.says)
            if not wide.ok:
                return (
                    f"Error calibrating {embryo_id}: no focus found within ±40 µm of the "
                    f"heuristic at the centre\n  " + "\n  ".join(said)
                )
            g_ref, p_ref = wide.data["galvo"], wide.data["piezo"]
            points.append((g_ref, p_ref, wide.data["r_squared"], wide.data["max_score"]))
        inset = 0.2 * (bottom - top)
        for g in np.linspace(top + inset, bottom - inset, n_focus):
            rep = await atoms.focus_at(
                client,
                galvo=float(g),
                expected_piezo=p_ref + slope0 * (float(g) - g_ref),
                exposure_ms=exposure_ms,
                budget=budget,
                side=side,
                embryo_id=embryo_id,
                show=show,
                judge=judge,
            )
            said.append(rep.says)
            if rep.ok:
                points.append(
                    (
                        rep.data["galvo"],
                        rep.data["piezo"],
                        rep.data["r_squared"],
                        rep.data["max_score"],
                    )
                )
        # ponytail: no second pass. If fewer than two points hold, say so;
        # a narrower re-scan at the worst residual is the upgrade.
        if not points or (len(points) < 2 and start is None):
            return (
                f"Error calibrating {embryo_id}: only {len(points)} focus point(s) held, "
                "two are needed\n  " + "\n  ".join(said)
            )
        g_arr, p_arr = np.array([p[0] for p in points]), np.array([p[1] for p in points])
        if start is not None:
            # The lender's slope is the optics'; only the offset is this embryo's.
            # A line through two points a few hundredths of a degree apart would
            # amplify every micron of noise into the slope — the legacy failure.
            slope = slope0
            offset = float(np.mean(p_arr - slope * g_arr))
            r2 = float(np.mean([p[2] for p in points]))
            said.append(
                f"kept the borrowed slope {slope:.2f} µm/deg; this embryo's offset is "
                f"{offset:.2f} µm from {len(points)} point(s)"
            )
        else:
            slope, offset, r2 = atoms.fit_line(g_arr, p_arr)
            said.append(
                f"line through {len(points)} points: {slope:.2f} µm/deg, "
                f"offset {offset:.2f} µm, R² {r2:.3f}"
            )
    except atoms.BudgetExceeded as exc:
        return f"Error calibrating {embryo_id}: {exc}\n  " + "\n  ".join(said)

    geo = atoms.geometry(top, bottom, slope, offset, z_buffer_um)
    fit = {
        "method": "scan",
        "slope_um_per_deg": slope,
        "offset_um": offset,
        "r_squared": r2,
        "galvo_top_deg": top,
        "galvo_bottom_deg": bottom,
        "galvo_calib_top_deg": float(g_arr.min()),
        "galvo_calib_bottom_deg": float(g_arr.max()),
        "piezo_top_um": float(p_arr[np.argmin(g_arr)]),
        "piezo_bottom_um": float(p_arr[np.argmax(g_arr)]),
        "points": [{"galvo": g, "piezo": p, "r_squared": q} for g, p, q, _ in points],
        # legacy readers judge quality by the two ends; give them the ends.
        "r_squared_top": points[int(np.argmin(g_arr))][2],
        "r_squared_bottom": points[int(np.argmax(g_arr))][2],
        "exposures": budget.spent,
        **geo,
    }
    if side == "B":
        base = dict(embryo.calibration or {})
        base["B"] = fit
        embryo.calibration = base
    else:
        embryo.calibration = fit
    try:
        agent.experiment.notify_embryos_changed()
    except Exception:
        logger.debug("notify_embryos_changed failed", exc_info=True)
    if side == "A":
        # The prior and the focus history are A's optics; B gets its own when
        # there is a second arm's worth of nights to learn from.
        agent.experiment.calibration_prior.update_from_calibration(
            slope=slope, offset=offset, r_squared=r2, extent_deg=bottom - top
        )
        for g, p, q, s in points:
            embryo.add_focus_datapoint(
                galvo=g,
                piezo=p,
                score=s,
                r_squared=q,
                method="calibration",
                algorithm="fft_bandpass",
            )
    embryo.record_exposure(exposure_ms=exposure_ms, num_frames=budget.spent)
    try:
        from gently.ui.web.plots import generate_calibration_summary_plot

        show(
            array=generate_calibration_summary_plot(
                embryo_id=embryo_id,
                galvo_top=float(g_arr.min()),
                galvo_bottom=float(g_arr.max()),
                piezo_top=fit["piezo_top_um"],
                piezo_bottom=fit["piezo_bottom_um"],
                slope=slope,
                offset=offset,
                r_squared_top=fit["r_squared_top"],
                r_squared_bottom=fit["r_squared_bottom"],
            ),
            uid=f"calibration_summary_{embryo_id}" + ("" if side == "A" else f"_{side}"),
            data_type="calibration_summary",
            metadata={
                "embryo_id": embryo_id,
                "slope": slope,
                "offset": offset,
                "r_squared": r2,
                "method": "scan",
                "side": side,
                "said": said,
            },
        )
    except Exception as exc:
        logger.warning("summary plot for %s failed: %s", embryo_id, exc)
    agent._mark_significant_action("calibration")
    return (
        f"✓ Calibrated {embryo_id} (scan, side {side}, {budget.spent} exposures)\n"
        f"  Embryo extent: galvo {top:.3f}° to {bottom:.3f}° (~{(bottom - top) * 100:.0f}µm)\n"
        f"  Slope: {slope:.2f} µm/deg\n"
        f"  Offset: {offset:.2f} µm (piezo at galvo=0)\n"
        f"  Fit: {len(points)} points, R² {r2:.3f}\n"
        f"  Volume params: galvo={geo['galvo_center']:.3f}°±{geo['galvo_amplitude']:.3f}°, "
        f"piezo={geo['piezo_center']:.1f}µm±{geo['piezo_amplitude']:.1f}µm\n"
        f"  Z buffer: ±{z_buffer_um:.0f}µm above/below embryo\n"
        "  " + "\n  ".join(said)
    )
