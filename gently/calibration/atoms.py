"""Calibration atoms. Each one measures one thing and says what it saw.

The ASIdiSPIM plugin has two scan modes it calls "unusual": slice scan only
(galvo sweeps, piezo still) and piezo scan only (piezo steps, galvo still).
Useless for imaging; exactly the two measurements calibration is made of.
Both are one hardware-timed volume with one axis's amplitude at zero, which
``acquire_volume`` passes straight through.

Every atom returns a ``Report``: numbers, one line for whoever is watching,
and what it cost. The pure functions underneath take arrays and are what the
test exercises.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gently.analysis.core import FitFunction, calculate_focus_score, fit_focus_curve
from gently.app.tools.hardware_common import select_best_view, select_view_and_crop_roi

logger = logging.getLogger(__name__)


# ponytail: ASI SingleAxisAmplitude is peak-to-peak, so slice i of N sits at
# center - amp/2 + i*amp/(N-1). The legacy geometry hands the device a
# half-range as amplitude; the z buffer hides it. Not corrected here —
# see #267 — but the atoms do their own slice→position mapping on this basis.
def slice_positions(center: float, amplitude: float, n: int) -> np.ndarray:
    if n < 2:
        return np.array([center])
    return np.linspace(center - amplitude / 2, center + amplitude / 2, n)


# ponytail: 0.0 exactly, as the plugin does. The legacy snap path parks the
# galvo at 0.0001°; if the controller rejects a zero amplitude, raise this.
STILL = 0.0

Show = Callable[..., None]  # show(array=, uid=, data_type=, metadata=)
Visible = Callable[[np.ndarray], Awaitable[tuple[bool, int, str]]]


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Exposures this embryo may still be given. Counted here, refused here."""

    ceiling: int
    spent: int = 0

    def spend(self, n: int) -> None:
        if self.spent + n > self.ceiling:
            raise BudgetExceeded(
                f"{self.spent + n} exposures would exceed the {self.ceiling} allowed"
            )
        self.spent += n


@dataclass
class Report:
    atom: str
    ok: bool
    says: str
    exposures: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "atom": self.atom,
            "ok": self.ok,
            "says": self.says,
            "exposures": self.exposures,
            **self.data,
        }


# ── pure ────────────────────────────────────────────────────────────────────


def slice_signal(volume: np.ndarray) -> np.ndarray:
    """How much is lit in each slice: bright tail over background."""
    out = np.empty(len(volume), dtype=float)
    for i, sl in enumerate(volume):
        v = select_best_view(sl).astype(float)
        out[i] = float(np.percentile(v, 99.5) - np.median(v))
    return out


def extent_from_signal(
    signal: np.ndarray, positions: np.ndarray, frac: float = 0.25
) -> tuple[float, float] | None:
    """First and last position where the signal clears `frac` of its peak.

    None when nothing clears the floor, or the lit run touches either end
    of the sweep — then the embryo is bigger than the sweep, not measured.
    """
    if len(signal) == 0 or signal.max() <= 0:
        return None
    lit = np.flatnonzero(signal >= frac * signal.max())
    if len(lit) == 0 or lit[0] == 0 or lit[-1] == len(signal) - 1:
        return None
    return float(positions[lit[0]]), float(positions[lit[-1]])


def focus_from_scores(positions: np.ndarray, scores: np.ndarray) -> tuple[float, float, Any]:
    """Best position, R², fit params. Argmax with R²=0 when the fit will not take."""
    best = float(positions[int(np.argmax(scores))])
    if len(positions) < 4:
        return best, 0.0, None
    try:
        _, _, params, r2 = fit_focus_curve(positions, scores, FitFunction.GAUSSIAN.value)
    except Exception:
        return best, 0.0, None
    mu = float(np.clip(params[1], positions.min(), positions.max()))
    return mu, float(r2), params


def fit_line(galvos: np.ndarray, piezos: np.ndarray) -> tuple[float, float, float]:
    """piezo = slope * galvo + offset, with R². Two points fit exactly (R²=1)."""
    g, p = np.asarray(galvos, float), np.asarray(piezos, float)
    if len(g) < 2 or np.ptp(g) == 0:
        raise ValueError("need two distinct galvo positions to fit a line")
    slope, offset = np.polyfit(g, p, 1)
    pred = slope * g + offset
    ss_res = float(np.sum((p - pred) ** 2))
    ss_tot = float(np.sum((p - p.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(offset), float(max(0.0, r2))


def geometry(
    top: float, bottom: float, slope: float, offset: float, z_buffer_um: float
) -> dict[str, float]:
    """Volume parameters from the extent and the fit. Same formula as legacy."""
    buf = z_buffer_um / 100.0  # 0.01° ≈ 1 µm
    p_top = slope * (top - buf) + offset
    p_bot = slope * (bottom + buf) + offset
    return {
        "galvo_center": (top + bottom) / 2,
        "galvo_amplitude": (bottom - top) / 2 + buf,
        "piezo_center": (p_top + p_bot) / 2,
        "piezo_amplitude": abs(p_bot - p_top) / 2,
        "z_buffer_um": z_buffer_um,
    }


# ── hardware ────────────────────────────────────────────────────────────────


async def _volume(
    client,
    *,
    n: int,
    exposure_ms: float,
    galvo_amp: float,
    galvo_c: float,
    piezo_amp: float,
    piezo_c: float,
    budget: Budget,
    side: str = "A",
) -> np.ndarray | None:
    budget.spend(n)
    r = await client.acquire_volume(
        num_slices=n,
        exposure_ms=exposure_ms,
        galvo_amplitude=galvo_amp,
        galvo_center=galvo_c,
        piezo_amplitude=piezo_amp,
        piezo_center=piezo_c,
        side=side,
    )
    if not r.get("success") or r.get("volume") is None:
        return None
    vol = np.asarray(r["volume"])
    return vol if vol.ndim == 3 else vol[None]


async def extent(
    client,
    *,
    piezo: float,
    galvo_center: float = 0.0,
    galvo_half_range: float = 0.5,
    step_deg: float = 0.01,
    exposure_ms: float = 10.0,
    budget: Budget,
    embryo_id: str = "",
    show: Show | None = None,
    visible: Visible | None = None,
    side: str = "A",
) -> Report:
    """Slice scan only: the sheet sweeps, the detection plane stays at `piezo`.

    The signal envelope along the sweep is the embryo's galvo extent. The
    sharpest slice is where the sheet crosses the plane: a (galvo, piezo)
    point for the fit, for free.
    """
    n = max(3, int(round(2 * galvo_half_range / step_deg)) + 1)
    vol = await _volume(
        client,
        n=n,
        exposure_ms=exposure_ms,
        galvo_amp=2 * galvo_half_range,
        galvo_c=galvo_center,
        piezo_amp=STILL,
        piezo_c=piezo,
        budget=budget,
        side=side,
    )
    if vol is None:
        return Report("extent", False, "the slice scan returned no volume", n)
    galvos = slice_positions(galvo_center, 2 * galvo_half_range, len(vol))
    signal = slice_signal(vol)
    ext = extent_from_signal(signal, galvos)
    if show:
        for i in np.linspace(0, len(vol) - 1, min(len(vol), 11)).astype(int):
            show(
                array=select_best_view(vol[i]),
                uid=f"extent_{embryo_id}_{galvos[i]:.3f}",
                data_type="edge_detection",
                metadata={
                    "embryo_id": embryo_id,
                    "galvo": float(galvos[i]),
                    "piezo": piezo,
                    "visible": bool(ext and ext[0] <= galvos[i] <= ext[1]),
                    "signal": float(signal[i]),
                },
            )
    if ext is None:
        return Report(
            "extent",
            False,
            f"nothing lit clears the floor across {n} slices at piezo {piezo:.1f} µm, "
            f"or the embryo overruns ±{galvo_half_range:.2f}°",
            n,
            {"signal_max": float(signal.max())},
        )
    top, bottom = ext
    mid = int(np.argmin(np.abs(galvos - (top + bottom) / 2)))
    if visible is not None:
        seen, score, desc = await visible(select_best_view(vol[mid]))
        if not seen:
            return Report(
                "extent",
                False,
                f"the envelope says {top:.3f}° to {bottom:.3f}° but vision sees no embryo "
                f"in the middle slice: {desc}",
                n,
                {"description": desc},
            )
    # The crossing: where, inside the lit run, the sheet is sharpest. Outside
    # it the camera sees noise, which scores high on a bandpass; masked out.
    inside = (galvos >= top) & (galvos <= bottom)
    scores = np.array(
        [calculate_focus_score(select_view_and_crop_roi(sl), "fft_bandpass") for sl in vol[inside]]
    )
    g_peak, r2, params = focus_from_scores(galvos[inside], scores)
    # A ramp fits a Gaussian too. A crossing is a peak well inside the run
    # and narrower than it; anything else is not a point on the line.
    margin = 0.1 * (bottom - top)
    peaked = (
        params is not None
        and top + margin < g_peak < bottom - margin
        and float(params[2]) < (bottom - top) / 2
    )
    if not peaked:
        r2 = 0.0
    return Report(
        "extent",
        True,
        f"embryo spans galvo {top:.3f}° to {bottom:.3f}° (~{(bottom - top) * 100:.0f} µm); "
        f"sheet crosses piezo {piezo:.1f} µm at {g_peak:.3f}° (R² {r2:.2f})",
        n,
        {
            "galvo_top": top,
            "galvo_bottom": bottom,
            "piezo": piezo,
            "galvo_at_piezo": g_peak,
            "r_squared": r2,
        },
    )


async def focus_at(
    client,
    *,
    galvo: float,
    expected_piezo: float,
    half_range_um: float = 15.0,
    step_um: float = 0.5,
    exposure_ms: float = 10.0,
    budget: Budget,
    embryo_id: str = "",
    show: Show | None = None,
    judge: Callable[[list[np.ndarray], list[float]], Awaitable[int | None]] | None = None,
    side: str = "A",
) -> Report:
    """Piezo scan only: the sheet stays at `galvo`, the detection plane steps.

    Focus score per slice, Gaussian fit, best piezo. When the fit is poor,
    `judge` (vision over the slices around the peak) gets the last word.
    """
    n = max(4, int(round(2 * half_range_um / step_um)) + 1)
    vol = await _volume(
        client,
        n=n,
        exposure_ms=exposure_ms,
        galvo_amp=STILL,
        galvo_c=galvo,
        piezo_amp=2 * half_range_um,
        piezo_c=expected_piezo,
        budget=budget,
        side=side,
    )
    if vol is None:
        return Report("focus_at", False, f"the piezo scan at {galvo:.3f}° returned no volume", n)
    piezos = slice_positions(expected_piezo, 2 * half_range_um, len(vol))
    views = [select_view_and_crop_roi(sl) for sl in vol]
    scores = np.array([calculate_focus_score(v, "fft_bandpass") for v in views])
    best, r2, params = focus_from_scores(piezos, scores)
    how = "fit"
    if judge is not None and r2 < 0.75:
        k = int(np.argmin(np.abs(piezos - best)))
        lo, hi = max(0, k - 3), min(len(vol), k + 4)
        pick = await judge(
            [select_best_view(vol[i]) for i in range(lo, hi)],
            [float(piezos[i] - best) for i in range(lo, hi)],
        )
        if pick is not None:
            best, how = float(piezos[lo + pick]), "vision"
    if show:
        for i in range(len(vol)):
            show(
                array=views[i],
                uid=f"focus_stack_{embryo_id}_{galvo:.3f}_{piezos[i]:.1f}",
                data_type="focus_sweep",
                metadata={
                    "embryo_id": embryo_id,
                    "sweep": "stack",
                    "galvo_name": f"{galvo:.3f}°",
                    "galvo": galvo,
                    "piezo": float(piezos[i]),
                    "score": float(scores[i]),
                    "r_squared": r2,
                },
            )
        try:
            from gently.ui.web.plots import generate_focus_curve_plot

            show(
                array=generate_focus_curve_plot(
                    positions=piezos,
                    scores=scores,
                    best_position=best,
                    fit_params=params,
                    r_squared=r2,
                    title=f"{embryo_id} focus at galvo {galvo:.3f}°",
                ),
                uid=f"focus_curve_{embryo_id}_{galvo:.3f}",
                data_type="focus_plot",
                metadata={
                    "embryo_id": embryo_id,
                    "galvo_name": f"{galvo:.3f}°",
                    "galvo": galvo,
                    "best_piezo": best,
                    "r_squared": r2,
                    "exposures": n,
                },
            )
        except Exception as exc:  # a missing plot is not a missing measurement
            logger.warning("focus plot for %s failed: %s", embryo_id, exc)
    edge = best <= piezos[0] + step_um or best >= piezos[-1] - step_um
    ok = r2 >= 0.5 or how == "vision"
    says = f"at galvo {galvo:.3f}° focus is sharpest at piezo {best:.1f} µm (R² {r2:.2f}, by {how})"
    if edge:
        says += "; that is the edge of the scan, so the true peak may lie beyond it"
        ok = False
    return Report(
        "focus_at",
        ok,
        says,
        n,
        {
            "galvo": galvo,
            "piezo": best,
            "r_squared": r2,
            "max_score": float(scores.max()),
            "how": how,
            "at_edge": edge,
        },
    )
