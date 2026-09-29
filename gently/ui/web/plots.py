"""
Plot generation utilities for visualization server.

All functions return numpy arrays (RGB images) suitable for push_image().
Uses matplotlib with Agg backend for thread safety.
"""

import functools
import math
from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# A figure is looked at in two places: small, beside the controls that made
# it, and large, when it is opened. It was drawn 600 px wide with 9 to 12 pt
# type, and shown 126 px wide, where that type is three pixels tall.
#
# So the type is large for the figure (3% of its width and more, which is
# 8 px when the figure is shown 260 px wide), and the figure is drawn with
# enough pixels to be opened full screen.
FIGSIZE: tuple[float, float] = (5.0, 3.4)
DPI = 220

STYLE: dict[str, object] = {
    "font.size": 13,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.0,
    "lines.linewidth": 2.2,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#0f172a",
    "text.color": "#0f172a",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
}

# The least type in a figure: annotations. Nothing is drawn smaller.
SMALL = 11


def _styled(draw):
    """Draw the whole figure in ``STYLE``.

    The whole of it: matplotlib reads a size when the text is made, so a
    title set outside the style is a title in the default size, whatever
    the axes were created with.
    """

    @functools.wraps(draw)
    def styled(*args, **kwargs):
        with plt.rc_context(STYLE):
            return draw(*args, **kwargs)

    return styled


def _figure(figsize: tuple[float, float] | None, dpi: int | None):
    """A figure and its axes, at the size every plot here is drawn at."""
    return plt.subplots(figsize=figsize or FIGSIZE, dpi=dpi or DPI)


def _finish(fig) -> np.ndarray:
    """The figure as an RGB array, and the figure closed."""
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())
    plt.close(fig)
    return buf[:, :, :3].astype(np.uint8)


def _thousands(values: np.ndarray) -> tuple[float, str]:
    """A divisor for an axis of large numbers, and how to say it in the label.

    matplotlib writes the multiplier ("1e6") above the axis, where it runs
    into the title. It is said in the axis label instead.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    top = float(np.max(np.abs(finite))) if finite.size else 0.0
    if top < 1e4:
        return 1.0, ""
    power = 3 * int(math.floor(math.log10(top) / 3))
    return 10.0**power, f" (×10$^{{{power}}}$)"


@_styled
def generate_focus_curve_plot(
    positions: np.ndarray,
    scores: np.ndarray,
    best_position: float,
    fit_params: np.ndarray | None = None,
    r_squared: float = 0.0,
    title: str = "Focus Curve",
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
) -> np.ndarray:
    """
    Generate focus curve plot as RGB numpy array.

    Parameters
    ----------
    positions : np.ndarray
        Piezo positions in micrometers
    scores : np.ndarray
        Focus scores at each position
    best_position : float
        Optimal focus position (piezo value)
    fit_params : np.ndarray, optional
        Gaussian fit parameters [amplitude, mean, sigma, offset]
    r_squared : float
        Fit quality (coefficient of determination)
    title : str
        Plot title
    figsize : tuple, optional
        Figure size in inches (width, height). ``FIGSIZE`` if not given.
    dpi : int, optional
        Resolution in dots per inch. ``DPI`` if not given.

    Returns
    -------
    np.ndarray
        RGB image array (H, W, 3), dtype uint8
    """
    fig, ax = _figure(figsize, dpi)
    positions = np.asarray(positions, dtype=float)
    scores = np.asarray(scores, dtype=float)
    unit, unit_label = _thousands(scores)

    # Data points
    ax.scatter(positions, scores / unit, c="#2196F3", s=70, zorder=3, label="Measurements")

    # Gaussian fit curve
    if fit_params is not None and len(fit_params) >= 4:
        a, mu, sigma, c = fit_params[:4]
        x_fit = np.linspace(positions.min(), positions.max(), 200)
        y_fit = a * np.exp(-((x_fit - mu) ** 2) / (2 * sigma**2)) + c
        ax.plot(
            x_fit,
            y_fit / unit,
            color="#F44336",
            linewidth=2,
            label=f"Gaussian fit (R²={r_squared:.3f})",
        )

    # Best position marker
    ax.axvline(
        best_position,
        color="#4CAF50",
        linestyle="--",
        linewidth=2,
        label=f"Best: {best_position:.2f} µm",
    )

    ax.set_xlabel("Piezo position (µm)")
    ax.set_ylabel(f"Focus score{unit_label}")
    ax.set_title(title)
    # Wherever it covers least: a focus curve peaks in the middle, and the
    # upper right corner is often where its shoulder is.
    ax.legend(loc="best", framealpha=0.92)
    ax.grid(True, alpha=0.3)

    return _finish(fig)


@_styled
def generate_calibration_summary_plot(
    embryo_id: str,
    galvo_top: float,
    galvo_bottom: float,
    piezo_top: float,
    piezo_bottom: float,
    slope: float,
    offset: float,
    r_squared_top: float = 0.0,
    r_squared_bottom: float = 0.0,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
) -> np.ndarray:
    """
    Generate calibration summary plot showing piezo-galvo relationship.

    Parameters
    ----------
    embryo_id : str
        Embryo identifier for title
    galvo_top : float
        Galvo position at top calibration point (degrees)
    galvo_bottom : float
        Galvo position at bottom calibration point (degrees)
    piezo_top : float
        Piezo position at top calibration point (micrometers)
    piezo_bottom : float
        Piezo position at bottom calibration point (micrometers)
    slope : float
        Linear fit slope (µm/deg)
    offset : float
        Linear fit offset (µm)
    r_squared_top : float
        Fit quality at top calibration point
    r_squared_bottom : float
        Fit quality at bottom calibration point
    figsize : tuple
        Figure size in inches
    dpi : int
        Resolution

    Returns
    -------
    np.ndarray
        RGB image array (H, W, 3), dtype uint8
    """
    fig, ax = _figure(figsize, dpi)

    # Calibration points
    galvos = [galvo_top, galvo_bottom]
    piezos = [piezo_top, piezo_bottom]
    ax.scatter(galvos, piezos, c="#2196F3", s=100, zorder=3, label="Calibration points")

    # Linear fit line
    margin = 0.05
    galvo_range = np.linspace(
        min(galvo_top, galvo_bottom) - margin,
        max(galvo_top, galvo_bottom) + margin,
        100,
    )
    piezo_fit = slope * galvo_range + offset
    ax.plot(
        galvo_range,
        piezo_fit,
        color="#F44336",
        linewidth=2,
        label=f"piezo = {slope:.1f}·galvo + {offset:.1f}",
    )

    # Annotations, on the side of each point the line does not run through:
    # below and right of it for a rising line, above and right for a falling one.
    away = (12, -12) if slope >= 0 else (12, 12)
    for name, galvo, piezo, r2 in (
        ("Top", galvo_top, piezo_top, r_squared_top),
        ("Bottom", galvo_bottom, piezo_bottom, r_squared_bottom),
    ):
        ax.annotate(
            f"{name}  R²={r2:.3f}",
            (galvo, piezo),
            textcoords="offset points",
            xytext=away,
            ha="left",
            va="top" if slope >= 0 else "bottom",
            fontsize=SMALL,
            color="#334155",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

    # Room beside the points for what is written beside them.
    ax.margins(x=0.22, y=0.12)

    # Room beside the points for what is written beside them.
    ax.margins(x=0.22, y=0.12)

    ax.set_xlabel("Galvo position (degrees)")
    ax.set_ylabel("Piezo position (µm)")
    ax.set_title(f"{embryo_id} - Piezo-Galvo Calibration")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return _finish(fig)


@_styled
def generate_edge_detection_plot(
    galvo_positions: list[float],
    visibility: list[bool],
    edge_top: float | None = None,
    edge_bottom: float | None = None,
    embryo_id: str = "embryo",
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
) -> np.ndarray:
    """
    Generate edge detection summary plot.

    Parameters
    ----------
    galvo_positions : list of float
        Galvo positions tested (degrees)
    visibility : list of bool
        Whether embryo was visible at each position
    edge_top : float, optional
        Detected top edge position
    edge_bottom : float, optional
        Detected bottom edge position
    embryo_id : str
        Embryo identifier for title
    figsize : tuple
        Figure size
    dpi : int
        Resolution

    Returns
    -------
    np.ndarray
        RGB image array (H, W, 3), dtype uint8
    """
    fig, ax = _figure(figsize, dpi)

    # Convert visibility to numeric for plotting
    vis_numeric = [1 if v else 0 for v in visibility]

    # Plot visibility as step function
    colors = ["#4CAF50" if v else "#F44336" for v in visibility]
    ax.scatter(galvo_positions, vis_numeric, c=colors, s=80, zorder=3)

    # Draw step-like connecting lines
    for i in range(len(galvo_positions) - 1):
        color = "#4CAF50" if visibility[i] else "#F44336"
        ax.hlines(
            vis_numeric[i],
            galvo_positions[i],
            galvo_positions[i + 1],
            color=color,
            alpha=0.3,
            linewidth=2,
        )

    # Mark edges if provided
    if edge_top is not None:
        ax.axvline(
            edge_top,
            color="#2196F3",
            linestyle="--",
            linewidth=2,
            label=f"Top edge: {edge_top:.3f}°",
        )
    if edge_bottom is not None:
        ax.axvline(
            edge_bottom,
            color="#FF9800",
            linestyle="--",
            linewidth=2,
            label=f"Bottom edge: {edge_bottom:.3f}°",
        )

    ax.set_xlabel("Galvo position (degrees)")
    ax.set_ylabel("Embryo visible")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_title(f"{embryo_id} - Edge Detection")
    if edge_top is not None or edge_bottom is not None:
        ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="x")

    return _finish(fig)
