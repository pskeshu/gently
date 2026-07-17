"""
Plot generation utilities for visualization server.

All functions return numpy arrays (RGB images) suitable for push_image().
Uses matplotlib with Agg backend for thread safety.
"""

from typing import cast

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def generate_focus_curve_plot(
    positions: np.ndarray,
    scores: np.ndarray,
    best_position: float,
    fit_params: np.ndarray | None = None,
    r_squared: float = 0.0,
    title: str = "Focus Curve",
    figsize: tuple[int, int] = (6, 4),
    dpi: int = 100,
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
    figsize : tuple
        Figure size in inches (width, height)
    dpi : int
        Resolution in dots per inch

    Returns
    -------
    np.ndarray
        RGB image array (H, W, 3), dtype uint8
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Data points
    ax.scatter(positions, scores, c="#2196F3", s=50, zorder=3, label="Measurements")

    # Gaussian fit curve
    if fit_params is not None and len(fit_params) >= 4:
        a, mu, sigma, c = fit_params[:4]
        x_fit = np.linspace(positions.min(), positions.max(), 200)
        y_fit = a * np.exp(-((x_fit - mu) ** 2) / (2 * sigma**2)) + c
        ax.plot(
            x_fit,
            y_fit,
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

    ax.set_xlabel("Piezo Position (µm)", fontsize=11)
    ax.set_ylabel("Focus Score", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Tight layout
    fig.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    buf = np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())
    plt.close(fig)

    return buf[:, :, :3].astype(np.uint8)


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
    figsize: tuple[int, int] = (7, 5),
    dpi: int = 100,
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
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

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
        label=f"Linear fit: piezo = {slope:.1f}·galvo + {offset:.1f}",
    )

    # Annotations
    ax.annotate(
        f"Top\nR²={r_squared_top:.3f}",
        (galvo_top, piezo_top),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
        color="#666",
    )
    ax.annotate(
        f"Bottom\nR²={r_squared_bottom:.3f}",
        (galvo_bottom, piezo_bottom),
        textcoords="offset points",
        xytext=(10, -20),
        fontsize=9,
        color="#666",
    )

    ax.set_xlabel("Galvo Position (degrees)", fontsize=11)
    ax.set_ylabel("Piezo Position (µm)", fontsize=11)
    ax.set_title(f"{embryo_id} - Piezo-Galvo Calibration", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())
    plt.close(fig)

    return buf[:, :, :3].astype(np.uint8)


def generate_edge_detection_plot(
    galvo_positions: list[float],
    visibility: list[bool],
    edge_top: float | None = None,
    edge_bottom: float | None = None,
    embryo_id: str = "embryo",
    figsize: tuple[int, int] = (6, 4),
    dpi: int = 100,
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
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

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

    ax.set_xlabel("Galvo Position (degrees)", fontsize=11)
    ax.set_ylabel("Embryo Visible", fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No", "Yes"])
    ax.set_title(f"{embryo_id} - Edge Detection", fontsize=12, fontweight="bold")
    if edge_top is not None or edge_bottom is not None:
        ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(cast(FigureCanvasAgg, fig.canvas).buffer_rgba())
    plt.close(fig)

    return buf[:, :, :3].astype(np.uint8)
