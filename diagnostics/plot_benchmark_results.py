#!/usr/bin/env python3
"""
Generate figures from Volume Scanning FPS Benchmark results.

Reads the latest (or specified) benchmark CSV and produces publication-quality
figures for the documentation report.

Usage:
    python diagnostics/plot_benchmark_results.py
    python diagnostics/plot_benchmark_results.py results/benchmark_volume_fps_20260127_123405.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_benchmark_csv(path: Path) -> dict:
    """Parse a benchmark CSV into structured data."""
    metadata = {}
    summary = []
    per_volume = []

    with open(path) as f:
        reader = csv.reader(f)
        section = "metadata"

        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue

            # Metadata comment rows
            if row[0].startswith("#"):
                key = row[0].lstrip("# ").strip()
                if key == "Per-volume timings (seconds)":
                    section = "per_volume"
                    continue
                if len(row) > 1:
                    metadata[key] = row[1].strip()
                continue

            # Header rows
            if row[0] == "slices":
                if section == "per_volume":
                    continue  # skip header
                section = "summary"
                continue

            if section == "summary":
                summary.append(
                    {
                        "slices": int(row[0]),
                        "exposure_ms": float(row[1]),
                        "approach": row[2],
                        "vol_per_sec": float(row[3]) if row[3] else None,
                        "mean_s": float(row[4]) if row[4] else None,
                        "std_s": float(row[5]) if row[5] else None,
                        "min_s": float(row[6]) if row[6] else None,
                        "max_s": float(row[7]) if row[7] else None,
                        "total_images": int(row[8]) if row[8] else 0,
                        "num_repeats": int(row[9]) if row[9] else 0,
                    }
                )
            elif section == "per_volume":
                per_volume.append(
                    {
                        "slices": int(row[0]),
                        "exposure_ms": float(row[1]),
                        "approach": row[2],
                        "repeat": int(row[3]),
                        "elapsed_s": float(row[4]),
                        "image_count": int(row[5]),
                    }
                )

    return {"metadata": metadata, "summary": summary, "per_volume": per_volume}


# ---------------------------------------------------------------------------
# Figure 1: Volume throughput by approach
# ---------------------------------------------------------------------------
def plot_throughput(data: dict, output_dir: Path):
    """Bar chart of vol/s for each approach at each slice count."""
    summary = data["summary"]

    # Group by slices
    slices_set = sorted(set(r["slices"] for r in summary))
    approaches = []
    for r in summary:
        if r["approach"] not in approaches:
            approaches.append(r["approach"])

    # Build matrix
    vps: dict = defaultdict(dict)
    for r in summary:
        if r["vol_per_sec"] is not None:
            vps[r["slices"]][r["approach"]] = r["vol_per_sec"]

    # Colors and labels
    color_map = {
        "raw": "#2563eb",
        "ophyd": "#dc2626",
        "ophyd_burst": "#16a34a",
        "burst_reconfig": "#ea580c",
        "reconfig_wfd": "#7c3aed",
    }
    label_map = {
        "raw": "Raw MMCore",
        "ophyd": "Ophyd (full)",
        "ophyd_burst": "Ophyd burst",
        "burst_reconfig": "Reconfig (sleep)",
        "reconfig_wfd": "Reconfig (waitForDevice)",
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(slices_set))
    n = len(approaches)
    width = 0.72 / n

    for i, approach in enumerate(approaches):
        vals = [vps[s].get(approach, 0) for s in slices_set]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.92,
            label=label_map.get(approach, approach),
            color=color_map.get(approach, "#888"),
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, v in zip(bars, vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    ax.set_xlabel("Slices per volume", fontsize=11)
    ax.set_ylabel("Volumes per second", fontsize=11)
    ax.set_title("Volume Acquisition Throughput by Approach", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(slices_set)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_throughput.png", dpi=180)
    plt.close(fig)
    print("  Saved: benchmark_throughput.png")


# ---------------------------------------------------------------------------
# Figure 2: Overhead vs raw
# ---------------------------------------------------------------------------
def plot_overhead(data: dict, output_dir: Path):
    """Grouped bar chart showing overhead (ms) vs raw for each approach."""
    summary = data["summary"]

    slices_set = sorted(set(r["slices"] for r in summary))
    raw_means = {}
    for r in summary:
        if r["approach"] == "raw" and r["mean_s"] is not None:
            raw_means[r["slices"]] = r["mean_s"]

    compare = ["ophyd", "burst_reconfig", "reconfig_wfd"]
    color_map = {
        "ophyd": "#dc2626",
        "burst_reconfig": "#ea580c",
        "reconfig_wfd": "#7c3aed",
    }
    label_map = {
        "ophyd": "Ophyd (full teardown/setup)",
        "burst_reconfig": "Reconfig (time.sleep)",
        "reconfig_wfd": "Reconfig (waitForDevice)",
    }

    overhead: dict = defaultdict(dict)
    for r in summary:
        if r["approach"] in compare and r["mean_s"] is not None:
            s = r["slices"]
            if s in raw_means:
                overhead[s][r["approach"]] = (r["mean_s"] - raw_means[s]) * 1000

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(slices_set))
    n = len(compare)
    width = 0.72 / n

    for i, approach in enumerate(compare):
        vals = [overhead[s].get(approach, 0) for s in slices_set]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.92,
            label=label_map.get(approach, approach),
            color=color_map.get(approach, "#888"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, v in zip(bars, vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 8,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Slices per volume", fontsize=11)
    ax.set_ylabel("Overhead vs raw MMCore (ms)", fontsize=11)
    ax.set_title("Per-Volume Overhead by Approach", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(slices_set)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_overhead.png", dpi=180)
    plt.close(fig)
    print("  Saved: benchmark_overhead.png")


# ---------------------------------------------------------------------------
# Figure 3: waitForDevice savings (sleep vs wfd side-by-side)
# ---------------------------------------------------------------------------
def plot_wfd_savings(data: dict, output_dir: Path):
    """Side-by-side comparison of burst_reconfig vs reconfig_wfd,
    broken down into acquisition time and overhead."""
    summary = data["summary"]

    slices_set = sorted(set(r["slices"] for r in summary))
    means: dict = defaultdict(dict)
    raw_means = {}
    for r in summary:
        if r["mean_s"] is not None:
            means[r["slices"]][r["approach"]] = r["mean_s"]
            if r["approach"] == "raw":
                raw_means[r["slices"]] = r["mean_s"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stacked bar showing acquisition vs overhead
    ax = axes[0]
    x = np.arange(len(slices_set))
    width = 0.32

    for i, (approach, label, color) in enumerate(
        [
            ("burst_reconfig", "sleep()", "#ea580c"),
            ("reconfig_wfd", "waitForDevice()", "#7c3aed"),
        ]
    ):
        acq_times = [raw_means.get(s, 0) for s in slices_set]
        overheads = [means[s].get(approach, 0) - raw_means.get(s, 0) for s in slices_set]

        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            acq_times,
            width * 0.92,
            color="#93c5fd",
            edgecolor="white",
            linewidth=0.5,
            label="Acquisition time" if i == 0 else None,
        )
        ax.bar(
            x + offset,
            overheads,
            width * 0.92,
            bottom=acq_times,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=f"Overhead ({label})",
        )

    ax.set_xlabel("Slices per volume", fontsize=11)
    ax.set_ylabel("Total time per volume (s)", fontsize=11)
    ax.set_title("Time Breakdown: Acquisition vs Overhead", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(slices_set)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: savings chart
    ax2 = axes[1]
    sleep_overhead = []
    wfd_overhead = []
    savings = []
    for s in slices_set:
        so = (means[s].get("burst_reconfig", 0) - raw_means.get(s, 0)) * 1000
        wo = (means[s].get("reconfig_wfd", 0) - raw_means.get(s, 0)) * 1000
        sleep_overhead.append(so)
        wfd_overhead.append(wo)
        savings.append(so - wo)

    bar_width = 0.55
    bars_sleep = ax2.barh(
        x + 0.15,
        sleep_overhead,
        bar_width * 0.48,
        color="#ea580c",
        label="time.sleep() overhead",
    )
    bars_wfd = ax2.barh(
        x - 0.15,
        wfd_overhead,
        bar_width * 0.48,
        color="#7c3aed",
        label="waitForDevice() overhead",
    )

    for bar, val, _sav in zip(bars_sleep, sleep_overhead, savings, strict=False):
        ax2.text(
            bar.get_width() + 8,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}ms",
            va="center",
            fontsize=9,
            color="#ea580c",
            fontweight="bold",
        )
    for bar, val in zip(bars_wfd, wfd_overhead, strict=False):
        ax2.text(
            bar.get_width() + 8,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}ms",
            va="center",
            fontsize=9,
            color="#7c3aed",
            fontweight="bold",
        )

    # Add savings annotation
    for i, (_s, sav) in enumerate(zip(slices_set, savings, strict=False)):
        ax2.annotate(
            f"-{sav:.0f}ms",
            xy=(sleep_overhead[i], i + 0.15),
            xytext=(sleep_overhead[i] + 60, i + 0.35),
            fontsize=8.5,
            fontweight="bold",
            color="#166534",
            arrowprops=dict(arrowstyle="->", color="#166534", lw=1.2),
        )

    ax2.set_yticks(x)
    ax2.set_yticklabels([f"{s} slices" for s in slices_set])
    ax2.set_xlabel("Overhead per volume (ms)", fontsize=11)
    ax2.set_title("waitForDevice() Saves ~520ms per Volume", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="x", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_wfd_savings.png", dpi=180)
    plt.close(fig)
    print("  Saved: benchmark_wfd_savings.png")


# ---------------------------------------------------------------------------
# Figure 4: Per-volume timing consistency (box plot)
# ---------------------------------------------------------------------------
def plot_consistency(data: dict, output_dir: Path):
    """Box plot showing per-volume timing consistency for each approach."""
    per_volume = data["per_volume"]

    approaches_order = ["raw", "ophyd_burst", "reconfig_wfd", "burst_reconfig", "ophyd"]
    label_map = {
        "raw": "Raw\nMMCore",
        "ophyd": "Ophyd\n(full)",
        "ophyd_burst": "Ophyd\nburst",
        "burst_reconfig": "Reconfig\n(sleep)",
        "reconfig_wfd": "Reconfig\n(wfd)",
    }
    color_map = {
        "raw": "#2563eb",
        "ophyd": "#dc2626",
        "ophyd_burst": "#16a34a",
        "burst_reconfig": "#ea580c",
        "reconfig_wfd": "#7c3aed",
    }

    slices_set = sorted(set(r["slices"] for r in per_volume))

    fig, axes = plt.subplots(1, len(slices_set), figsize=(14, 4.5), sharey=False)
    if len(slices_set) == 1:
        axes = [axes]

    for ax, ns in zip(axes, slices_set, strict=False):
        box_data = []
        labels = []
        colors = []
        for approach in approaches_order:
            timings = [
                r["elapsed_s"]
                for r in per_volume
                if r["slices"] == ns and r["approach"] == approach
            ]
            if timings:
                box_data.append(timings)
                labels.append(label_map.get(approach, approach))
                colors.append(color_map.get(approach, "#888"))

        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
        )
        for patch, c in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(f"{ns} slices", fontsize=11, fontweight="bold")
        ax.set_ylabel("Time per volume (s)" if ax == axes[0] else "", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Per-Volume Timing Consistency", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_consistency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: benchmark_consistency.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Find CSV
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        csvs = sorted(results_dir.glob("benchmark_volume_fps_*.csv"))
        if not csvs:
            print("No benchmark CSV files found in results/")
            sys.exit(1)
        csv_path = csvs[-1]  # latest

    print(f"Reading: {csv_path}")
    data = parse_benchmark_csv(csv_path)
    print(f"  {len(data['summary'])} summary rows, {len(data['per_volume'])} per-volume rows")

    output_dir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures in: {output_dir}")
    plot_throughput(data, output_dir)
    plot_overhead(data, output_dir)
    plot_wfd_savings(data, output_dir)
    plot_consistency(data, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
