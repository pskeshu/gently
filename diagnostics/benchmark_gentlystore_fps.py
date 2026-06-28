#!/usr/bin/env python3
"""
FileStore Volume Storage Benchmark
==================================

Benchmarks volume storage throughput through the FileStore system,
measuring the overhead of:
- TIFF writing (with zlib compression)
- JPEG projection generation
- File-based storage writes

This isolates the storage layer from hardware acquisition.

Usage:
    python diagnostics/benchmark_gentlystore_fps.py
    python diagnostics/benchmark_gentlystore_fps.py --slices 50 100 --repeats 10
    python diagnostics/benchmark_gentlystore_fps.py --width 2048 --height 512
    python diagnostics/benchmark_gentlystore_fps.py --no-projection  # skip projection
    python diagnostics/benchmark_gentlystore_fps.py --compare-compression
"""

import argparse
import shutil
import statistics

# Add gently to path
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

GENTLY_ROOT = Path(__file__).resolve().parent.parent
if str(GENTLY_ROOT) not in sys.path:
    sys.path.insert(0, str(GENTLY_ROOT))

from gently.core.file_store import FileStore  # noqa: E402

# ---------------------------------------------------------------------------
# Default parameters -- typical diSPIM volume dimensions
# ---------------------------------------------------------------------------
DEFAULT_SLICES = [25, 50, 100, 200]
DEFAULT_WIDTH = 2048
DEFAULT_HEIGHT = 512
DEFAULT_DTYPE = np.uint16
NUM_REPEATS = 10
NUM_WARMUP = 2
NUM_EMBRYOS = 4  # Simulate multi-embryo round-robin


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    approach: str
    num_slices: int
    volume_shape: tuple
    timings: list[float] = field(default_factory=list)
    sizes_mb: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.timings) if self.timings else float("nan")

    @property
    def std(self) -> float:
        return statistics.stdev(self.timings) if len(self.timings) > 1 else 0.0

    @property
    def min_t(self) -> float:
        return min(self.timings) if self.timings else float("nan")

    @property
    def max_t(self) -> float:
        return max(self.timings) if self.timings else float("nan")

    @property
    def vol_per_sec(self) -> float:
        return 1.0 / self.mean if self.mean > 0 else 0.0

    @property
    def avg_size_mb(self) -> float:
        return statistics.mean(self.sizes_mb) if self.sizes_mb else 0.0

    @property
    def mb_per_sec(self) -> float:
        if self.mean > 0 and self.sizes_mb:
            return self.avg_size_mb / self.mean
        return 0.0


# ---------------------------------------------------------------------------
# Volume generation
# ---------------------------------------------------------------------------
def generate_synthetic_volume(
    num_slices: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    dtype=DEFAULT_DTYPE,
    pattern: str = "noise",
) -> np.ndarray:
    """
    Generate a synthetic volume for benchmarking.

    Parameters
    ----------
    pattern : str
        'noise' - random noise (incompressible, worst-case)
        'gradient' - smooth gradient (compressible)
        'embryo' - simulated embryo pattern (realistic compression)
    """
    shape = (num_slices, height, width)

    if pattern == "noise":
        # Random noise - worst case for compression
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.random.randint(info.min, info.max, shape, dtype=dtype)
        else:
            return np.random.random(shape).astype(dtype)

    elif pattern == "gradient":
        # Smooth gradient - best case for compression
        z = np.linspace(0, 1, num_slices)[:, None, None]
        y = np.linspace(0, 1, height)[None, :, None]
        x = np.linspace(0, 1, width)[None, None, :]
        vol = z * 0.3 + y * 0.3 + x * 0.4
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            vol = (vol * info.max).astype(dtype)
        return vol.astype(dtype)

    elif pattern == "embryo":
        # Simulated embryo: sparse bright spots in dark background
        # More realistic compression characteristics
        vol = np.zeros(shape, dtype=np.float32)

        # Add some Gaussian blobs (nuclei)
        n_nuclei = num_slices * 2
        for _ in range(n_nuclei):
            cz = np.random.randint(0, num_slices)
            cy = np.random.randint(height // 4, 3 * height // 4)
            cx = np.random.randint(width // 4, 3 * width // 4)
            sz, sy, sx = 3, 15, 15

            z_idx = np.clip(np.arange(cz - sz * 2, cz + sz * 2), 0, num_slices - 1)
            y_idx = np.clip(np.arange(cy - sy * 2, cy + sy * 2), 0, height - 1)
            x_idx = np.clip(np.arange(cx - sx * 2, cx + sx * 2), 0, width - 1)

            zz, yy, xx = np.meshgrid(z_idx, y_idx, x_idx, indexing="ij")
            d2 = ((zz - cz) / sz) ** 2 + ((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2
            blob = np.exp(-d2 / 2)
            vol[
                z_idx[0] : z_idx[-1] + 1,
                y_idx[0] : y_idx[-1] + 1,
                x_idx[0] : x_idx[-1] + 1,
            ] += blob

        # Add background noise
        vol += np.random.random(shape) * 0.1

        # Normalize and convert
        vol = np.clip(vol, 0, 1)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            vol = (vol * info.max).astype(dtype)
        return vol.astype(dtype)

    else:
        raise ValueError(f"Unknown pattern: {pattern}")


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------
def benchmark_raw_tiff_write(
    volume: np.ndarray,
    output_dir: Path,
    compression: str | None = "zlib",
) -> tuple[float, float]:
    """
    Benchmark raw tifffile write (no FileStore).

    Returns (elapsed_seconds, file_size_mb).
    """
    import tifffile

    output_path = output_dir / f"raw_{time.perf_counter_ns()}.tif"

    t0 = time.perf_counter()
    if compression:
        tifffile.imwrite(str(output_path), volume, compression=compression)
    else:
        tifffile.imwrite(str(output_path), volume)
    elapsed = time.perf_counter() - t0

    size_mb = output_path.stat().st_size / (1024 * 1024)
    output_path.unlink()  # Clean up

    return elapsed, size_mb


def benchmark_put_volume(
    store: FileStore,
    session_id: str,
    embryo_id: str,
    timepoint: int,
    volume: np.ndarray,
) -> tuple[float, float]:
    """
    Benchmark FileStore.put_volume() - full pipeline.

    Returns (elapsed_seconds, file_size_mb).
    """
    t0 = time.perf_counter()
    path = store.put_volume(session_id, embryo_id, timepoint, volume)
    elapsed = time.perf_counter() - t0

    size_mb = path.stat().st_size / (1024 * 1024)
    return elapsed, size_mb


def benchmark_register_volume(
    store: FileStore,
    session_id: str,
    embryo_id: str,
    timepoint: int,
    volume: np.ndarray,
) -> tuple[float, float]:
    """
    Benchmark FileStore.register_volume() - zero-copy path.

    This simulates the device layer writing a TIFF, then FileStore
    moving it to canonical location.

    Returns (elapsed_seconds, file_size_mb).
    """
    import tifffile

    # Simulate device layer writing to incoming/
    incoming_path = store.incoming_dir / f"incoming_{time.perf_counter_ns()}.tif"
    tifffile.imwrite(str(incoming_path), volume)  # No compression (device layer)

    t0 = time.perf_counter()
    path = store.register_volume(session_id, embryo_id, timepoint, incoming_path)
    elapsed = time.perf_counter() - t0

    size_mb = path.stat().st_size / (1024 * 1024)
    return elapsed, size_mb


# ---------------------------------------------------------------------------
# Main benchmark sweep
# ---------------------------------------------------------------------------
def run_benchmark_sweep(
    slices_list: list[int],
    width: int,
    height: int,
    num_repeats: int,
    num_warmup: int,
    pattern: str = "embryo",
    run_raw: bool = True,
    run_put_volume: bool = True,
    run_register: bool = True,
    skip_projection: bool = False,
) -> list[BenchmarkResult]:
    """Run the full benchmark sweep."""
    results: list[BenchmarkResult] = []

    # Create temporary directory for benchmark
    temp_dir = Path(tempfile.mkdtemp(prefix="gently_benchmark_"))
    print(f"Benchmark temp directory: {temp_dir}")

    try:
        # Initialize FileStore
        store = FileStore(temp_dir / "store")
        session_id = "benchmark_session"
        store.create_session(session_id, name="FPS Benchmark")

        # Register embryos
        for i in range(NUM_EMBRYOS):
            store.register_embryo(session_id, f"embryo_{i}")

        # Raw TIFF output directory
        raw_dir = temp_dir / "raw"
        raw_dir.mkdir()

        total_configs = len(slices_list)
        timepoint = 0

        for config_idx, num_slices in enumerate(slices_list):
            print(f"\n{'=' * 60}")
            print(
                f"Config {config_idx + 1}/{total_configs}: "
                f"slices={num_slices}, shape=({num_slices}, {height}, {width})"
            )
            print(f"{'=' * 60}")

            volume_shape = (num_slices, height, width)

            # Generate test volume
            print(f"Generating {pattern} volume...", end=" ", flush=True)
            volume = generate_synthetic_volume(num_slices, width, height, pattern=pattern)
            raw_size_mb = volume.nbytes / (1024 * 1024)
            print(f"{raw_size_mb:.1f} MB raw")

            # --- Raw TIFF write (baseline) ---
            if run_raw:
                res_raw = BenchmarkResult("raw_tiff_zlib", num_slices, volume_shape)
                print("\n[Raw TIFF zlib]")

                # Warmup
                for w in range(num_warmup):
                    print(f"  Warmup {w + 1}/{num_warmup}...", end=" ", flush=True)
                    dur, size = benchmark_raw_tiff_write(volume, raw_dir, "zlib")
                    print(f"{dur:.3f}s, {size:.1f} MB")

                # Timed repeats
                for r in range(num_repeats):
                    print(f"  Repeat {r + 1}/{num_repeats}...", end=" ", flush=True)
                    try:
                        dur, size = benchmark_raw_tiff_write(volume, raw_dir, "zlib")
                        res_raw.timings.append(dur)
                        res_raw.sizes_mb.append(size)
                        print(f"{dur:.3f}s, {size:.1f} MB ({1 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_raw.errors.append(str(e))

                results.append(res_raw)
                _print_single_result(res_raw)

            # --- put_volume (full pipeline) ---
            if run_put_volume:
                res_put = BenchmarkResult("put_volume", num_slices, volume_shape)
                print("\n[FileStore.put_volume]")

                # Warmup
                for w in range(num_warmup):
                    embryo_id = f"embryo_{w % NUM_EMBRYOS}"
                    print(
                        f"  Warmup {w + 1}/{num_warmup} ({embryo_id})...",
                        end=" ",
                        flush=True,
                    )
                    dur, size = benchmark_put_volume(
                        store, session_id, embryo_id, timepoint, volume
                    )
                    timepoint += 1
                    print(f"{dur:.3f}s, {size:.1f} MB")

                # Timed repeats
                for r in range(num_repeats):
                    embryo_id = f"embryo_{r % NUM_EMBRYOS}"
                    print(
                        f"  Repeat {r + 1}/{num_repeats} ({embryo_id})...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        dur, size = benchmark_put_volume(
                            store, session_id, embryo_id, timepoint, volume
                        )
                        timepoint += 1
                        res_put.timings.append(dur)
                        res_put.sizes_mb.append(size)
                        print(f"{dur:.3f}s, {size:.1f} MB ({1 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_put.errors.append(str(e))

                results.append(res_put)
                _print_single_result(res_put)

            # --- register_volume (zero-copy path) ---
            if run_register:
                res_reg = BenchmarkResult("register_volume", num_slices, volume_shape)
                print("\n[FileStore.register_volume]")

                # Warmup
                for w in range(num_warmup):
                    embryo_id = f"embryo_{w % NUM_EMBRYOS}"
                    print(
                        f"  Warmup {w + 1}/{num_warmup} ({embryo_id})...",
                        end=" ",
                        flush=True,
                    )
                    dur, size = benchmark_register_volume(
                        store, session_id, embryo_id, timepoint, volume
                    )
                    timepoint += 1
                    print(f"{dur:.3f}s, {size:.1f} MB")

                # Timed repeats
                for r in range(num_repeats):
                    embryo_id = f"embryo_{r % NUM_EMBRYOS}"
                    print(
                        f"  Repeat {r + 1}/{num_repeats} ({embryo_id})...",
                        end=" ",
                        flush=True,
                    )
                    try:
                        dur, size = benchmark_register_volume(
                            store, session_id, embryo_id, timepoint, volume
                        )
                        timepoint += 1
                        res_reg.timings.append(dur)
                        res_reg.sizes_mb.append(size)
                        print(f"{dur:.3f}s, {size:.1f} MB ({1 / dur:.1f} vol/s)")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        res_reg.errors.append(str(e))

                results.append(res_reg)
                _print_single_result(res_reg)

        # Print final stats
        print(f"\nFileStore stats: {store.stats()}")
        store.close()

    finally:
        # Cleanup
        print(f"\nCleaning up {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def _print_single_result(res: BenchmarkResult):
    """Print a single intermediate result."""
    if not res.timings:
        print(f"  -> {res.approach}: NO SUCCESSFUL RUNS ({len(res.errors)} errors)")
        return
    print(
        f"  -> {res.approach}: {res.vol_per_sec:.2f} vol/s, "
        f"mean={res.mean:.3f}s, std={res.std:.3f}s, "
        f"avg_size={res.avg_size_mb:.1f} MB"
    )


def print_results_table(results: list[BenchmarkResult]):
    """Print formatted ASCII results table."""
    if not results:
        print("No results to display.")
        return

    header = (
        f"{'Slices':>6} | {'Approach':>18} | {'Vol/s':>7} | "
        f"{'Mean(s)':>7} | {'Std(s)':>6} | {'Min(s)':>6} | "
        f"{'Max(s)':>6} | {'Size(MB)':>8} | {'MB/s':>7}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("RESULTS")
    print(sep)
    print(header)
    print(sep)

    for r in results:
        if r.timings:
            print(
                f"{r.num_slices:>6} | {r.approach:>18} | "
                f"{r.vol_per_sec:>7.2f} | {r.mean:>7.3f} | {r.std:>6.3f} | "
                f"{r.min_t:>6.3f} | {r.max_t:>6.3f} | {r.avg_size_mb:>8.1f} | "
                f"{r.mb_per_sec:>7.1f}"
            )
        else:
            print(
                f"{r.num_slices:>6} | {r.approach:>18} | "
                f"{'FAIL':>7} | {'---':>7} | {'---':>6} | "
                f"{'---':>6} | {'---':>6} | {'---':>8} | {'---':>7}"
            )

    print(sep)


def print_overhead_analysis(results: list[BenchmarkResult]):
    """Print overhead analysis comparing FileStore to raw TIFF."""
    from collections import defaultdict

    groups: dict = defaultdict(dict)
    for r in results:
        groups[r.num_slices][r.approach] = r

    has_data = [
        (k, v)
        for k, v in groups.items()
        if "raw_tiff_zlib" in v and ("put_volume" in v or "register_volume" in v)
    ]

    if not has_data:
        return

    print()
    header = f"{'Slices':>6} | {'Approach':>18} | {'Overhead(ms)':>12} | {'Overhead(%)':>11}"
    sep = "-" * len(header)

    print(sep)
    print("OVERHEAD ANALYSIS (vs raw TIFF zlib)")
    print(sep)
    print(header)
    print(sep)

    for ns, approaches in sorted(has_data):
        raw_mean = approaches["raw_tiff_zlib"].mean
        if raw_mean <= 0:
            continue

        for label in ["put_volume", "register_volume"]:
            if label not in approaches or not approaches[label].timings:
                continue
            other_mean = approaches[label].mean
            overhead_ms = (other_mean - raw_mean) * 1000.0
            overhead_pct = ((other_mean - raw_mean) / raw_mean) * 100.0
            print(f"{ns:>6} | {label:>18} | {overhead_ms:>+12.1f} | {overhead_pct:>+10.1f}%")

    print(sep)


def save_results_csv(results: list[BenchmarkResult], path: Path, run_params: dict):
    """Save results to CSV."""
    import csv
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Metadata header
        writer.writerow(["# FileStore Volume Storage Benchmark"])
        writer.writerow(["# datetime", datetime.now().isoformat()])
        writer.writerow(["# slices_series", json.dumps(run_params["slices"])])
        writer.writerow(["# width", run_params["width"]])
        writer.writerow(["# height", run_params["height"]])
        writer.writerow(["# pattern", run_params["pattern"]])
        writer.writerow(["# repeats", run_params["repeats"]])
        writer.writerow(["# warmup", run_params["warmup"]])
        writer.writerow([])

        # Summary table
        writer.writerow(
            [
                "slices",
                "approach",
                "vol_per_sec",
                "mean_s",
                "std_s",
                "min_s",
                "max_s",
                "avg_size_mb",
                "mb_per_sec",
                "num_repeats",
                "errors",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.num_slices,
                    r.approach,
                    f"{r.vol_per_sec:.4f}" if r.timings else "",
                    f"{r.mean:.6f}" if r.timings else "",
                    f"{r.std:.6f}" if r.timings else "",
                    f"{r.min_t:.6f}" if r.timings else "",
                    f"{r.max_t:.6f}" if r.timings else "",
                    f"{r.avg_size_mb:.2f}" if r.sizes_mb else "",
                    f"{r.mb_per_sec:.2f}" if r.timings else "",
                    len(r.timings),
                    "; ".join(r.errors) if r.errors else "",
                ]
            )

        # Per-volume timings
        writer.writerow([])
        writer.writerow(["# Per-volume timings"])
        writer.writerow(["slices", "approach", "repeat", "elapsed_s", "size_mb"])
        for r in results:
            for i, (t, s) in enumerate(zip(r.timings, r.sizes_mb, strict=False)):
                writer.writerow([r.num_slices, r.approach, i + 1, f"{t:.6f}", f"{s:.2f}"])

    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark FileStore volume storage throughput")
    parser.add_argument(
        "--slices",
        type=int,
        nargs="+",
        default=DEFAULT_SLICES,
        help=f"Slice counts to test (default: {DEFAULT_SLICES})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Image width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Image height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=NUM_REPEATS,
        help=f"Number of timed repeats (default: {NUM_REPEATS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=NUM_WARMUP,
        help=f"Number of warmup runs (default: {NUM_WARMUP})",
    )
    parser.add_argument(
        "--pattern",
        choices=["noise", "gradient", "embryo"],
        default="embryo",
        help="Volume pattern: noise, gradient, or embryo (default: embryo)",
    )
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    args = parser.parse_args()

    print("FileStore Volume Storage Benchmark")
    print(f"  Slices:    {args.slices}")
    print(f"  Dimensions: {args.width} x {args.height}")
    print(f"  Pattern:   {args.pattern}")
    print(f"  Repeats:   {args.repeats} (+ {args.warmup} warmup)")
    print("  Approaches: raw_tiff_zlib / put_volume / register_volume")

    results = run_benchmark_sweep(
        slices_list=args.slices,
        width=args.width,
        height=args.height,
        num_repeats=args.repeats,
        num_warmup=args.warmup,
        pattern=args.pattern,
    )

    print_results_table(results)
    print_overhead_analysis(results)

    if args.save:
        csv_path = (
            Path("results")
            / f"benchmark_gentlystore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        run_params = {
            "slices": args.slices,
            "width": args.width,
            "height": args.height,
            "pattern": args.pattern,
            "repeats": args.repeats,
            "warmup": args.warmup,
        }
        save_results_csv(results, csv_path, run_params)


if __name__ == "__main__":
    main()
