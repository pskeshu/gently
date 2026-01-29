"""
End-to-End Volume Acquisition Benchmark
========================================

Measures the full pipeline latency:
1. Acquisition: HTTP → device layer → hardware → file written
2. Storage: GentlyStore registration (move + projection + DB)
3. Viz push: Push projection to visualization server (if running)

All benchmark data is stored in a temporary directory and cleaned up
after the benchmark completes. No data persists to the user's session.

Usage from copilot CLI:
    /benchmark
    /benchmark --volumes 10 --slices 50
"""

import asyncio
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from .copilot import MicroscopyCopilot


@dataclass
class VolumeTiming:
    """Timing breakdown for a single volume acquisition."""
    volume_idx: int
    embryo_id: str
    timepoint: int

    # Timing stages (seconds)
    acquisition_time: float = 0.0    # HTTP → device layer → hardware → file
    storage_time: float = 0.0        # GentlyStore registration
    viz_push_time: float = 0.0       # Viz server push (if running)
    total_time: float = 0.0          # End-to-end

    # Metadata
    num_slices: int = 0
    volume_shape: tuple = ()
    file_size_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""
    timings: List[VolumeTiming] = field(default_factory=list)
    num_embryos: int = 1
    num_slices: int = 50
    exposure_ms: float = 10.0
    started_at: str = ""
    completed_at: str = ""

    @property
    def successful(self) -> List[VolumeTiming]:
        return [t for t in self.timings if t.success]

    @property
    def failed(self) -> List[VolumeTiming]:
        return [t for t in self.timings if not t.success]

    def _stat(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values."""
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

    @property
    def acquisition_stats(self) -> Dict[str, float]:
        return self._stat([t.acquisition_time for t in self.successful])

    @property
    def storage_stats(self) -> Dict[str, float]:
        return self._stat([t.storage_time for t in self.successful])

    @property
    def viz_push_stats(self) -> Dict[str, float]:
        return self._stat([t.viz_push_time for t in self.successful])

    @property
    def total_stats(self) -> Dict[str, float]:
        return self._stat([t.total_time for t in self.successful])

    @property
    def volumes_per_second(self) -> float:
        stats = self.total_stats
        return 1.0 / stats["mean"] if stats["mean"] > 0 else 0.0

    @property
    def avg_file_size_mb(self) -> float:
        sizes = [t.file_size_mb for t in self.successful if t.file_size_mb > 0]
        return statistics.mean(sizes) if sizes else 0.0


async def run_benchmark(
    copilot: "MicroscopyCopilot",
    num_volumes: int = 5,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    warmup: int = 1,
    console: Optional[Console] = None,
) -> BenchmarkResults:
    """
    Run end-to-end volume acquisition benchmark.

    All data is stored in a temporary directory and cleaned up after
    the benchmark completes. No data persists to the user's session.

    Parameters
    ----------
    copilot : MicroscopyCopilot
        Active copilot instance with microscope connection
    num_volumes : int
        Number of volumes to acquire for timing
    num_slices : int
        Slices per volume
    exposure_ms : float
        Camera exposure time
    warmup : int
        Number of warmup volumes (not timed)
    console : Console, optional
        Rich console for output

    Returns
    -------
    BenchmarkResults
        Timing breakdown and statistics
    """
    from ..store import GentlyStore

    if console is None:
        console = Console()

    # Validate connection
    if not copilot.client or not copilot.client.is_connected:
        raise RuntimeError("Microscope not connected. Cannot run hardware benchmark.")

    # Get embryos for round-robin (optional - works without embryos too)
    embryos = list(copilot.experiment.embryos.values()) if copilot.experiment.embryos else []

    # If no embryos, create a synthetic one for benchmarking at current position
    use_synthetic = len(embryos) == 0
    if use_synthetic:
        embryo_ids = ["benchmark_pos"]
    else:
        embryo_ids = [e.embryo_id for e in embryos]

    results = BenchmarkResults(
        num_embryos=len(embryo_ids),
        num_slices=num_slices,
        exposure_ms=exposure_ms,
        started_at=datetime.now().isoformat(),
    )

    # Check if viz server is running
    viz_server = getattr(copilot, '_viz_server', None)
    has_viz = viz_server is not None

    # Create temporary directory for benchmark data
    temp_dir = Path(tempfile.mkdtemp(prefix="gently_benchmark_"))
    temp_store = None

    try:
        # Create temporary GentlyStore (isolated from user's data)
        temp_store = GentlyStore(temp_dir)
        benchmark_session = "benchmark"
        temp_store.create_session(benchmark_session, name="Benchmark Session")

        # Register embryos in temp store
        for eid in embryo_ids:
            temp_store.register_embryo(benchmark_session, eid)

        position_info = "current position" if use_synthetic else f"{len(embryos)} embryos (round-robin)"
        console.print(Panel(
            f"[bold]End-to-End Volume Benchmark[/bold]\n\n"
            f"Volumes: {num_volumes} (+ {warmup} warmup)\n"
            f"Slices: {num_slices}\n"
            f"Exposure: {exposure_ms} ms\n"
            f"Position: {position_info}\n"
            f"Viz server: {'running' if has_viz else 'not running'}\n"
            f"[dim]Data: temporary (auto-cleanup)[/dim]",
            title="Benchmark Configuration",
            border_style="blue",
        ))

        # Track timepoints per embryo
        timepoints: Dict[str, int] = {eid: 0 for eid in embryo_ids}

        total_iterations = warmup + num_volumes

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Running benchmark...", total=total_iterations)

            for i in range(total_iterations):
                is_warmup = i < warmup
                volume_idx = i - warmup if not is_warmup else -1

                # Round-robin embryo selection
                embryo_id = embryo_ids[i % len(embryo_ids)]
                embryo = embryos[i % len(embryos)] if embryos else None
                tp = timepoints[embryo_id]
                timepoints[embryo_id] += 1

                status = "Warmup" if is_warmup else f"Volume {volume_idx + 1}/{num_volumes}"
                progress.update(task, description=f"{status}: t={tp}")

                timing = VolumeTiming(
                    volume_idx=volume_idx,
                    embryo_id=embryo_id,
                    timepoint=tp,
                    num_slices=num_slices,
                )

                try:
                    # Get calibration (from embryo if available, else defaults)
                    if embryo and embryo.calibration:
                        cal = embryo.calibration
                        galvo_amp = cal.get('galvo_amplitude', 0.5)
                        galvo_center = cal.get('galvo_center', 0.0)
                        piezo_amp = cal.get('piezo_amplitude', 25.0)
                        piezo_center = cal.get('piezo_center', 50.0)
                    else:
                        # Default calibration for benchmark
                        galvo_amp = 0.5
                        galvo_center = 0.0
                        piezo_amp = 25.0
                        piezo_center = 50.0

                    # Move to embryo position if available
                    if embryo and embryo.position:
                        await copilot.client.move_stage(
                            embryo.position['x'],
                            embryo.position['y']
                        )

                    # === Stage 1: Acquisition ===
                    t0 = time.perf_counter()
                    result = await copilot.client.acquire_volume(
                        num_slices=num_slices,
                        exposure_ms=exposure_ms,
                        galvo_amplitude=galvo_amp,
                        galvo_center=galvo_center,
                        piezo_amplitude=piezo_amp,
                        piezo_center=piezo_center,
                    )
                    t1 = time.perf_counter()
                    timing.acquisition_time = t1 - t0

                    if not result.get('success'):
                        timing.success = False
                        timing.error = result.get('error', 'Acquisition failed')
                        if not is_warmup:
                            results.timings.append(timing)
                        progress.advance(task)
                        continue

                    volume = result.get('volume')
                    volume_path = result.get('volume_path')
                    timing.volume_shape = volume.shape if volume is not None else ()

                    # === Stage 2: GentlyStore registration (to temp store) ===
                    t2 = time.perf_counter()
                    if volume_path:
                        # Register the volume (zero-copy path)
                        canonical_path = temp_store.register_volume(
                            benchmark_session,
                            embryo_id,
                            tp,
                            Path(volume_path),
                        )
                        timing.file_size_mb = canonical_path.stat().st_size / (1024 * 1024)
                    elif volume is not None:
                        # put_volume path (writes TIFF)
                        canonical_path = temp_store.put_volume(
                            benchmark_session,
                            embryo_id,
                            tp,
                            volume,
                        )
                        timing.file_size_mb = canonical_path.stat().st_size / (1024 * 1024)
                    t3 = time.perf_counter()
                    timing.storage_time = t3 - t2

                    # === Stage 3: Viz server push (optional, doesn't persist) ===
                    t4 = time.perf_counter()
                    if has_viz and volume is not None:
                        # Generate and push projection (in-memory only)
                        proj = volume.max(axis=0) if volume.ndim == 3 else volume
                        await viz_server.push_image(
                            proj,
                            uid=f"benchmark_{embryo_id}_t{tp:04d}",
                            data_type="benchmark",
                            metadata={"embryo_id": embryo_id, "timepoint": tp, "benchmark": True},
                        )
                    t5 = time.perf_counter()
                    timing.viz_push_time = t5 - t4

                    timing.total_time = t5 - t0

                except Exception as e:
                    timing.success = False
                    timing.error = str(e)
                    timing.total_time = time.perf_counter() - t0

                if not is_warmup:
                    results.timings.append(timing)

                progress.advance(task)

        results.completed_at = datetime.now().isoformat()

    finally:
        # Clean up temporary store and directory
        if temp_store:
            temp_store.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            console.print(f"[dim]Cleaned up temporary benchmark data[/dim]")

    return results


def print_benchmark_results(results: BenchmarkResults, console: Optional[Console] = None):
    """Print formatted benchmark results."""
    if console is None:
        console = Console()

    # Summary stats
    total = results.total_stats
    acq = results.acquisition_stats
    stor = results.storage_stats
    viz = results.viz_push_stats

    # Results table
    table = Table(title="Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="dim")
    table.add_column("Mean (s)", justify="right")
    table.add_column("Std (s)", justify="right")
    table.add_column("Min (s)", justify="right")
    table.add_column("Max (s)", justify="right")
    table.add_column("% of Total", justify="right")

    def pct(val: float) -> str:
        if total["mean"] > 0:
            return f"{(val / total['mean']) * 100:.1f}%"
        return "—"

    table.add_row(
        "Acquisition",
        f"{acq['mean']:.3f}", f"{acq['std']:.3f}",
        f"{acq['min']:.3f}", f"{acq['max']:.3f}",
        pct(acq['mean']),
    )
    table.add_row(
        "Storage",
        f"{stor['mean']:.3f}", f"{stor['std']:.3f}",
        f"{stor['min']:.3f}", f"{stor['max']:.3f}",
        pct(stor['mean']),
    )
    table.add_row(
        "Viz Push",
        f"{viz['mean']:.3f}", f"{viz['std']:.3f}",
        f"{viz['min']:.3f}", f"{viz['max']:.3f}",
        pct(viz['mean']),
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total['mean']:.3f}[/bold]", f"{total['std']:.3f}",
        f"{total['min']:.3f}", f"{total['max']:.3f}",
        "[bold]100%[/bold]",
        style="bold",
    )

    console.print(table)

    # Summary panel
    summary = Text()
    summary.append(f"\nThroughput: ", style="bold")
    summary.append(f"{results.volumes_per_second:.2f} volumes/sec\n", style="green bold")
    summary.append(f"Latency: ", style="bold")
    summary.append(f"{total['mean'] * 1000:.0f} ms/volume\n", style="yellow")
    summary.append(f"File size: ", style="bold")
    summary.append(f"{results.avg_file_size_mb:.1f} MB avg\n")
    summary.append(f"Success rate: ", style="bold")
    success_rate = len(results.successful) / len(results.timings) * 100 if results.timings else 0
    summary.append(f"{success_rate:.0f}% ({len(results.successful)}/{len(results.timings)})\n")

    console.print(Panel(summary, title="Summary", border_style="green"))

    # Per-volume breakdown (if requested)
    if len(results.timings) <= 20:
        detail_table = Table(title="Per-Volume Breakdown", show_header=True)
        detail_table.add_column("#", justify="right", style="dim")
        detail_table.add_column("Embryo")
        detail_table.add_column("Acq (s)", justify="right")
        detail_table.add_column("Store (s)", justify="right")
        detail_table.add_column("Viz (s)", justify="right")
        detail_table.add_column("Total (s)", justify="right")
        detail_table.add_column("Status")

        for t in results.timings:
            status = "[green]✓[/green]" if t.success else f"[red]✗ {t.error}[/red]"
            detail_table.add_row(
                str(t.volume_idx + 1),
                t.embryo_id,
                f"{t.acquisition_time:.3f}",
                f"{t.storage_time:.3f}",
                f"{t.viz_push_time:.3f}",
                f"{t.total_time:.3f}",
                status,
            )

        console.print(detail_table)


def save_benchmark_csv(results: BenchmarkResults, path: Path):
    """Save benchmark results to CSV."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Metadata
        writer.writerow(["# End-to-End Volume Benchmark"])
        writer.writerow(["# started_at", results.started_at])
        writer.writerow(["# completed_at", results.completed_at])
        writer.writerow(["# num_slices", results.num_slices])
        writer.writerow(["# exposure_ms", results.exposure_ms])
        writer.writerow(["# num_embryos", results.num_embryos])
        writer.writerow([])

        # Summary stats
        total = results.total_stats
        writer.writerow(["# Summary"])
        writer.writerow(["# volumes_per_second", f"{results.volumes_per_second:.4f}"])
        writer.writerow(["# mean_total_s", f"{total['mean']:.6f}"])
        writer.writerow(["# std_total_s", f"{total['std']:.6f}"])
        writer.writerow([])

        # Per-volume data
        writer.writerow([
            "volume_idx", "embryo_id", "timepoint",
            "acquisition_s", "storage_s", "viz_push_s", "total_s",
            "file_size_mb", "success", "error"
        ])
        for t in results.timings:
            writer.writerow([
                t.volume_idx, t.embryo_id, t.timepoint,
                f"{t.acquisition_time:.6f}",
                f"{t.storage_time:.6f}",
                f"{t.viz_push_time:.6f}",
                f"{t.total_time:.6f}",
                f"{t.file_size_mb:.2f}",
                t.success,
                t.error or "",
            ])

    return path
