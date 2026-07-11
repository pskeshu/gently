"""
End-to-End Volume Acquisition Benchmark
========================================

Measures the full pipeline latency:
1. Acquisition: HTTP -> device layer -> hardware -> file written
2. Storage: FileStore registration (move + projection + DB)
3. Viz push: Push projection to visualization server (if running)

All benchmark data is stored in a temporary directory and cleaned up
after the benchmark completes. No data persists to the user's session.

Usage from agent CLI:
    /benchmark
    /benchmark --volumes 10 --slices 50
"""

import csv
import logging
import shutil
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import MicroscopyAgent

logger = logging.getLogger(__name__)


@dataclass
class VolumeTiming:
    """Timing breakdown for a single volume acquisition."""

    volume_idx: int
    embryo_id: str
    timepoint: int

    # Timing stages (seconds)
    acquisition_time: float = 0.0
    storage_time: float = 0.0
    viz_push_time: float = 0.0
    total_time: float = 0.0

    # Metadata
    num_slices: int = 0
    volume_shape: tuple = ()
    file_size_mb: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""

    timings: list[VolumeTiming] = field(default_factory=list)
    num_embryos: int = 1
    num_slices: int = 50
    exposure_ms: float = 10.0
    started_at: str = ""
    completed_at: str = ""

    @property
    def successful(self) -> list[VolumeTiming]:
        return [t for t in self.timings if t.success]

    @property
    def failed(self) -> list[VolumeTiming]:
        return [t for t in self.timings if not t.success]

    def _stat(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

    @property
    def acquisition_stats(self) -> dict[str, float]:
        return self._stat([t.acquisition_time for t in self.successful])

    @property
    def storage_stats(self) -> dict[str, float]:
        return self._stat([t.storage_time for t in self.successful])

    @property
    def viz_push_stats(self) -> dict[str, float]:
        return self._stat([t.viz_push_time for t in self.successful])

    @property
    def total_stats(self) -> dict[str, float]:
        return self._stat([t.total_time for t in self.successful])

    @property
    def volumes_per_second(self) -> float:
        stats = self.total_stats
        return 1.0 / stats["mean"] if stats["mean"] > 0 else 0.0

    @property
    def avg_file_size_mb(self) -> float:
        sizes = [t.file_size_mb for t in self.successful if t.file_size_mb > 0]
        return statistics.mean(sizes) if sizes else 0.0

    # Convenience properties used by bridge /benchmark handler
    @property
    def n_volumes(self) -> int:
        return len(self.timings)

    @property
    def mean_acquisition(self) -> float:
        return self.acquisition_stats["mean"]

    @property
    def mean_storage(self) -> float:
        return self.storage_stats["mean"]

    @property
    def mean_total(self) -> float:
        return self.total_stats["mean"]

    @property
    def fps(self) -> float:
        return self.volumes_per_second


async def run_benchmark(
    agent: "MicroscopyAgent",
    num_volumes: int = 5,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    warmup: int = 1,
    # Legacy parameters (ignored, kept for API compat with bridge)
    n_volumes: int | None = None,
    n_slices: int | None = None,
    n_warmup: int | None = None,
    progress_fn: Callable[..., Any] | None = None,
) -> BenchmarkResults:
    """
    Run end-to-end volume acquisition benchmark.

    All data is stored in a temporary directory and cleaned up after
    the benchmark completes.
    """
    from ..core.file_store import FileStore

    # Support bridge's keyword names
    if n_volumes is not None:
        num_volumes = n_volumes
    if n_slices is not None:
        num_slices = n_slices
    if n_warmup is not None:
        warmup = n_warmup

    if not agent.client or not agent.client.is_connected:
        raise RuntimeError("Microscope not connected. Cannot run hardware benchmark.")

    embryos = list(agent.experiment.embryos.values()) if agent.experiment.embryos else []
    use_synthetic = len(embryos) == 0
    embryo_ids = ["benchmark_pos"] if use_synthetic else [e.id for e in embryos]

    results = BenchmarkResults(
        num_embryos=len(embryo_ids),
        num_slices=num_slices,
        exposure_ms=exposure_ms,
        started_at=datetime.now().isoformat(),
    )

    viz_server = getattr(agent, "viz_server", None)

    temp_dir = Path(tempfile.mkdtemp(prefix="gently_benchmark_"))
    temp_store = None

    try:
        temp_store = FileStore(temp_dir)
        benchmark_session = "benchmark"
        temp_store.create_session(benchmark_session, name="Benchmark Session")
        for eid in embryo_ids:
            temp_store.register_embryo(benchmark_session, eid)

        logger.info(
            "Benchmark: %d volumes + %d warmup, %d slices, %.0f ms exposure",
            num_volumes,
            warmup,
            num_slices,
            exposure_ms,
        )

        timepoints: dict[str, int] = {eid: 0 for eid in embryo_ids}
        total_iterations = warmup + num_volumes

        for i in range(total_iterations):
            is_warmup = i < warmup
            volume_idx = i - warmup if not is_warmup else -1

            embryo_id = embryo_ids[i % len(embryo_ids)]
            embryo = embryos[i % len(embryos)] if embryos else None
            tp = timepoints[embryo_id]
            timepoints[embryo_id] += 1

            status = "warmup" if is_warmup else f"{volume_idx + 1}/{num_volumes}"
            logger.info("Benchmark %s: embryo=%s tp=%d", status, embryo_id, tp)

            if progress_fn:
                if is_warmup:
                    await progress_fn("warmup", i + 1, warmup, None)
                else:
                    await progress_fn("acquiring", volume_idx + 1, num_volumes, None)

            timing = VolumeTiming(
                volume_idx=volume_idx,
                embryo_id=embryo_id,
                timepoint=tp,
                num_slices=num_slices,
            )

            try:
                if embryo and embryo.calibration:
                    cal = embryo.calibration
                    galvo_amp = cal.get("galvo_amplitude", 0.5)
                    galvo_center = cal.get("galvo_center", 0.0)
                    piezo_amp = cal.get("piezo_amplitude", 25.0)
                    piezo_center = cal.get("piezo_center", 50.0)
                else:
                    galvo_amp, galvo_center = 0.5, 0.0
                    piezo_amp, piezo_center = 25.0, 50.0

                if embryo and embryo.stage_position:
                    await agent.client.move_stage(
                        embryo.stage_position["x"], embryo.stage_position["y"]
                    )

                # Stage 1: Acquisition
                t0 = time.perf_counter()
                result = await agent.client.acquire_volume(
                    num_slices=num_slices,
                    exposure_ms=exposure_ms,
                    galvo_amplitude=galvo_amp,
                    galvo_center=galvo_center,
                    piezo_amplitude=piezo_amp,
                    piezo_center=piezo_center,
                )
                t1 = time.perf_counter()
                timing.acquisition_time = t1 - t0

                if not result.get("success"):
                    timing.success = False
                    timing.error = result.get("error", "Acquisition failed")
                    if not is_warmup:
                        results.timings.append(timing)
                    continue

                volume = result.get("volume")
                volume_path = result.get("volume_path")
                timing.volume_shape = volume.shape if volume is not None else ()

                # Stage 2: Storage
                t2 = time.perf_counter()
                if volume_path:
                    canonical_path = temp_store.register_volume(
                        benchmark_session,
                        embryo_id,
                        tp,
                        Path(volume_path),
                        volume_data=volume,
                    )
                    timing.file_size_mb = canonical_path.stat().st_size / (1024 * 1024)
                elif volume is not None:
                    canonical_path = temp_store.put_volume(
                        benchmark_session,
                        embryo_id,
                        tp,
                        volume,
                    )
                    timing.file_size_mb = canonical_path.stat().st_size / (1024 * 1024)
                t3 = time.perf_counter()
                timing.storage_time = t3 - t2

                # Stage 3: Viz push
                t4 = time.perf_counter()
                if viz_server is not None and volume is not None:
                    proj = volume.max(axis=0) if volume.ndim == 3 else volume
                    await viz_server.push_image(
                        proj,
                        uid=f"benchmark_{embryo_id}_t{tp:04d}",
                        data_type="benchmark",
                        metadata={
                            "embryo_id": embryo_id,
                            "timepoint": tp,
                            "benchmark": True,
                        },
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
                if progress_fn:
                    await progress_fn("volume_done", volume_idx + 1, num_volumes, timing)

        results.completed_at = datetime.now().isoformat()

    finally:
        if temp_store:
            temp_store.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info("Cleaned up temporary benchmark data")

    return results


def print_benchmark_results(results: BenchmarkResults):
    """Print benchmark results as plain text."""
    total = results.total_stats
    acq = results.acquisition_stats
    stor = results.storage_stats

    lines = [
        "Benchmark Results",
        f"  Volumes:    {len(results.successful)}/{len(results.timings)} successful",
        f"  Throughput: {results.volumes_per_second:.2f} volumes/sec",
        f"  Latency:    {total['mean'] * 1000:.0f} ms/volume",
        "",
        "  Stage breakdown (mean):",
        f"    Acquisition: {acq['mean']:.3f}s",
        f"    Storage:     {stor['mean']:.3f}s",
        f"    Total:       {total['mean']:.3f}s",
    ]
    if results.avg_file_size_mb > 0:
        lines.append(f"  File size:  {results.avg_file_size_mb:.1f} MB avg")

    logger.info("\n".join(lines))


def save_benchmark_csv(results: BenchmarkResults, path: Path):
    """Save benchmark results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["# End-to-End Volume Benchmark"])
        writer.writerow(["# started_at", results.started_at])
        writer.writerow(["# completed_at", results.completed_at])
        writer.writerow(["# num_slices", results.num_slices])
        writer.writerow(["# exposure_ms", results.exposure_ms])
        writer.writerow(["# num_embryos", results.num_embryos])
        writer.writerow([])

        total = results.total_stats
        writer.writerow(["# Summary"])
        writer.writerow(["# volumes_per_second", f"{results.volumes_per_second:.4f}"])
        writer.writerow(["# mean_total_s", f"{total['mean']:.6f}"])
        writer.writerow(["# std_total_s", f"{total['std']:.6f}"])
        writer.writerow([])

        writer.writerow(
            [
                "volume_idx",
                "embryo_id",
                "timepoint",
                "acquisition_s",
                "storage_s",
                "viz_push_s",
                "total_s",
                "file_size_mb",
                "success",
                "error",
            ]
        )
        for t in results.timings:
            writer.writerow(
                [
                    t.volume_idx,
                    t.embryo_id,
                    t.timepoint,
                    f"{t.acquisition_time:.6f}",
                    f"{t.storage_time:.6f}",
                    f"{t.viz_push_time:.6f}",
                    f"{t.total_time:.6f}",
                    f"{t.file_size_mb:.2f}",
                    t.success,
                    t.error or "",
                ]
            )

    return path
