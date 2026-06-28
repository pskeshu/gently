"""
Launch the viz server with real benchmark data — no agent, no device layer, no API credits.

Loads TIFF volumes from the perception benchmark, generates three-view
projections, and pushes them to the viz server for GUI testing.

Usage:
    python scripts/launch_viz_server.py
    python scripts/launch_viz_server.py --port 8080
    python scripts/launch_viz_server.py --delay 0.5          # seconds between pushes
    python scripts/launch_viz_server.py --timepoints 20       # limit per embryo
    python scripts/launch_viz_server.py --embryos embryo_1    # specific embryos
    python scripts/launch_viz_server.py --fast                # push everything, then serve
"""

import argparse
import asyncio
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.perception.ground_truth import GroundTruth
from benchmarks.perception.testset import _discover_volumes
from gently.core.event_bus import EventBus, EventType
from gently.core.imaging import (
    apply_crop_bounds,
    compute_crop_bounds,
    load_volume,
    projection_three_view,
)
from gently.settings import settings

# Benchmark data paths
REPO_ROOT = Path(__file__).resolve().parents[1]
_VOLUMES_REL = Path("benchmarks", "data", "hf-benchmark", "data", "volumes")
_GT_REL = Path("benchmarks", "data", "ground_truth", "59799c78.json")


def _find_main_worktree() -> Path:
    """Find the main git worktree root (handles worktree checkouts)."""
    try:
        out = subprocess.check_output(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if line.startswith("worktree "):
                return Path(line.split(" ", 1)[1])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return REPO_ROOT


def _resolve_data_path(rel: Path) -> Path:
    """Resolve a data path, checking repo root then main worktree."""
    local = REPO_ROOT / rel
    if local.exists():
        return local
    main = _find_main_worktree() / rel
    if main.exists():
        return main
    return local  # fallback (will show "not found" later)


def _effective_tp(paths: list, max_timepoints: int) -> int:
    """Number of timepoints to use for this embryo."""
    return min(len(paths), max_timepoints) if max_timepoints > 0 else len(paths)


def project_volume(tiff_path: Path):
    """Load a TIFF volume and generate a three-view projection (uint8)."""
    vol = load_volume(tiff_path)
    bounds = compute_crop_bounds(vol)
    cropped = apply_crop_bounds(vol, bounds)
    img, _ = projection_three_view(cropped)
    return img


async def run(
    port: int,
    delay: float,
    max_timepoints: int,
    embryo_ids: list[str] | None,
    fast: bool,
    volumes_dir: Path,
    gt_path: Path,
):
    event_bus = EventBus()

    from gently.ui.web.server import VisualizationServer

    server = VisualizationServer(port=port, event_bus=event_bus)
    await server.start()
    print(f"\n  Viz server: http://localhost:{port}\n")

    # Load data
    all_vols = _discover_volumes(volumes_dir)
    if embryo_ids:
        all_vols = {k: v for k, v in all_vols.items() if k in embryo_ids}

    if not all_vols:
        print(f"No volumes found at {volumes_dir}")
        print("Download: https://huggingface.co/datasets/pskeshu/gently-perception-benchmark")
        print("\nServer running empty. Ctrl+C to exit.")
        await asyncio.Event().wait()
        return

    gt = GroundTruth.from_json(gt_path) if gt_path.exists() else GroundTruth()
    session_id = "demo"

    for eid, paths in sorted(all_vols.items()):
        n = _effective_tp(paths, max_timepoints)
        stages = gt.transitions.get(eid, {})
        print(f"  {eid}: {n}/{len(paths)} timepoints", end="")
        if stages:
            print(f"  stages: {list(stages.keys())}", end="")
        print()

    actual_delay = 0 if fast else delay
    print()

    event_bus.publish(EventType.SESSION_STARTED, {"session_id": session_id}, source="demo")

    for eid in sorted(all_vols):
        event_bus.publish(EventType.EMBRYO_DETECTED, {"embryo_id": eid}, source="demo")

    # Push projections round-robin (like a real timelapse)
    total = sum(_effective_tp(p, max_timepoints) for p in all_vols.values())
    max_tp = max(_effective_tp(p, max_timepoints) for p in all_vols.values())
    pushed = 0

    event_bus.publish(
        EventType.ACQUISITION_STARTED,
        {
            "session_id": session_id,
            "embryo_ids": sorted(all_vols.keys()),
            "total_timepoints": max_tp,
        },
        source="demo",
    )

    for t in range(max_tp):
        for eid in sorted(all_vols):
            paths = all_vols[eid]
            if t >= _effective_tp(paths, max_timepoints):
                continue

            # paths[t] is a (timestamp, path) tuple from _discover_volumes
            projection = project_volume(paths[t][1])

            proj_uid = f"proj_{session_id}_{eid}_t{t:04d}"
            vol_uid = f"volume_{session_id}_{eid}_t{t:04d}"

            await server.push_image(
                array=projection,
                uid=proj_uid,
                data_type="volume_projection",
                metadata={
                    "embryo_id": eid,
                    "timepoint": t,
                    "projection_uid": proj_uid,
                    "volume_uid": vol_uid,
                    "projection_type": "three_view",
                },
            )

            stage = gt.get_stage_at(eid, t)
            event_bus.publish(
                EventType.VOLUME_ACQUIRED,
                {
                    "embryo_id": eid,
                    "timepoint": t,
                    "volume_uid": vol_uid,
                    "projection_uid": proj_uid,
                    "stage": stage,
                },
                source="demo",
            )

            pushed += 1
            print(
                f"\r  [{pushed}/{total}] {eid} t={t} [{stage or '?'}]",
                end="",
                flush=True,
            )

            if actual_delay:
                await asyncio.sleep(actual_delay)

    event_bus.publish(
        EventType.ACQUISITION_COMPLETED,
        {"session_id": session_id, "total_timepoints": max_tp},
        source="demo",
    )

    print(f"\n\n  Done — {pushed} projections pushed.")
    print(f"  Viz server running at http://localhost:{port}")
    print("  Ctrl+C to exit.\n")

    await asyncio.Event().wait()


def main():
    parser = argparse.ArgumentParser(description="Launch viz server with benchmark data")
    parser.add_argument("--port", type=int, default=settings.network.viz_port)
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Seconds between pushes (default 1)"
    )
    parser.add_argument("--fast", action="store_true", help="Push all data immediately, then serve")
    parser.add_argument(
        "--timepoints", type=int, default=0, help="Max timepoints per embryo (0=all)"
    )
    parser.add_argument("--embryos", nargs="+", default=None, help="Specific embryo IDs")
    parser.add_argument("--volumes", type=str, default=None, help="Path to volumes dir")
    parser.add_argument("--ground-truth", type=str, default=None, help="Path to ground truth JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    volumes_dir = Path(args.volumes) if args.volumes else _resolve_data_path(_VOLUMES_REL)
    gt_path = Path(args.ground_truth) if args.ground_truth else _resolve_data_path(_GT_REL)

    try:
        asyncio.run(
            run(
                port=args.port,
                delay=args.delay,
                max_timepoints=args.timepoints,
                embryo_ids=args.embryos,
                fast=args.fast,
                volumes_dir=volumes_dir,
                gt_path=gt_path,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
