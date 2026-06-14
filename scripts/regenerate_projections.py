#!/usr/bin/env python3
"""Regenerate JPEG projections for a session (or all sessions) from volumes.

Use after the projection format changes (e.g. the three-view fix) to refresh
sessions whose on-disk projections are stale (e.g. the old flat-XY max
projection). Reads each volume TIFF and rewrites its projections/t{NNNN}.jpg
via the current generate_jpeg_projection.

Usage:
    python scripts/regenerate_projections.py <session_id>
    python scripts/regenerate_projections.py <session_id> --embryo embryo_001
    python scripts/regenerate_projections.py --all
"""

import argparse
import re
import sys
from pathlib import Path

# Ensure the project root is importable when run as scripts/regenerate_projections.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gently.core.imaging import generate_jpeg_projection, load_volume  # noqa: E402
from gently.settings import settings  # noqa: E402


def _regen_folder(folder: Path, only_embryo: str | None = None) -> int:
    embryos_dir = folder / "embryos"
    if not embryos_dir.exists():
        return 0
    total = 0
    for emb_dir in sorted(d for d in embryos_dir.iterdir() if d.is_dir()):
        if only_embryo and emb_dir.name != only_embryo:
            continue
        vol_dir = emb_dir / "volumes"
        proj_dir = emb_dir / "projections"
        if not vol_dir.exists():
            continue
        tifs = sorted(vol_dir.glob("t*.tif"))
        if not tifs:
            continue
        n = 0
        for vf in tifs:
            m = re.match(r"(t\d+)", vf.stem)
            if not m:
                continue
            proj_path = proj_dir / f"{m.group(1)}.jpg"
            try:
                if generate_jpeg_projection(load_volume(vf), proj_path) is not None:
                    n += 1
            except Exception as e:
                print(f"    ! {vf.name}: {e}")
        print(f"    {emb_dir.name}: regenerated {n}/{len(tifs)} projections")
        total += n
    return total


def _session_dir(storage_base: Path, session_id: str) -> Path | None:
    sessions = storage_base / "sessions"
    if not sessions.exists():
        return None
    matches = [p for p in sessions.glob(f"*{session_id}*") if p.is_dir()]
    return matches[0] if matches else None


def regenerate(session_id: str, only_embryo: str | None = None) -> int:
    base = Path(settings.storage.base_path)
    sdir = _session_dir(base, session_id)
    if sdir is None:
        print(f"Session {session_id} not found under {base / 'sessions'}")
        return 0
    total = _regen_folder(sdir, only_embryo)
    print(f"Done: {total} projections regenerated for {session_id}")
    return total


def regenerate_all() -> int:
    base = Path(settings.storage.base_path)
    sessions = base / "sessions"
    if not sessions.exists():
        print(f"No sessions dir at {sessions}")
        return 0
    folders = sorted(
        p
        for p in sessions.iterdir()
        if p.is_dir() and any((p / "embryos").glob("*/volumes/t*.tif"))
    )
    print(f"Regenerating projections for {len(folders)} session(s) with volumes...")
    grand = 0
    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] {folder.name}")
        grand += _regen_folder(folder)
    print(f"ALL DONE: {grand} projections regenerated across {len(folders)} sessions")
    return grand


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate session projections from volumes")
    ap.add_argument("session_id", nargs="?", help="Session id (omit with --all)")
    ap.add_argument("--all", action="store_true", help="Regenerate every session that has volumes")
    ap.add_argument("--embryo", default=None, help="Only this embryo id")
    args = ap.parse_args()
    if args.all:
        regenerate_all()
    elif args.session_id:
        regenerate(args.session_id, args.embryo)
    else:
        ap.error("provide a session_id or --all")
