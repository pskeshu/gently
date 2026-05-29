#!/usr/bin/env python3
"""Regenerate JPEG projections for a session from its stored volumes.

Use after the projection format changes (e.g. the three-view fix) to refresh
older sessions whose on-disk projections are stale (e.g. the old A|B
side-by-side max projection). Reads each volume TIFF and rewrites its
projections/t{NNNN}.jpg via the current generate_jpeg_projection.

Usage:
    python scripts/regenerate_projections.py <session_id>
    python scripts/regenerate_projections.py <session_id> --embryo embryo_001
"""

import argparse
import re
import sys
from pathlib import Path

from gently.settings import settings
from gently.core.imaging import load_volume, generate_jpeg_projection


def _session_dir(storage_base: Path, session_id: str) -> Path | None:
    sessions = storage_base / "sessions"
    if not sessions.exists():
        return None
    # Folder names are {date}_{time}_{slug}_{id8}; match the id suffix.
    matches = [p for p in sessions.glob(f"*{session_id}*") if p.is_dir()]
    return matches[0] if matches else None


def regenerate(session_id: str, only_embryo: str | None = None) -> int:
    base = Path(settings.storage.base_path)
    sdir = _session_dir(base, session_id)
    if sdir is None:
        print(f"Session {session_id} not found under {base / 'sessions'}")
        return 0
    embryos_dir = sdir / "embryos"
    if not embryos_dir.exists():
        print(f"No embryos dir in {sdir}")
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
        n = 0
        for vf in tifs:
            m = re.match(r"(t\d+)", vf.stem)
            if not m:
                continue
            proj_path = proj_dir / f"{m.group(1)}.jpg"
            try:
                vol = load_volume(vf)
                if generate_jpeg_projection(vol, proj_path) is not None:
                    n += 1
            except Exception as e:
                print(f"  ! {vf.name}: {e}")
        print(f"  {emb_dir.name}: regenerated {n}/{len(tifs)} projections")
        total += n
    print(f"Done: {total} projections regenerated for {session_id}")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Regenerate session projections from volumes")
    ap.add_argument("session_id")
    ap.add_argument("--embryo", default=None, help="Only this embryo id")
    args = ap.parse_args()
    regenerate(args.session_id, args.embryo)
