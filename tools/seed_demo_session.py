#!/usr/bin/env python3
"""Seed a realistic session so the UI can be designed against real content.

Without hardware every panel renders its empty state, so you cannot judge
layout, density, gauges or legibility — every folded window just says "empty".
This writes a plausible session through the REAL FileStore API, so the result is
indistinguishable from an acquired one: same directory layout, same
projections, same predictions.jsonl, same index.

    GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python tools/seed_demo_session.py
    GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python tools/seed_demo_session.py \
        --embryos 6 --timepoints 24

Not a test fixture and not for CI — a bench for interface work. It only ever
adds a session; it never touches existing ones.

ponytail: synthetic volumes are gaussian blobs on a noise floor, which is enough
for projections, thumbnails and layout. Swap in real TIFFs if perception
behaviour rather than interface behaviour is what you are looking at.
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gently.core.file_store import FileStore  # noqa: E402
from gently.harness.memory.file_store import FileContextStore  # noqa: E402
from gently.harness.memory.notebook import Author, Note, NoteKind, NoteStatus  # noqa: E402
from gently.settings import settings  # noqa: E402

# A real C. elegans staging sequence, so the UI shows plausible transitions
# rather than lorem ipsum.
STAGES = [
    "1-cell",
    "2-cell",
    "4-cell",
    "8-cell",
    "26-cell",
    "gastrula",
    "lima-bean",
    "comma",
    "1.5-fold",
    "2-fold",
    "3-fold",
    "hatching",
]
ROLES = ["test", "test", "test", "calibration"]


def synthetic_volume(
    rng: np.random.Generator, z: int = 12, h: int = 128, w: int = 128, n_cells: int = 4
) -> np.ndarray:
    """A gaussian-blob volume on a noise floor. Enough for projections."""
    vol = rng.normal(180, 25, (z, h, w)).astype(np.float32)
    zz, yy, xx = np.mgrid[0:z, 0:h, 0:w]
    for _ in range(n_cells):
        cz, cy, cx = rng.uniform(2, z - 2), rng.uniform(20, h - 20), rng.uniform(20, w - 20)
        r = rng.uniform(8, 18)
        d2 = ((zz - cz) * 2.5) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol += 900 * np.exp(-d2 / (2 * r * r))
    return np.clip(vol, 0, 65535).astype(np.uint16)


def seed(store: FileStore, *, n_embryos: int, n_timepoints: int, seed_val: int) -> str:
    rng = np.random.default_rng(seed_val)
    random.seed(seed_val)

    sid = uuid.uuid4().hex[:8]
    store.create_session(
        sid,
        name="seeded demo",
        description="Synthetic session for interface work. Not acquired data.",
        metadata={"organism": "C. elegans", "seeded": True, "seed": seed_val},
    )
    print(f"session {sid}")

    incoming = Path(settings.storage.base_path) / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)

    run_id = store.create_perception_run(
        session_id=sid,
        name="seeded staging pass",
        method="vlm",
        model_name="seeded-vlm",
        source="seed",
        config={"seeded": True},
    )

    for i in range(n_embryos):
        eid = f"e{i + 1:02d}"
        # positions on a plausible dish, in microns
        px, py = 1400 + i * 190 + rng.uniform(-30, 30), 850 + rng.uniform(-90, 90)
        store.register_embryo(
            session_id=sid,
            embryo_id=eid,
            embryo_uid=f"{sid}-{eid}",
            nickname=None,
            position_x=float(px),
            position_y=float(py),
            position_coarse={"x": float(px), "y": float(py)},
            position_fine={
                "x": float(px + rng.uniform(-4, 4)),
                "y": float(py + rng.uniform(-4, 4)),
                "z": float(rng.uniform(-20, -5)),
            },
            role=ROLES[i % len(ROLES)],
            strain="N2",
        )

        # each embryo starts at a different point in development
        start = rng.integers(0, 4)
        for t in range(n_timepoints):
            vol = synthetic_volume(rng, n_cells=min(2 + t // 2, 12))
            tmp = incoming / f"{uuid.uuid4().hex[:12]}.tif"
            try:
                import tifffile

                tifffile.imwrite(tmp, vol)
            except ImportError:
                np.save(tmp.with_suffix(".npy"), vol)
                tmp = tmp.with_suffix(".npy")

            acquired = datetime.now() - timedelta(minutes=(n_timepoints - t) * 4)
            store.register_volume(
                session_id=sid,
                embryo_id=eid,
                timepoint=t,
                incoming_path=tmp,
                metadata={"acquired_at": acquired.isoformat(), "seeded": True},
                volume_data=vol,
            )

            idx = min(int(start) + t // 2, len(STAGES) - 1)
            transitional = t % 5 == 4
            store.store_prediction(
                run_id=run_id,
                session_id=sid,
                embryo_id=eid,
                timepoint=t,
                predicted_stage=STAGES[idx],
                confidence=float(np.clip(rng.normal(0.84, 0.09), 0.4, 0.99)),
                reasoning=(
                    f"{'Transitioning: ' if transitional else ''}"
                    f"nuclei count and furrow geometry consistent with {STAGES[idx]}."
                ),
                is_transitional=transitional,
                execution_time_ms=float(rng.uniform(600, 2200)),
                observed_features={
                    "cells": int(min(2 + t, 40)),
                    "symmetry": round(float(rng.uniform(0.6, 0.98)), 2),
                },
            )
        print(f"  {eid}: {n_timepoints} timepoints, role={ROLES[i % len(ROLES)]}")

    return sid


NOTES = [
    (
        NoteKind.OBSERVATION,
        "e02 drifts in z after t6",
        "Focus wanders roughly 8 um over four timepoints on e02 only. e01 and e03 "
        "hold. Suspect stage settling rather than the sample.",
    ),
    (
        NoteKind.FINDING,
        "gastrula timing is consistent across the dish",
        "All four embryos reach gastrula within one timepoint of each other, which "
        "argues the thermal gradient across the coverslip is small.",
    ),
    (
        NoteKind.QUESTION,
        "is the 2-fold call reliable at this exposure?",
        "Confidence dips below 0.7 on every 2-fold prediction. Worth checking "
        "against ground truth before trusting the transition.",
    ),
    (
        NoteKind.OBSERVATION,
        "laser at 35% is enough for this strain",
        "No visible photobleaching over 40 timepoints at 35%/40ms.",
    ),
]

CAMPAIGN = (
    "Characterise developmental timing variance across a single coverslip",
    "timing-variance",
    "Do embryos on one dish develop in lockstep, and if not, does position predict the lag?",
)


def seed_context(root: Path, session_id: str) -> None:
    """Notebook notes and a campaign, so those windows have something to show."""
    try:
        cs = FileContextStore(root / "agent")
    except Exception as exc:  # pragma: no cover - seeding is best-effort
        print(f"  context store unavailable ({exc}); skipping notes/campaigns")
        return

    nb = getattr(cs, "notebook", None)
    if nb is not None:
        for kind, title, body in NOTES:
            nb.write_note(
                Note(
                    id="",
                    kind=kind,
                    title=title,
                    body=body,
                    author=Author.AGENT,
                    status=NoteStatus.OPEN if kind is NoteKind.QUESTION else NoteStatus.CONFIRMED,
                    sessions=[session_id],
                    embryos=["e02"] if "e02" in title else [],
                    threads=["timing-variance"],
                )
            )
        print(f"  notebook: {len(NOTES)} notes")

    desc, shorthand, summary = CAMPAIGN
    cs.create_campaign(description=desc, shorthand=shorthand, summary=summary)
    print("  campaign: 1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embryos", type=int, default=4)
    ap.add_argument("--timepoints", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(settings.storage.base_path)
    if not root.is_absolute():
        print(
            f"refusing to seed a relative storage path ({root!r}).\n"
            "Set GENTLY_STORAGE_PATH — see CLAUDE.md on the D:/ trap off-Windows.",
            file=sys.stderr,
        )
        return 2

    store = FileStore(root)
    sid = seed(store, n_embryos=args.embryos, n_timepoints=args.timepoints, seed_val=args.seed)
    seed_context(root, sid)
    print(f"\nseeded {args.embryos} embryos x {args.timepoints} timepoints into {root}")
    print(f"resume it at http://localhost:8080/?atrium=1  (session {sid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
