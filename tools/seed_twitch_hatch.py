"""Seed brightfield frames into a session so the twitch-hatch tool has
something to annotate offline: F fields × N frames, five minutes apart, with
a temperature log over the same span.

    GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python tools/seed_twitch_hatch.py <session_id>
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from gently.core.file_store import FileStore
from gently.settings import settings


def main() -> int:
    sid = sys.argv[1]
    fields, frames = 2, 40
    store = FileStore(Path(settings.storage.base_path))
    rng = np.random.default_rng(0)
    t0 = datetime.now() - timedelta(minutes=5 * frames)
    for n in range(frames):
        at = t0 + timedelta(minutes=5 * n)
        for f in range(fields):
            img = rng.integers(80, 120, (256, 320), dtype=np.uint16)
            for k in range(3):  # three "embryos" per field, drifting a little
                cy, cx = 60 + 70 * k, 90 + 60 * f + n // 8
                img[cy - 12 : cy + 12, cx - 20 : cx + 20] = 40 + 3 * k
            meta = {
                "frame": n,
                "field": f,
                "fields": fields,
                "round": n,
                "captured_at": at.isoformat(),
            }
            store.put_snapshot(sid, "bf", img, metadata=meta, stem=f"f{f}_{n:04d}")
        store.append_temperature_sample(
            sid,
            {
                "t": at.isoformat(),
                "water_c": 20.0 + 0.1 * (n % 3),
                "setpoint_c": 20.0,
                "state": "ok",
            },
        )
    print(f"seeded {fields} fields × {frames} frames into {sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
