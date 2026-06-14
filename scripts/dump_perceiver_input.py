"""Save the exact PNG the dopaminergic perceiver receives.

Re-runs _volume_to_b64 on raw volumes from a session and writes the
resulting PNG to ``./perceiver_input/<embryo>_t<NNNN>.png`` so we can
visually compare what the VLM sees against the saved projection JPG and
against the user's expectation.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gently.app.detectors.dopaminergic_signal import _volume_to_b64  # noqa: E402

SESSION = Path(r"D:/Gently3/sessions/20260522_1801_unnamed_ec5ea7ba")
OUT = Path("./perceiver_input")
OUT.mkdir(exist_ok=True)

CASES = [
    ("embryo_003", 20),
    ("embryo_003", 25),
    ("embryo_003", 28),
    ("embryo_003", 40),
    ("embryo_004", 1),  # known false positive baseline
]


def main():
    for embryo, tp in CASES:
        path = SESSION / "embryos" / embryo / "volumes" / f"t{tp:04d}.tif"
        if not path.exists():
            print(f"missing {path}")
            continue
        vol = tifffile.imread(str(path))
        b64 = _volume_to_b64(vol, calibration=None)
        if b64 is None:
            print(f"failed {embryo} t{tp:04d}")
            continue
        out_path = OUT / f"{embryo}_t{tp:04d}.png"
        out_path.write_bytes(base64.b64decode(b64))
        print(f"wrote {out_path}  ({len(base64.b64decode(b64)):,} bytes, vol shape={vol.shape})")


if __name__ == "__main__":
    main()
