"""Dry-run the two-stage dopaminergic detector against yesterday's
hand-labelled frames.

Loads raw .tif volumes from session 20260522_1801_unnamed_ec5ea7ba and
runs them through the actual DopaminergicSignalDetector.run() pipeline.
Prints the perceiver's description and the classifier's JSON for each
frame, alongside the human-annotated ground truth.

Usage: python scripts/dry_run_dopaminergic_detector.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import anthropic
import tifffile

# Make the local package importable when run from repo root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gently.app.detectors.dopaminergic_signal import DopaminergicSignalDetector  # noqa: E402

SESSION = Path(r"D:/Gently3/sessions/20260522_1801_unnamed_ec5ea7ba")

# (embryo, timepoint, expected behaviour from human annotation)
CASES = [
    ("embryo_004", 1, "FALSE POSITIVE: blank embryo, no puncta. Expect NONE / NONE."),
    ("embryo_003", 20, "Just before onset. Expect NONE or WEAK / NONE."),
    ("embryo_003", 21, "Onset starting. Expect WEAK / NONE or PARTIAL."),
    ("embryo_003", 25, "Two bulbs visible. Expect MEDIUM / PARTIAL."),
    ("embryo_003", 26, "Three bulbs. Expect MEDIUM / PARTIAL."),
    ("embryo_003", 28, "Neurite extension starting. Expect MEDIUM / PARTIAL or GOOD."),
    (
        "embryo_003",
        29,
        "Neurite extending further. Expect MEDIUM-STRONG / PARTIAL or GOOD.",
    ),
    (
        "embryo_003",
        40,
        "Mature pattern, multiple bright bodies + neurites. Expect STRONG / GOOD.",
    ),
]


def _load_volume(embryo: str, timepoint: int):
    path = SESSION / "embryos" / embryo / "volumes" / f"t{timepoint:04d}.tif"
    if not path.exists():
        return None, f"missing {path}"
    return tifffile.imread(str(path)), None


async def _run_one(detector, claude, embryo: str, timepoint: int, expected: str):
    vol, err = _load_volume(embryo, timepoint)
    if err:
        print(f"  SKIP {embryo} t{timepoint:04d}: {err}")
        return

    print(f"\n{'=' * 80}")
    print(f"{embryo} t{timepoint:04d}  ({vol.shape}, dtype={vol.dtype})")
    print(f"EXPECTED: {expected}")
    print(f"{'-' * 80}")

    context = {
        "embryo_id": embryo,
        "timepoint": timepoint,
        "claude": claude,
        "calibration": None,  # no calibration -> dynamic-range scaling
    }
    result = await detector.run(vol, context)

    raw = result.raw_response or {}
    if isinstance(raw, dict):
        description = raw.get("description", "(no description)")
        raw.get("classifier_raw", "")
    else:
        # legacy single-call shape
        description = "(no description — legacy single-call detector)"
        raw if isinstance(raw, str) else ""

    print("PERCEIVER (Stage 1):")
    print(description.strip())
    print("\nCLASSIFIER (Stage 2):")
    findings = result.findings or {}
    print(
        json.dumps(
            {
                "intensity_level": findings.get("intensity_level"),
                "structure_quality": findings.get("structure_quality"),
                "has_hatched": findings.get("has_hatched"),
                "reasoning": result.reasoning,
            },
            indent=2,
        )
    )
    if result.error:
        print(f"ERROR: {result.error}")
    print(f"elapsed: {result.elapsed_ms:.0f}ms")


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)
    claude = anthropic.Anthropic(api_key=api_key)
    detector = DopaminergicSignalDetector(claude_client=claude)

    print(f"Dry-running {len(CASES)} cases through DopaminergicSignalDetector")
    print(f"Session: {SESSION}")

    for embryo, tp, expected in CASES:
        await _run_one(detector, claude, embryo, tp, expected)

    print(f"\n{'=' * 80}\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
