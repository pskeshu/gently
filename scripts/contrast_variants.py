"""Generate side-by-side contrast variants for a single labelled frame.

Helps pick a preprocessing recipe where blank frames stay blank but the
embryo body autofluorescence is visible to the VLM.
"""

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

SESSION = Path(r"D:/Gently3/sessions/20260522_1801_unnamed_ec5ea7ba")
OUT = Path("./contrast_variants")
OUT.mkdir(exist_ok=True)


def preprocess(proj: np.ndarray, recipe: str) -> np.ndarray:
    """Apply one preprocessing variant to a 2D projection."""
    arr = proj.astype(np.float32)
    if recipe == "current_div16":  # current production
        arr = np.maximum(arr - 100.0, 0.0)
        return np.clip(arr / 16.0, 0, 255).astype(np.uint8)
    if recipe == "div8":
        arr = np.maximum(arr - 100.0, 0.0)
        return np.clip(arr / 8.0, 0, 255).astype(np.uint8)
    if recipe == "div4":
        arr = np.maximum(arr - 100.0, 0.0)
        return np.clip(arr / 4.0, 0, 255).astype(np.uint8)
    if recipe == "log":
        arr = np.maximum(arr - 100.0, 1.0)
        return np.clip(np.log10(arr) * 80.0, 0, 255).astype(np.uint8)
    if recipe == "gamma":  # divisor 8 + gamma 0.5
        arr = np.maximum(arr - 100.0, 0.0)
        scaled = np.clip(arr / 8.0, 0, 255) / 255.0
        return np.clip(np.power(scaled, 0.5) * 255.0, 0, 255).astype(np.uint8)
    if recipe == "soft_log":  # gentler log: log10(x+1)*60
        arr = np.maximum(arr - 100.0, 0.0)
        return np.clip(np.log10(arr + 1.0) * 60.0, 0, 255).astype(np.uint8)
    raise ValueError(recipe)


CASES = [
    ("embryo_004", 1),  # blank baseline — must stay dark
    ("embryo_003", 20),  # pre-onset — body should be visible
    ("embryo_003", 28),  # mid-onset — body + puncta
    ("embryo_003", 40),  # mature — body + puncta + neurites
]

RECIPES = ["current_div16", "div8", "div4", "log", "gamma", "soft_log"]


def make_projection(volume):
    vol = np.squeeze(volume)
    if vol.ndim == 4:
        vol = vol[0]
    proj = np.max(vol, axis=0) if vol.ndim == 3 else vol
    # side-A crop (left half) for dual-view layout
    h, w = proj.shape[-2], proj.shape[-1]
    if w >= 2 * h:
        proj = proj[..., : w // 2]
    return proj


def main():
    for embryo, tp in CASES:
        path = SESSION / "embryos" / embryo / "volumes" / f"t{tp:04d}.tif"
        if not path.exists():
            continue
        vol = tifffile.imread(str(path))
        proj = make_projection(vol)
        print(
            f"\n{embryo} t{tp:04d}: shape={proj.shape}"
            f" min={proj.min()} max={proj.max()} p99={np.percentile(proj, 99):.0f}"
        )
        for recipe in RECIPES:
            out_img = preprocess(proj, recipe)
            p = OUT / f"{embryo}_t{tp:04d}_{recipe}.png"
            Image.fromarray(out_img).save(p)
            print(
                f"  {recipe:<15}  display p50={np.percentile(out_img, 50):.0f}"
                f" p90={np.percentile(out_img, 90):.0f} max={out_img.max()}"
            )


if __name__ == "__main__":
    main()
