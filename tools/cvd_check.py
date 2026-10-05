"""Colour-vision-deficiency guardrail for the web UI's status tokens.

Stdlib only. Simulates protan/deutan/tritan (Machado 2009, severity 1.0)
in linear RGB, measures CIE76 ΔE between every pair of the five status
tokens, and WCAG contrast of every accent token against --bg-card.
Run: ``uv run python tools/cvd_check.py`` — exit 1 on any failure.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MAIN_CSS = Path(__file__).resolve().parents[1] / "gently/ui/web/static/css/main.css"
STATUS = ("--accent-green", "--accent-amber", "--accent", "--accent-red", "--text-muted")
MIN_DELTA_E = 20.0
MIN_CONTRAST = 4.5

Mat = tuple[tuple[float, float, float], ...]
CVD: dict[str, Mat] = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}
RGB_TO_XYZ: Mat = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
D65 = (0.95047, 1.0, 1.08883)


def _mul(m: Mat, v: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(a * b for a, b in zip(row, v, strict=True)) for row in m)  # type: ignore[return-value]


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def to_srgb(c: float) -> float:
    c = min(1.0, max(0.0, c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def simulate(rgb: tuple[float, float, float], kind: str) -> tuple[float, float, float]:
    lin = tuple(to_linear(c) for c in rgb)
    return tuple(to_srgb(c) for c in _mul(CVD[kind], lin))  # type: ignore[arg-type, return-value]


def to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    xyz = _mul(RGB_TO_XYZ, tuple(to_linear(c) for c in rgb))  # type: ignore[arg-type]

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116

    fx, fy, fz = (f(v / w) for v, w in zip(xyz, D65, strict=True))
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def delta_e(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(to_lab(a), to_lab(b), strict=True)) ** 0.5


def luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = (to_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def parse_css_tokens(path: Path, selector_prefix: str) -> dict[str, str]:
    """``--name: #hex;`` tokens from the first ``{...}`` block whose selector starts with prefix."""
    css = path.read_text()
    m = re.search(re.escape(selector_prefix) + r"[^{]*\{(.*?)\}", css, re.S)
    if not m:
        raise ValueError(f"no block for {selector_prefix!r} in {path}")
    return dict(re.findall(r"(--[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;", m.group(1)))


THEMES = {"dark": ':root, [data-theme="dark"]', "light": '[data-theme="light"]'}


def check(tokens: dict[str, str]) -> list[str]:
    """Return failure messages for one theme's tokens."""
    fails = []
    for i, a in enumerate(STATUS):
        for b in STATUS[i + 1 :]:
            for kind in CVD:
                de = delta_e(
                    simulate(hex_to_rgb(tokens[a]), kind), simulate(hex_to_rgb(tokens[b]), kind)
                )
                if de < MIN_DELTA_E:
                    fails.append(f"{a} vs {b} under {kind}: ΔE {de:.1f} < {MIN_DELTA_E}")
    bg = hex_to_rgb(tokens["--bg-card"])
    for name, hx in tokens.items():
        if name.startswith("--accent") and name != "--accent-soft":  # soft is a fill, not text
            cr = contrast(hex_to_rgb(hx), bg)
            if cr < MIN_CONTRAST:
                fails.append(
                    f"{name} {hx} on --bg-card {tokens['--bg-card']}: "
                    f"contrast {cr:.2f} < {MIN_CONTRAST}"
                )
    return fails


def main() -> int:
    ok = True
    for theme, sel in THEMES.items():
        tokens = parse_css_tokens(MAIN_CSS, sel)
        print(f"\n== {theme} ==  " + "  ".join(f"{t}={tokens[t]}" for t in STATUS))
        print(f"{'pair':38s} {'normal':>7s} {'protan':>7s} {'deutan':>7s} {'tritan':>7s}")
        for i, a in enumerate(STATUS):
            for b in STATUS[i + 1 :]:
                ra, rb = hex_to_rgb(tokens[a]), hex_to_rgb(tokens[b])
                row = [delta_e(ra, rb)] + [delta_e(simulate(ra, k), simulate(rb, k)) for k in CVD]
                print(f"{a + ' vs ' + b:38s} " + " ".join(f"{v:7.1f}" for v in row))
        bg = hex_to_rgb(tokens["--bg-card"])
        print(
            "contrast on --bg-card:",
            "  ".join(
                f"{n}={contrast(hex_to_rgb(h), bg):.2f}"
                for n, h in tokens.items()
                if n.startswith("--accent") and n != "--accent-soft"
            ),
        )
        for f in check(tokens):
            ok = False
            print("FAIL", theme, f)
    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    # self-check: pure red/green collapse under deutan, black/white don't
    assert (
        delta_e(
            simulate(hex_to_rgb("#ff0000"), "deutan"), simulate(hex_to_rgb("#00ff00"), "deutan")
        )
        < 30
    )
    assert delta_e(hex_to_rgb("#000000"), hex_to_rgb("#ffffff")) > 99
    assert abs(contrast(hex_to_rgb("#000000"), hex_to_rgb("#ffffff")) - 21) < 0.01
    sys.exit(main())
