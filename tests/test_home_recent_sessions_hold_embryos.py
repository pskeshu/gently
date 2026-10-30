"""The home view's recent sessions hold embryo data.

"in recent sessions in the main view of the gently, no need to show recent
sessions without any embryo data." A session opened and closed with nothing
in it is still in the Sessions tab; the landing surface shows the ones with
embryos, and the live one, whatever it holds, because it is the one in use.
The filter comes before the cap, so the five shown are five with data.

CI drives no browser; this reads the source.
"""

from __future__ import annotations

from pathlib import Path

HOME = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "home.js"
).read_text(encoding="utf-8")


def test_sessions_without_embryos_are_left_out_and_the_live_one_stays():
    assert ".filter(s => s.active || (s.embryo_count || 0) > 0)" in HOME


def test_the_filter_comes_before_the_cap():
    i, j = (
        HOME.index(".filter(s => s.active || (s.embryo_count"),
        HOME.index(".slice(0, SESSIONS_N)"),
    )
    assert i < j
