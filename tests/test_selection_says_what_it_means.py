"""Two kinds of highlighted embryo are told apart by words, not by shade.

"it is not clear what a selected or highlighted embryo means, as there are
two selections one lighter and another darker"

With several embryos selected the roster has two kinds of highlighted row:
the members of the set a run will image, and the one the instrument acts on.
They were a lighter blue and a darker blue, and nothing said which was which.
"""

from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
ROSTER = (WEB / "static" / "js" / "panels" / "roster.js").read_text(encoding="utf-8")
CSS = (WEB / "static" / "css" / "operate.css").read_text(encoding="utf-8")


def _fn(name: str, length: int = 1400) -> str:
    return ROSTER[ROSTER.index(f"    function {name}(") :][:length]


def test_each_highlighted_row_carries_its_word():
    fn = _fn("mark")
    assert "'target'" in fn and "'selected'" in fn
    assert "${mark(emb, selected, inSet, opts)}" in ROSTER


def test_the_word_says_what_it_does():
    fn = _fn("mark")
    assert "The instrument acts on this one" in fn
    assert "Selected for a run" in fn


def test_one_embryo_selected_needs_no_explaining():
    assert "if (inSet.size < 2 || !inSet.has(emb.id)) return '';" in _fn("mark")
    assert "if (members.length < 2) return '';" in _fn("legend")


def test_the_list_says_it_once_above_itself():
    fn = _fn("legend")
    assert "selected</b> for a run" in fn
    assert "The instrument acts on" in fn
    assert "host.innerHTML = legend(embryos, selected, inSet)" in ROSTER


def test_it_says_how_the_set_is_changed():
    assert "Ctrl-click adds or takes out" in _fn("legend")


def test_the_narrow_rail_keeps_the_name_on_one_line():
    assert "opts.compact ? (isTarget ? '◎' : '')" in _fn("mark")


def test_the_target_is_marked_by_more_than_a_shade():
    assert ".rp-mark.is-target {" in CSS
    assert ".rp-row.is-primary:has(.rp-mark) { box-shadow: inset 3px 0 0 var(--accent); }" in CSS


def test_the_names_are_escaped():
    assert "Embryo ${esc(labelOf(target))}" in _fn("legend")
