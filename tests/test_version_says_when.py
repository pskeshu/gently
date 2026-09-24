"""How old the running build is, not just which one it is.

The build id names a tree exactly — `1.0.0.dev1+ga90332d`. Only half of that
comes from git: the sha is HEAD, read at server start, but `1.0.0.dev1` is a
literal in `gently/_version.py` that happens to match a tag name. So the
version string cannot say how old the code is, and the question behind "which
version is this?" — on a microscope, usually "is this from before the change" —
had no answer on screen.

The date belongs to the commit the id already names, so it moves with it. A
tag date would not: it would have read "3 Sep" through every commit since.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from gently._version import __version__, build_date, build_id

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
LAUNCH = (WEB / "templates" / "launch.html").read_text(encoding="utf-8")
SETTINGS = (WEB / "templates" / "settings.html").read_text(encoding="utf-8")
SERVER = (WEB / "server.py").read_text(encoding="utf-8")


def test_the_date_is_iso_8601_with_an_offset_or_absent():
    """Parseable by `new Date()`, or nothing at all.

    None is the honest answer outside a checkout, or when no tag names this
    version. The templates render nothing for it rather than a placeholder.
    """
    stamp = build_date()
    if stamp is None:
        return
    parsed = datetime.fromisoformat(stamp)
    assert parsed.tzinfo is not None, (
        f"{stamp!r} has no offset, so it means a different moment to every reader"
    )


def test_it_does_not_change_the_build_id():
    """The build id's shape is load-bearing — it goes into bug reports.

    Appending a date to it would break `test_build_id_extends_the_version`,
    and would put a date on the clipboard when someone copies the id.
    """
    assert build_id().startswith(__version__)
    stamp = build_date()
    if stamp:
        assert stamp not in build_id()


def test_the_date_moves_with_the_commit_in_the_id():
    """Otherwise it describes a different tree than the sha beside it.

    This is the whole reason it is not the tag's date: `v1.0.0.dev1` was
    tagged once and has named every commit since, so a tag date would have
    read "3 Sep" on a build cut three weeks later.
    """
    import subprocess

    stamp = build_date()
    if stamp is None:
        return
    head = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    if head.returncode == 0 and head.stdout.strip():
        assert stamp == head.stdout.strip()


def test_the_date_is_not_inside_the_thing_that_gets_copied():
    """`#foot-version` is a button whose handler copies its own textContent.

    Anything put inside it lands on the clipboard and into a bug report, where
    the commit is wanted and the date is noise.
    """
    button = re.search(r'<button[^>]*id="foot-version".*?</button>', LAUNCH, re.S)
    assert button, "the copyable build id is gone"
    assert "gently_build_date" not in button.group(0)
    assert "foot-when" not in button.group(0)


def test_the_gate_shows_it_on_its_own_line():
    assert re.search(
        r'<div class="foot foot-when">built\s*<time id="foot-when"'
        r'\s*datetime="\{\{ gently_build_date \}\}"',
        LAUNCH,
    ), "the build date is not rendered as a dated <time> on its own line"
    assert "{% if gently_build_date %}" in LAUNCH, (
        "a checkout-less install would render the word 'tagged' and nothing else"
    )


def test_the_iso_survives_when_the_script_does_not():
    """The element keeps the true value in `datetime`.

    The script rewrites the text into the reader's own timezone — the commit
    carries the committer's offset, which elsewhere is a puzzle rather than a
    fact. If it never runs, the ISO string is still on screen and still true.
    """
    assert ">{{ gently_build_date }}</time>" in LAUNCH
    localise = re.search(r'getElementById\("foot-when"\)(.*?)\}\)\(\);', LAUNCH, re.S)
    assert localise, "nothing localises the stamp"
    assert 'getAttribute("datetime")' in localise.group(1)
    assert "toLocaleString" in localise.group(1)


def test_settings_carries_it_too():
    assert "built {{ gently_build_date }}" in SETTINGS


def test_the_server_hands_both_to_every_template():
    assert 'globals["gently_build_date"]' in SERVER
    assert "gently.build_date()" in SERVER
