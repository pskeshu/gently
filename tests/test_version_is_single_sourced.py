"""The version is written in one place, and the UI reports the build exactly.

Feedback is only useful if it names a build. Two problems make that fail:

1. A version literal in more than one file. `pyproject.toml` and
   `gently/__init__.py` each carried `"1.0.0.dev0"`, so a release could ship a
   wheel saying one thing and a UI saying another, and nothing would notice.
   `gently/_version.py` is now the only literal; pyproject derives it via
   `[tool.setuptools.dynamic]`.

2. A version too coarse to identify a tree. Everything between the
   `v1.0.0.dev1` tag and the next one calls itself `1.0.0.dev1`, so
   `build_id()` appends the commit inside a checkout.

Both are guarded here rather than in a release checklist, because a checklist
is what fails at 2am before a handoff.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]

# PEP 440, the subset this project uses: N(.N)* with an optional .devN / rcN.
PEP440 = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def test_pyproject_does_not_carry_its_own_literal() -> None:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = cfg["project"]
    assert project.get("version") is None, (
        "pyproject.toml has a hardcoded version again — it will drift from gently/_version.py"
    )
    assert "version" in project.get("dynamic", []), "pyproject no longer declares a dynamic version"
    assert cfg["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "gently._version.__version__"
    }, "the dynamic version no longer points at gently/_version.py"


def test_only_version_module_declares_the_number() -> None:
    """`gently/__init__.py` must re-export, never redeclare."""
    src = (ROOT / "gently" / "__init__.py").read_text(encoding="utf-8")
    assert "from gently._version import" in src, "__init__.py no longer re-exports the version"
    for n, line in enumerate(src.splitlines(), 1):
        if line.startswith("__version__") and "=" in line and "import" not in line:
            raise AssertionError(f"__init__.py:{n} redeclares the version: {line.strip()}")


def test_version_module_stays_importable_without_dependencies() -> None:
    """setuptools falls back to importing this file; it must import cleanly.

    Anything at module scope beyond the literal risks a build that needs the
    whole runtime installed just to read a number.
    """
    src = (ROOT / "gently" / "_version.py").read_text(encoding="utf-8")
    toplevel = [
        line
        for line in src.splitlines()
        if line.startswith(("import ", "from ")) and not line.startswith("from __future__")
    ]
    assert not toplevel, f"gently/_version.py grew top-level imports: {toplevel}"


def test_version_is_pep440() -> None:
    from gently._version import __version__

    assert PEP440.match(__version__), f"{__version__!r} is not a version pip will accept"


def test_build_id_extends_the_version_and_never_replaces_it() -> None:
    from gently._version import __version__, build_id

    bid = build_id()
    assert bid.startswith(__version__), (
        f"build_id() {bid!r} does not start with {__version__!r} — a report "
        "quoting it would not name the release"
    )
    extra = bid[len(__version__) :]
    assert extra == "" or re.match(r"^\+g[0-9a-f]{7,}(-dirty)?$", extra), (
        f"unexpected build id suffix {extra!r}"
    )


if __name__ == "__main__":
    test_pyproject_does_not_carry_its_own_literal()
    test_only_version_module_declares_the_number()
    test_version_module_stays_importable_without_dependencies()
    test_version_is_pep440()
    test_build_id_extends_the_version_and_never_replaces_it()
    print("ok")
