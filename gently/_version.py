"""The single source of truth for Gently's version.

`pyproject.toml` reads `__version__` from here (`[tool.setuptools.dynamic]`),
`gently/__init__.py` re-exports it, and the web server renders it. The number
is written in exactly one place, so a release cannot ship a package that says
one version and a UI that says another.

This module deliberately imports nothing at the top level: setuptools reads
`__version__` by parsing the file, and falls back to *importing* it when the
parse fails. `gently/__init__.py` pulls in the whole harness, so an import
fallback there would need every runtime dependency present just to build a
wheel. A file with one literal in it can always be parsed.
"""

from __future__ import annotations

__version__ = "1.0.0.dev1"


def build_id() -> str:
    """`__version__`, plus the commit it is running from inside a checkout.

    A version string alone is not enough to route feedback while a dev release
    is being iterated: everything between the `v1.0.0.dev1` tag and the next
    one calls itself `1.0.0.dev1`, so "it's broken on dev1" can name a dozen
    different trees. The commit makes a bug report land on an exact one, and
    `-dirty` says the tree had uncommitted edits — which is the difference
    between a real bug and someone's half-finished change.

    Outside a checkout (an installed wheel, a copied directory) there is no
    commit to report and the bare version is the honest answer.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    if not (repo / ".git").exists():
        return __version__

    def _git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    try:
        head = _git("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return __version__
        commit = head.stdout.strip()
        # Tracked changes only — untracked screenshots and scratch files are
        # not a different build.
        dirty = "-dirty" if _git("diff", "--quiet", "HEAD").returncode == 1 else ""
    except (OSError, subprocess.SubprocessError):
        return __version__

    return f"{__version__}+g{commit}{dirty}"
