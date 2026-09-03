# Contributing to Gently

## Code quality toolchain

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and
[mypy](https://mypy-lang.org/) for type checking, enforced automatically before every
commit via [pre-commit](https://pre-commit.com/).

### First-time setup

Install the dev dependencies (includes ruff, mypy, and pre-commit):

```bash
uv sync
```

Then install the pre-commit hooks:

```bash
pre-commit install
```

From this point on, ruff runs on staged files and mypy runs across the whole
project whenever you `git commit`.

### Running manually

To check all files at once (useful before opening a PR):

```bash
pre-commit run --all-files
```

Or run the tools directly:

```bash
ruff check .          # lint
ruff format .         # format in-place
```

### Keeping hooks up to date

To update hook versions to their latest releases:

```bash
pre-commit autoupdate
```

### Type checking

mypy runs two ways, and they can disagree:

- `mypy .` — no project dependencies installed. `[tool.mypy]` sets
  `ignore_missing_imports = true`, so third-party imports fall back to `Any`.
  This is the form the pre-commit hook runs.
- `uv run mypy .` — after `uv sync`, with the real packages present, so mypy
  checks their actual types and can surface mismatches the deps-less run cannot.

A green run of one says nothing about the other, so run `uv run mypy .` before
pushing if you touched code that uses a typed third-party library. New code must
type-clean under both.

### CI

`.github/workflows/lint.yml` is the source of truth for what gates a pull
request, and it changes — read it rather than trusting a summary here.
`pre-commit run --all-files` reproduces the ruff checks and the deps-less
`mypy .` locally, but **not** the deps-installed run; reproduce that with
`uv run mypy .` after `uv sync`.

## Releasing

Feedback is only actionable if it names a build. Keep that true.

### Where the version lives

`gently/_version.py` — **one literal, nowhere else.** `pyproject.toml` derives
it via `[tool.setuptools.dynamic]`, `gently/__init__.py` re-exports it, and the
web server renders it. `tests/test_version_is_single_sourced.py` fails if a
second copy reappears.

### What the UI shows

`build_id()` — the version plus the commit, e.g. `1.0.0.dev1+g92816ea`, and
`-dirty` when the tree had uncommitted tracked changes. It appears on the
launch gate footer, on the Settings page, and as the OpenAPI version at
`/openapi.json`. Outside a git checkout the bare version is shown.

**Ask for the string from the launch gate in every bug report.** `1.0.0.dev1`
alone names every commit between two tags; the suffix names exactly one, and
`-dirty` distinguishes a real bug from someone's half-finished edit.

### Cutting a release

```bash
# 1. Bump the one literal
$EDITOR gently/_version.py           # e.g. 1.0.0.dev1 -> 1.0.0.dev2

# 2. Prove the package agrees with the module
uv build --wheel --out-dir /tmp/gently-build
ls /tmp/gently-build                 # filename must carry the new version

# 3. Land it through a PR like anything else, then tag the merge commit
git tag -a v1.0.0.dev2 -m "gently 1.0.0.dev2"
git push upstream v1.0.0.dev2
gh release create v1.0.0.dev2 --repo gently-project/gently \
    --target development --title "gently 1.0.0.dev2" --notes-file <notes>
```

Tag names are `v` + the literal, so `git tag` and `_version.py` read the same.
Tag the commit that is actually on `development` after the merge — not the tip
of the feature branch, which nobody else will ever check out.

### Version numbers

`1.0.0.devN` while 1.0.0 is being iterated with a reviewer; `1.0.0rcN` once the
milestone is empty and only regressions are being fixed; `1.0.0` at release.
Bump `devN` for each build handed to someone else, so their feedback has a
distinct number to attach to.
