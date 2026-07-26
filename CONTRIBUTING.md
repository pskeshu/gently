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
