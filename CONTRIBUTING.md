# Contributing to Gently

## Code quality toolchain

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [pre-commit](https://pre-commit.com/) to enforce this automatically before every commit.

> **Note:** Type checking with mypy is not yet enforced. See the tracking issue for gradually
> introducing mypy across the codebase.

### First-time setup

Install the dev dependencies (includes ruff and pre-commit):

```bash
uv sync
```

Then install the pre-commit hooks:

```bash
pre-commit install
```

From this point on, ruff runs automatically on staged files whenever you `git commit`.

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

### CI

Every pull request runs the lint job (`.github/workflows/lint.yml`), which checks ruff lint and formatting across the entire project. Fix any failures locally with `pre-commit run --all-files` before pushing.
