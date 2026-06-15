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

Run mypy the same way pre-commit and CI do:

```bash
mypy .
```

The codebase is being typed incrementally (see issue #46). Modules with
pre-existing errors are listed in the `[[tool.mypy.overrides]]` block in
`pyproject.toml` with `ignore_errors = true`, so `mypy .` passes today even
though not every module is fully typed yet.

Policy for working with this list:

- **New modules** must pass `mypy .` cleanly — do not add them to the
  overrides list.
- **PRs that substantively touch a module on the overrides list** should fix
  that module's type errors and remove it from the list as part of the
  change.

### CI

Every pull request runs the lint job (`.github/workflows/lint.yml`), which checks ruff lint and formatting and runs mypy across the entire project. Fix any failures locally with `pre-commit run --all-files` before pushing.
