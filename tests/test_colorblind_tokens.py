"""Status tokens in main.css stay distinguishable under CVD and readable on cards."""

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "cvd_check", Path(__file__).resolve().parents[1] / "tools" / "cvd_check.py"
)
assert _spec is not None and _spec.loader is not None
cvd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cvd)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_status_tokens_are_cvd_safe_and_contrasting(theme):
    fails = cvd.check(cvd.parse_css_tokens(cvd.MAIN_CSS, cvd.THEMES[theme]))
    assert not fails, "\n".join(f"{theme}: {f}" for f in fails)
