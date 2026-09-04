"""The device-layer banner must not advertise detection it cannot perform.

The READY banner printed `Detection  SAM on cuda (loads on first use)` whether
or not `segment_anything` was importable or the checkpoint existed. Both loads
are lazy, so the truth arrived on the first Detect — mid-workflow, at the
microscope, with a sample already positioned. What the operator saw was a bare
`502`; the real message was `ModuleNotFoundError: No module named
'segment_anything'`, in a device-layer traceback in a different terminal.

This is not an exotic state. `segment-anything` lives in the `sam` extra, and
`uv sync` without that flag uninstalls it, so the environment is one forgotten
argument away from it at any time. Boot is the cheap place to find out.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

from gently.hardware.dispim.device_layer import DeviceLayerServer

# Deliberately called against a stub: the check reads two attributes and must
# not need a booted device layer to be testable.
readiness = cast(Any, DeviceLayerServer._sam_readiness)


class _Stub:
    """Only the two attributes the check reads."""

    _sam_device = "cuda"

    def __init__(self, checkpoint: str) -> None:
        self._sam_checkpoint = checkpoint


def test_missing_module_is_named_with_its_fix(tmp_path) -> None:  # noqa: ANN001
    ckpt = tmp_path / "sam_vit_b_01ec64.pth"
    ckpt.write_bytes(b"not a real checkpoint")

    with patch("gently.hardware.dispim.device_layer.find_spec", return_value=None):
        line, ok = readiness(_Stub(str(ckpt)))

    assert ok is False
    assert "UNAVAILABLE" in line
    assert "segment-anything not installed" in line
    # The remedy belongs in the message; the operator is at a microscope, not
    # in the dependency file.
    assert "--extra sam" in line


def test_missing_checkpoint_is_named_with_its_path(tmp_path) -> None:  # noqa: ANN001
    absent = tmp_path / "nope.pth"

    with patch("gently.hardware.dispim.device_layer.find_spec", return_value=object()):
        line, ok = readiness(_Stub(str(absent)))

    assert ok is False
    assert "UNAVAILABLE" in line
    assert "checkpoint not found" in line
    assert str(absent) in line
    assert "segment-anything not installed" not in line


def test_both_missing_are_reported_together(tmp_path) -> None:  # noqa: ANN001
    """One restart should reveal every reason, not the first one only."""
    with patch("gently.hardware.dispim.device_layer.find_spec", return_value=None):
        line, ok = readiness(_Stub(str(tmp_path / "nope.pth")))

    assert "segment-anything not installed" in line
    assert "checkpoint not found" in line


def test_a_working_install_still_reads_normally(tmp_path) -> None:  # noqa: ANN001
    ckpt = tmp_path / "sam_vit_b_01ec64.pth"
    ckpt.write_bytes(b"not a real checkpoint")

    with patch("gently.hardware.dispim.device_layer.find_spec", return_value=object()):
        line, ok = readiness(_Stub(str(ckpt)))

    assert line == "SAM on cuda (loads on first use)"
    assert ok is True
    assert "UNAVAILABLE" not in line


def test_the_check_never_imports_the_module() -> None:
    """`find_spec`, not `import` — this runs at boot and must not drag torch in."""
    import inspect

    src = inspect.getsource(DeviceLayerServer._sam_readiness)
    assert "find_spec" in src
    assert "import segment_anything" not in src
