"""Detector registry — maps role's ``detector_name`` to a Detector factory.

Roles in ``gently.harness.roles.REGISTRY`` declare a ``detector_name``
string (e.g. ``"dopaminergic_signal"`` for test, ``"perception"`` for
calibration). The orchestrator looks up the corresponding Detector
instance here for each acquired volume.
"""

from collections.abc import Callable

from .base import Detector

# Factory signature: (claude_client=None, perceiver=None) -> Detector
DetectorFactory = Callable[..., Detector]


def _make_dopaminergic(*, claude_client=None, perceiver=None, **_) -> Detector:
    from .dopaminergic_signal import DopaminergicSignalDetector

    return DopaminergicSignalDetector(claude_client=claude_client)


def _make_hatching(*, claude_client=None, perceiver=None, **_) -> Detector:
    from .hatching import HatchingDetector

    return HatchingDetector(claude_client=claude_client)


def _make_blank(*, claude_client=None, perceiver=None, **_) -> Detector:
    from .blank_image import BlankImageDetector

    return BlankImageDetector(claude_client=claude_client)


def _make_perception(*, claude_client=None, perceiver=None, **_) -> Detector:
    from .perception_proxy import PerceptionProxy

    return PerceptionProxy(perceiver=perceiver)


DETECTOR_REGISTRY: dict[str, DetectorFactory] = {
    "dopaminergic_signal": _make_dopaminergic,
    "hatching": _make_hatching,
    "blank_image": _make_blank,
    "perception": _make_perception,
}


def get_detector(
    name: str | None,
    *,
    claude_client=None,
    perceiver=None,
) -> Detector | None:
    """Return a Detector instance for ``name``, or None if unknown.

    Unknown / None names return None so the orchestrator can choose
    to skip detection rather than crash.
    """
    if not name:
        return None
    factory = DETECTOR_REGISTRY.get(name)
    if factory is None:
        return None
    return factory(claude_client=claude_client, perceiver=perceiver)
