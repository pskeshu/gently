"""
Embryo role registry — declarative taxonomy for embryos in a session.

Roles drive multiple behaviors:
- Cadence policy (default interval, fast/burst eligibility)
- Detector selection (which observation pipeline runs each round)
- Photodose budget (per-role total exposure ceiling multiplier)
- UI presentation (color, icon on the map view)

Extend ``REGISTRY`` to add new roles. An ``EmbryoState.role`` field stores
the role name as a string key into this registry; validation happens at
assignment time via :func:`get_role`.

Built-in roles:
- ``test``: the biological subject (precious sample). Custom ad-hoc detector.
- ``calibration``: reference embryo used for staging/calibration. Absorbs
  more photodose. Standard perception pipeline.
- ``unassigned``: explicit "not yet classified" state. Treated like ``test``
  for safety until the user resolves the assignment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EmbryoRole:
    """Declarative role specification.

    Frozen so role definitions are immutable references after registry build.
    """
    name: str
    description: str
    default_cadence_seconds: float = 300.0
    detector_name: Optional[str] = None
    photodose_budget_multiplier: float = 1.0
    ui_color: str = "#888888"
    ui_icon: str = "circle"
    # After N consecutive "no_object" detections, treat the embryo as
    # gone (likely hatched / drifted out of FOV) and terminate imaging.
    # None means "never terminate on consecutive no_object" — keep the
    # existing recheck-forever behavior. Calibration embryos don't
    # drift back; once they're out of view they stay out, so they get
    # a short threshold. Test embryos can occasionally pop out and
    # back, so they get a longer one.
    no_object_consecutive_terminal: Optional[int] = None


REGISTRY: Dict[str, EmbryoRole] = {
    "unassigned": EmbryoRole(
        name="unassigned",
        description="No role assigned yet — treated like 'test' for safety.",
        default_cadence_seconds=300.0,
        detector_name=None,
        photodose_budget_multiplier=1.0,
        ui_color="#888888",
        ui_icon="circle",
        no_object_consecutive_terminal=None,
    ),
    "test": EmbryoRole(
        name="test",
        description=(
            "Biological subject (precious sample). For experiments like the "
            "dopaminergic-reporter run, these are the embryos carrying the "
            "reporter; they accelerate on signal onset and become burst-mode "
            "candidates."
        ),
        default_cadence_seconds=300.0,
        detector_name="dopaminergic_signal",  # filled in by Phase 2
        photodose_budget_multiplier=1.0,
        ui_color="#ff66cc",  # magenta
        ui_icon="star",
        no_object_consecutive_terminal=5,  # forgiving — they might drift back
    ),
    "calibration": EmbryoRole(
        name="calibration",
        description=(
            "Reference embryo used for staging and two-point + edge "
            "calibration. Absorbs more photodose (decoy budget). Runs the "
            "standard nuclear-marker perception pipeline."
        ),
        default_cadence_seconds=300.0,
        detector_name="perception",
        photodose_budget_multiplier=10.0,
        ui_color="#00cccc",  # cyan
        ui_icon="diamond",
        no_object_consecutive_terminal=2,  # they don't drift back; gone == gone
    ),
}

DEFAULT_ROLE: str = "test"


def get_role(name: str) -> EmbryoRole:
    """Look up a role by name. Raises KeyError with helpful message."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown embryo role: {name!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]


def is_valid_role(name: str) -> bool:
    return name in REGISTRY


def list_roles() -> List[str]:
    """All registered role names, sorted."""
    return sorted(REGISTRY.keys())
