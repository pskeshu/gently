"""
Declarative monitoring modes — "we are anticipating an event" abstractions.

Each mode bundles a trigger predicate + on_trigger transition + applies_to
filter so the orchestrator's reactive package can be described as a small
typed object rather than a constellation of imperative add_*_rule calls.

Built-in modes:

- ``ExpressionMonitoringMode`` — dopaminergic-reporter onset on Test
  embryos. Watches the Claude detector's intensity_level; transitions
  the embryo to fast cadence + installs sticky-downward power ramp on
  saturation.
- ``PreTerminalMonitoringMode`` — wraps the existing
  ``enable_pre_hatching_speedup`` pattern declaratively.
- ``IdleMode`` — default no-op mode (every session has it active).

Activate a mode via ``TimelapseOrchestrator.enable_monitoring_mode(name)``.
"""

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class MonitoringMode:
    """A declarative bundle of (trigger, transition, scope).

    Concrete modes implement ``activate(orchestrator, embryo_ids)`` to
    install whatever rules / state they need. The orchestrator records
    the active modes so the UI / persistence layer can show what
    anticipation logic is in play.
    """

    name: str = ""
    description: str = ""
    applies_to_roles: list[str] = field(default_factory=list)

    def activate(self, orchestrator, embryo_ids: list[str] | None = None):
        """Install the mode's rules / detectors on the orchestrator."""
        raise NotImplementedError


@dataclass
class ExpressionMonitoringMode(MonitoringMode):
    """Anticipating fluorescent-reporter onset on TestEmbryos.

    On signal onset (intensity_level >= MEDIUM reported by the dopaminergic
    detector → ``lit_up`` pseudo-stage), the embryo's cadence accelerates
    to ``fast_interval`` seconds. Onset firing is per-embryo and requires
    ``onset_confirm_timepoints`` extra consecutive detections to suppress
    single-frame false positives. On sustained saturation
    (intensity_level == SATURATING), the 488 laser power steps down by
    ``rampdown_step_pct`` until ``rampdown_floor_pct``.

    Burst on good structure is handled by Phase 7's BurstAcquisition.
    """

    fast_interval: float = 60.0
    onset_confirm_timepoints: int = 2
    rampdown_step_pct: float = 1.0
    rampdown_floor_pct: float = 2.0
    rampdown_ceiling_pct: float = 6.0
    # Auto-burst on stable structure (one-time per embryo)
    burst_frames: int = 60
    burst_mode: str = "1hz"
    burst_num_slices: int = 1
    burst_confirm_timepoints: int = 2

    def __post_init__(self):
        if not self.name:
            self.name = "expression_monitoring"
        if not self.description:
            self.description = (
                "Anticipating fluorescent-reporter onset on Test embryos: "
                f"accelerate to {self.fast_interval}s on signal "
                f"(confirm={self.onset_confirm_timepoints}), ramp 488 "
                f"down to {self.rampdown_floor_pct}% on saturation, "
                f"auto-burst ({self.burst_frames}f @ {self.burst_mode}) "
                f"on stable structure."
            )
        if not self.applies_to_roles:
            self.applies_to_roles = ["test"]

    def activate(self, orchestrator, embryo_ids: list[str] | None = None):
        orchestrator.add_test_onset_speedup(
            fast_interval=self.fast_interval,
            confirm_timepoints=self.onset_confirm_timepoints,
            embryo_ids=embryo_ids,
        )
        orchestrator.add_test_saturation_rampdown(
            step_pct=self.rampdown_step_pct,
            floor_pct=self.rampdown_floor_pct,
            ceiling_pct=self.rampdown_ceiling_pct,
            embryo_ids=embryo_ids,
        )
        orchestrator.add_test_burst_on_good_structure(
            frames=self.burst_frames,
            mode=self.burst_mode,
            num_slices=self.burst_num_slices,
            confirm_timepoints=self.burst_confirm_timepoints,
            embryo_ids=embryo_ids,
        )


@dataclass
class PreTerminalMonitoringMode(MonitoringMode):
    """Anticipating the pre-terminal stage (e.g. pretzel for C. elegans).

    Wraps ``enable_pre_hatching_speedup`` declaratively. Uses the
    organism's PRE_TERMINAL_SPEEDUP_STAGE as the trigger.
    """

    fast_interval: float = 30.0

    def __post_init__(self):
        if not self.name:
            self.name = "pre_terminal_monitoring"
        if not self.description:
            self.description = (
                "Anticipating the organism's pre-terminal stage: accelerate "
                f"to {self.fast_interval}s on detection."
            )

    def activate(self, orchestrator, embryo_ids: list[str] | None = None):
        orchestrator.add_pre_terminal_speedup(fast_interval=self.fast_interval)


@dataclass
class IdleMode(MonitoringMode):
    """No-op default — no anticipated event."""

    def __post_init__(self):
        if not self.name:
            self.name = "idle"
        if not self.description:
            self.description = "No active anticipation; standard timelapse cadence."

    def activate(self, orchestrator, embryo_ids: list[str] | None = None):
        pass


# Public registry. ``enable_monitoring_mode(name)`` on the orchestrator
# looks up by key.
MONITORING_MODES: dict[str, Callable[[], MonitoringMode]] = {
    "idle": IdleMode,
    "expression_monitoring": ExpressionMonitoringMode,
    "pre_terminal_monitoring": PreTerminalMonitoringMode,
}


def get_monitoring_mode(name: str) -> MonitoringMode | None:
    """Return an instance of the named monitoring mode, or None if unknown."""
    factory = MONITORING_MODES.get(name)
    if factory is None:
        return None
    return factory()
