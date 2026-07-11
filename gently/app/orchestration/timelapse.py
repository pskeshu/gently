"""
Timelapse Orchestrator for Adaptive Multi-Embryo Acquisition

Manages background timelapse acquisition with:
- Per-embryo stop conditions (e.g., hatching detection)
- Dynamic interval adjustment
- Non-blocking operation (agent stays responsive)
- Event-driven status updates
"""

import asyncio
import json
import logging
import traceback
from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from gently.core import EventType, get_event_bus
from gently.core.imaging import (
    apply_crop_bounds,
    compute_crop_bounds,
    image_to_base64,
    normalize_to_uint8,
    projection_three_view,
)
from gently.harness.error_log import GlobalErrorLog
from gently.harness.state import EmbryoState
from gently.organisms import get_organism
from gently.settings import settings

# Re-export models for backward compatibility
from .timelapse_models import (
    BurstRule,
    IntervalRule,
    PowerRule,
    StopCondition,
    StopConditionType,
    TimelapseState,
    TimelapseStatus,
)

if TYPE_CHECKING:
    from gently.core.file_store import FileStore

# Default trace directory
TRACE_BASE_PATH = settings.storage.traces_dir

logger = logging.getLogger(__name__)


class TimelapseOrchestrator:
    """
    Background timelapse manager

    Runs independently of agent conversation. The agent can:
    - Query status anytime via get_status()
    - Modify parameters via modify_embryo()
    - Stop embryos or entire timelapse via stop()

    Events are emitted for UI updates without blocking.
    """

    def __init__(
        self,
        microscope_client,
        experiment_state,
        perceiver=None,
        on_volume_callback: Callable | None = None,
        session_id: str | None = None,
        store: Optional["FileStore"] = None,
        claude_client=None,
        temperature_provider=None,
    ):
        """
        Parameters
        ----------
        microscope_client : QueueServerClient
            Client for hardware control
        experiment_state : ExperimentState
            Shared experiment state
        perceiver : gently_perception.Perceiver, optional
            VLM-based perception system for stage classification (used for
            ``role=calibration`` embryos via the PerceptionProxy detector).
        on_volume_callback : callable, optional
            Called after each volume: on_volume_callback(embryo_id, timepoint, volume)
        session_id : str, optional
            Session identifier for trace file storage
        store : FileStore, optional
            Unified data store for persisting perception predictions
        claude_client : anthropic.Anthropic, optional
            Anthropic SDK client for the ad-hoc Claude detectors
            (DopaminergicSignalDetector, HatchingDetector, BlankImageDetector).
            Required for ``role=test`` detection.
        """
        self.client = microscope_client
        self.experiment = experiment_state
        self.perceiver = perceiver
        self.on_volume_callback = on_volume_callback
        self.claude_client = claude_client

        # Zero-arg callable returning the latest temperature sample dict (or None).
        # Threaded from the agent's TemperatureSampler so burst frames carry a
        # temperature block in their metadata.
        self._temperature_provider = temperature_provider

        # Trace file storage (writes JSON files to disk)
        self._session_id = session_id
        self._trace_dir: Path | None = None

        # Unified store for perception persistence
        self._store = store
        self._perception_run_id: int | None = None

        # Event bus for status updates
        self._event_bus = get_event_bus()

        # Timelapse state
        # Holds references to EmbryoState objects from self.experiment.embryos
        # for embryos currently active in this timelapse. Same objects, no
        # duplication of state.
        self._embryo_states: dict[str, EmbryoState] = {}
        self._status = TimelapseStatus.IDLE
        self._started_at: datetime | None = None
        self._total_timepoints = 0
        self._current_round = 0
        self._error_message: str | None = None

        # Round-based scheduling (global timing for all embryos)
        self._base_interval_seconds: float = 120.0
        self._total_pause_duration: timedelta = timedelta(0)
        self._pause_start: datetime | None = None

        # Control
        self._acquisition_task: asyncio.Task | None = None
        self._stop_requested = False

        # In-flight perception tasks. Acquisition fires perception tasks
        # as create_task() and moves on to the next embryo immediately, so
        # round-to-round throughput is bounded by volume acquisition time,
        # not VLM API latency. Tasks are tracked here so (a) stop() can
        # await them before returning and (b) we can cap concurrency if
        # the Claude API rate limit becomes a problem.
        self._perception_tasks: set[asyncio.Task] = set()

        # Interval adjustment rules
        self._interval_rules: list[IntervalRule] = []
        self._applied_rules: dict[str, set[str]] = {}  # embryo_id -> set of applied rule names
        self._interval_rule_consecutive: dict[
            str, dict[str, int]
        ] = {}  # embryo_id -> {rule_name: consecutive matches}

        # Phase 5 reactive control: per-line power rules (sticky-downward
        # ramp on saturation, etc). Evaluated alongside _interval_rules in
        # _check_adaptive_rules / _check_interval_rules.
        self._power_rules: list[PowerRule] = []
        self._power_rule_consecutive: dict[
            str, dict[str, int]
        ] = {}  # embryo_id -> {rule_name: count}

        # Auto-burst rules: fire queue_burst() once per embryo on a
        # structure/intensity predicate match. One-time semantics are
        # enforced by queue_burst via _burst_applied.
        self._burst_rules: list[Any] = []  # List[BurstRule] without forward import
        self._burst_rule_consecutive: dict[str, dict[str, int]] = {}

        # Active monitoring modes (declarative anticipation bundles).
        self._active_monitoring_modes: list[Any] = []

        # Phase 6: calibration data from CalibrationEmbryos. Populated by
        # ``run_calibration_pipelines()``; consumed by _run_detector as
        # detector context["calibration"].
        self._calibration_data: dict[str, Any] | None = None

        # Phase 8: per-role photodose budgets. ``base_dose_budget_ms`` is the
        # ceiling for a 1× role (e.g. test). Other roles get this scaled by
        # their EmbryoRole.photodose_budget_multiplier (e.g. calibration
        # gets 10×). None = no enforcement. See ``set_photodose_budget``.
        self._dose_budget_base_ms: float | None = None
        # Embryos that have hit the budget — set to paused once, emit a
        # STATUS_CHANGED, then leave alone (don't spam).
        self._dose_budget_exceeded: set[str] = set()

        # Async cadence state (Phase 4). Single-burst exclusion handled by
        # _burst_in_progress: while set to an embryo_id, _run_loop skips
        # all other embryos and runs only the burst executor (Phase 7).
        self._burst_in_progress: str | None = None

        # Phase 7: exclusive acquisitions (bursts, etc) — FIFO queue.
        self._exclusive_queue: deque = deque()
        # Track which embryos have already had a burst applied
        # (one-time per embryo by default).
        self._burst_applied: set[str] = set()

        # Global error log for cross-embryo hardware error correlation
        self.global_error_log = GlobalErrorLog()

    async def start(
        self,
        embryo_ids: list[str] | None = None,
        stop_condition: str = "manual",
        base_interval_seconds: float = 120.0,
        condition_value: Any = None,
    ) -> str:
        """
        Start timelapse in background

        Returns immediately with status message. The timelapse runs
        asynchronously and can be monitored via get_status().

        Parameters
        ----------
        embryo_ids : list of str
            Embryo IDs to image (None = all active embryos)
        stop_condition : str
            One of: "manual", "hatching", "comma", "timepoints:N", "duration:Xh"
        base_interval_seconds : float
            Default interval between acquisitions
        condition_value : any, optional
            Value for stop condition (e.g., number of timepoints)

        Returns
        -------
        str
            Status message
        """
        if self._status == TimelapseStatus.RUNNING:
            return (
                "Timelapse already running. Use stop() first or modify_embryo() to change"
                " parameters."
            )

        # Parse stop condition
        stop_cond = self._parse_stop_condition(stop_condition, condition_value)

        # Tolerate a comma-separated string: some agent tool calls pass
        # embryo_ids as "embryo_1,embryo_2" rather than a JSON list. Without
        # this, the membership check below iterates the string character by
        # character and reports every letter as a missing embryo.
        if isinstance(embryo_ids, str):
            embryo_ids = [e.strip() for e in embryo_ids.split(",") if e.strip()]

        # Get embryo list
        if not embryo_ids:
            embryo_ids = [e.id for e in self.experiment.embryos.values() if not e.should_skip]

        if not embryo_ids:
            return "No embryos to image. Add embryos first."

        # Validate embryos exist
        missing = [eid for eid in embryo_ids if eid not in self.experiment.embryos]
        if missing:
            return f"Embryos not found: {missing}"

        # Initialize timelapse runtime state on EmbryoState references.
        # No separate state object — orchestrator and agent share the same
        # EmbryoState instances.
        now = datetime.now()
        self._embryo_states = {}
        for eid in embryo_ids:
            embryo = self.experiment.embryos[eid]
            embryo.stop_condition = stop_cond
            embryo.is_complete = False
            embryo.completion_reason = None
            embryo.error_count = 0
            embryo.last_error = None
            embryo.detection_triggered_at = None
            embryo.detection_type = None
            embryo.no_object_since_timepoint = None

            # Async cadence init: the caller-supplied base_interval_seconds
            # is the intent for *this* timelapse run and always wins. We
            # broadcast it to every embryo so per-embryo state matches what
            # the agent told the user ("starting at 120s interval").
            #
            # NOTE: this DOES overwrite any pre-existing per-embryo
            # interval (e.g. from a prior modify_parameters call). That's
            # deliberate — the agent's explicit start() call is the most
            # recent intent. Phase 5 rules (test_onset_speedup etc.)
            # mutate per-embryo intervals AFTER start, and those changes
            # stick because rules go through transition_cadence.
            embryo.interval_seconds = base_interval_seconds
            embryo.cadence_phase = "normal"
            embryo.next_due_at = now  # image immediately on first tick
            self._embryo_states[eid] = embryo

        # Burst state is per-session; clear at start.
        self._burst_in_progress = None

        # Initialize trace directory for file-based persistence
        if self._session_id:
            self._trace_dir = TRACE_BASE_PATH / self._session_id
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Trace file storage enabled: {self._trace_dir}")

        # Create a perception run in the unified store
        self._perception_run_id = None
        if self._store and self._session_id and self.perceiver:
            try:
                self._perception_run_id = self._store.create_perception_run(
                    session_id=self._session_id,
                    name=f"timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    method="vlm_stage_classification",
                    model_name=settings.models.perception,
                    source="live",
                    config={
                        "stop_condition": stop_condition,
                        "interval": base_interval_seconds,
                    },
                )
                logger.info(f"Created perception run {self._perception_run_id} in FileStore")
            except Exception as e:
                logger.warning(f"Failed to create perception run in store: {e}")

        # Initialize round-based scheduling (global timing for all embryos)
        self._base_interval_seconds = base_interval_seconds
        self._started_at = datetime.now()
        self._total_pause_duration = timedelta(0)
        self._pause_start = None

        # Reset state (but preserve total timepoints from existing embryos)
        self._status = TimelapseStatus.RUNNING
        self._total_timepoints = sum(e.timepoints_acquired for e in self._embryo_states.values())
        self._current_round = -1  # Will become 0 on first round
        self._stop_requested = False
        self._error_message = None

        # Start background task
        self._acquisition_task = asyncio.create_task(self._run_loop())

        # Emit event
        self._emit_event(
            EventType.ACQUISITION_STARTED,
            {
                "embryo_ids": embryo_ids,
                "stop_condition": stop_condition,
                "interval_seconds": base_interval_seconds,
            },
        )

        logger.info(f"Started timelapse for {len(embryo_ids)} embryos")

        return (
            f"Started timelapse for {len(embryo_ids)} embryos\n"
            f"Stop condition: {stop_condition}\n"
            f"Interval: {base_interval_seconds}s\n"
            f"Use get_timelapse_status to monitor progress."
        )

    def _parse_stop_condition(self, condition: str, value: Any = None) -> StopCondition:
        """Parse stop condition string into StopCondition object"""
        condition = condition.lower().strip()

        # Handle legacy keyword forms
        if condition in ("until_hatching",):
            condition = "hatching"
        elif condition in ("until_comma",):
            condition = "comma"
        elif condition == "fixed_timepoints" and value is not None:
            return StopCondition.fixed_timepoints(int(value))
        elif condition == "duration" and value is not None:
            return StopCondition.duration_hours(float(value))

        try:
            return StopCondition.parse(condition)
        except ValueError:
            logger.warning(f"Unknown stop condition '{condition}', using manual")
            return StopCondition.manual()

    def _get_round_scheduled_time(self, round_num: int) -> datetime:
        """
        Get when a round should start, accounting for pauses.

        Parameters
        ----------
        round_num : int
            Round number (0-indexed)

        Returns
        -------
        datetime
            Scheduled time for the round
        """
        assert self._started_at is not None  # only called on a running timelapse
        base_time = self._started_at + timedelta(seconds=round_num * self._base_interval_seconds)
        return base_time + self._total_pause_duration

    def _get_elapsed_active_time(self) -> float:
        """Get elapsed time excluding pauses, in seconds"""
        now = datetime.now()
        assert self._started_at is not None  # only called on a running timelapse
        elapsed = (now - self._started_at).total_seconds()
        pause_seconds = self._total_pause_duration.total_seconds()
        return elapsed - pause_seconds

    # ------------------------------------------------------------------
    # Async cadence helpers
    # ------------------------------------------------------------------

    def _is_eligible(self, embryo) -> bool:
        """An embryo is eligible for the due-loop pick if it's active.

        ``paused`` phase skips it (over-budget, manual pause, or another
        embryo's burst is in flight). ``burst`` phase means an exclusive
        burst-executor handles it — the normal due-loop should skip too.

        Phase 8: enforces per-role photodose budgets. If
        ``_dose_budget_base_ms`` is set, any embryo whose
        ``total_exposure_ms`` exceeds ``base * role.photodose_budget_multiplier``
        gets paused, a STATUS_CHANGED is emitted once, and the embryo
        falls out of the eligible set.
        """
        if embryo.is_complete:
            return False
        if embryo.should_skip:
            return False
        phase = getattr(embryo, "cadence_phase", "normal")
        if phase in ("paused", "burst"):
            return False

        if self._dose_budget_base_ms is not None and embryo.id not in self._dose_budget_exceeded:
            from gently.harness.roles import REGISTRY as _ROLE_REGISTRY

            role_def = _ROLE_REGISTRY.get(getattr(embryo, "role", "test"))
            mult = role_def.photodose_budget_multiplier if role_def else 1.0
            budget = self._dose_budget_base_ms * mult
            if embryo.total_exposure_ms > budget:
                embryo.cadence_phase = "paused"
                self._dose_budget_exceeded.add(embryo.id)
                self._emit_event(
                    EventType.STATUS_CHANGED,
                    {
                        "embryo_id": embryo.id,
                        "change": "photodose_budget_exceeded",
                        "role": embryo.role,
                        "total_exposure_ms": embryo.total_exposure_ms,
                        "budget_ms": budget,
                        "multiplier": mult,
                    },
                )
                logger.warning(
                    "[%s] photodose budget exceeded: %.0f ms > %.0f ms "
                    "(role=%s, mult=%.1fx). Pausing.",
                    embryo.id,
                    embryo.total_exposure_ms,
                    budget,
                    embryo.role,
                    mult,
                )
                return False

        return True

    def set_photodose_budget(self, base_dose_budget_ms: float | None) -> str:
        """Set the per-role photodose ceiling.

        Each embryo's ``total_exposure_ms`` is checked against
        ``base_dose_budget_ms * role.photodose_budget_multiplier``. Test
        embryos use 1× (tight); calibration embryos default to 10× (decoy).

        Pass ``None`` to disable enforcement.
        """
        self._dose_budget_base_ms = base_dose_budget_ms
        self._dose_budget_exceeded.clear()
        if base_dose_budget_ms is None:
            return "Photodose budget enforcement disabled."
        return f"Photodose budget set: {base_dose_budget_ms:.0f} ms base (scaled per role)."

    def _pick_next_due(self) -> tuple:
        """Pick the most-overdue eligible embryo.

        Returns
        -------
        (embryo, wait_seconds) : tuple
            ``embryo`` is the EmbryoState ready to acquire now (next_due_at
            <= now), or None if no embryo is due. When None, ``wait_seconds``
            is how long until the soonest-due embryo (capped at 5s for
            responsiveness). When (embryo, 0.0), acquire immediately.
        """
        now = datetime.now()
        eligible = [e for e in self._embryo_states.values() if self._is_eligible(e)]
        if not eligible:
            return None, 5.0  # nothing to do; check back later

        # Embryos with no next_due_at yet: schedule them immediately.
        for e in eligible:
            if e.next_due_at is None:
                e.next_due_at = now

        # Find the smallest next_due_at (all are set to a datetime just above)
        soonest = min(eligible, key=lambda e: e.next_due_at or now)
        assert soonest.next_due_at is not None
        if soonest.next_due_at <= now:
            return soonest, 0.0

        wait_s = (soonest.next_due_at - now).total_seconds()
        return None, min(wait_s, 5.0)

    def _reschedule(self, embryo, *, from_now: bool = True) -> None:
        """Set ``embryo.next_due_at`` to now + interval_seconds.

        ``from_now=True`` (default) anchors at the actual current time —
        the right choice when acquisitions can take noticeable time. If
        you'd rather anchor at the previous due-time (catch up on missed
        rounds after a burst), pass False.
        """
        interval_s = embryo.interval_seconds or self._base_interval_seconds or 300.0
        anchor = datetime.now() if from_now else (embryo.next_due_at or datetime.now())
        embryo.next_due_at = anchor + timedelta(seconds=interval_s)

    def transition_cadence(
        self,
        embryo,
        *,
        new_phase: str | None = None,
        new_interval_seconds: float | None = None,
        reschedule: bool = True,
        reason: str | None = None,
    ) -> None:
        """Public API for cadence transitions (Phase 5 / agent tools use this).

        Mutates the embryo, schedules its next acquisition, and emits an
        ``EMBRYO_CADENCE_CHANGED`` event so downstream UI / persistence
        stays consistent.
        """
        old_phase = getattr(embryo, "cadence_phase", "normal")
        old_interval = embryo.interval_seconds

        changed = False
        if new_phase is not None and new_phase != old_phase:
            embryo.cadence_phase = new_phase
            changed = True
        if new_interval_seconds is not None and new_interval_seconds != old_interval:
            embryo.interval_seconds = new_interval_seconds
            changed = True

        if not changed:
            return

        if reschedule:
            self._reschedule(embryo)

        self._emit_event(
            EventType.EMBRYO_CADENCE_CHANGED,
            {
                "embryo_id": embryo.id,
                "old_phase": old_phase,
                "new_phase": embryo.cadence_phase,
                "old_interval_s": old_interval,
                "new_interval_s": embryo.interval_seconds,
                "next_due_at": embryo.next_due_at.isoformat() if embryo.next_due_at else None,
                "reason": reason,
            },
        )

    async def _finalize_timelapse(self):
        """Drain perception tasks, log trace count, emit completion event."""
        if self._perception_tasks:
            pending = list(self._perception_tasks)
            logger.info(
                f"Draining {len(pending)} perception task(s) before completing timelapse..."
            )
            _done, still_pending = await asyncio.wait(pending, timeout=60.0)
            if still_pending:
                logger.warning(
                    f"{len(still_pending)} perception task(s) did not finish "
                    f"in 60s - proceeding to completion anyway"
                )
            self._perception_tasks.clear()

        if self._trace_dir:
            try:
                trace_count = len(list(self._trace_dir.glob("*.json")))
                logger.info(f"Trace files saved: {trace_count} files in {self._trace_dir}")
            except Exception:
                pass

        self._finalize_perception_run("completed")

        self._emit_event(
            EventType.ACQUISITION_COMPLETED,
            {
                "total_timepoints": self._total_timepoints,
                "duration_minutes": (
                    (datetime.now() - self._started_at).total_seconds() / 60
                    if self._started_at
                    else 0
                ),
            },
        )

    async def _run_loop(self):
        """Async per-embryo acquisition loop (Phase 4).

        Priority-queue model: each tick, pick the most-overdue eligible
        embryo and acquire it. Each embryo carries its own
        ``next_due_at`` / ``interval_seconds`` / ``cadence_phase`` —
        independent cadences interleave naturally.

        Exclusive burst (Phase 7): while ``_burst_in_progress`` is set,
        the loop yields the wheel to the burst executor and skips normal
        acquisitions until burst completes.
        """
        try:
            while not self._stop_requested:
                # Handle pause (whole-session)
                if self._status == TimelapseStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # Exclusive burst (Phase 7). If something's already in
                # flight, yield and check back. Otherwise, see if the
                # queue has a pending burst to dispatch.
                if self._burst_in_progress is not None:
                    await asyncio.sleep(0.2)
                    continue
                if self._exclusive_queue:
                    next_op = self._exclusive_queue.popleft()
                    self._burst_in_progress = next_op.target_embryo_id
                    # Pause every other embryo so the due-loop skips them
                    # for the duration of the exclusive op.
                    paused_ids = []
                    for eid, e in self._embryo_states.items():
                        if eid == next_op.target_embryo_id:
                            continue
                        if getattr(e, "cadence_phase", "normal") not in (
                            "burst",
                            "paused",
                        ):
                            e.cadence_phase = "paused"
                            paused_ids.append(eid)
                    # Bursting embryo's phase reflects what it's doing.
                    target_emb = self._embryo_states.get(next_op.target_embryo_id)
                    if target_emb is not None:
                        target_emb.cadence_phase = "burst"

                    try:
                        await next_op.run(self)
                        self._burst_applied.add(next_op.target_embryo_id)
                    except Exception as e:
                        logger.error(
                            "Exclusive op %s failed: %s",
                            next_op.request_id,
                            e,
                            exc_info=True,
                        )
                    finally:
                        self._burst_in_progress = None
                        # Restore paused embryos to normal phase. They will
                        # be picked up by the priority queue ASAP (their
                        # next_due_at has been advancing the whole time).
                        # Route through transition_cadence so a CADENCE_CHANGED
                        # event fires for each — the experiment swimlane view
                        # uses these to draw phase boundaries.
                        for eid in paused_ids:
                            emb = self._embryo_states.get(eid)
                            if emb and emb.cadence_phase == "paused":
                                self.transition_cadence(
                                    emb,
                                    new_phase="normal",
                                    reschedule=False,
                                    reason="burst_end:resume_paused",
                                )
                        # Bursting embryo returns to normal cadence per the
                        # experiment's state machine (Phase 5 sets fast→normal).
                        # Use transition_cadence so the burst→normal change
                        # emits an EMBRYO_CADENCE_CHANGED event with the new
                        # interval; rely on it to reschedule the embryo.
                        if target_emb is not None and target_emb.cadence_phase == "burst":
                            from gently.harness.roles import REGISTRY as _ROLE_REGISTRY

                            role_def = _ROLE_REGISTRY.get(target_emb.role)
                            new_interval = (
                                role_def.default_cadence_seconds
                                if role_def is not None
                                else target_emb.interval_seconds
                            )
                            self.transition_cadence(
                                target_emb,
                                new_phase="normal",
                                new_interval_seconds=new_interval,
                                reason="burst_end",
                            )
                    continue

                # All embryos complete?
                active_count = sum(
                    1
                    for e in self._embryo_states.values()
                    if not e.is_complete and not e.should_skip
                )
                if active_count == 0:
                    self._status = TimelapseStatus.COMPLETED
                    await self._finalize_timelapse()
                    logger.info("Timelapse completed - all embryos finished")
                    break

                # Pick the next due embryo (or wait if none ready)
                embryo, wait_s = self._pick_next_due()
                if embryo is None:
                    await asyncio.sleep(max(0.1, wait_s))
                    continue

                # Acquire this embryo. Each acquisition is awaited here —
                # the embryo's next_due_at is advanced AFTER acquisition
                # lands, so we don't pile up overdue items if a single
                # acquisition takes longer than the interval.
                if self._stop_requested:
                    break
                acquire_start = datetime.now()
                await self._acquire_embryo(embryo, round_time=acquire_start)

                # Logical acquisition counter (replaces _current_round for
                # backward-compat with UI consumers that read it).
                self._current_round += 1

                # Reschedule (anchor at "now" so a slow acquisition doesn't
                # cause immediate retrigger).
                self._reschedule(embryo, from_now=True)

                # Persist runtime state so we can resume across an agent
                # restart. Cheap YAML write; OK to do every acquisition.
                try:
                    self.save_state()
                except Exception:
                    logger.debug("save_state failed", exc_info=True)

                # Small yield between back-to-back picks so other tasks
                # (perception, event handlers) get cycles.
                await asyncio.sleep(0.05)

                # Periodic cleanup every ~10 acquisitions.
                if self._store and self._current_round > 0 and self._current_round % 10 == 0:
                    try:
                        self._store.cleanup_incoming()
                    except Exception:
                        logger.debug("cleanup_incoming failed", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Timelapse cancelled")
            self._status = TimelapseStatus.IDLE
        except Exception as e:
            logger.error(f"Timelapse error: {e}\n{traceback.format_exc()}")
            self._status = TimelapseStatus.FAILED
            self._error_message = str(e)
            self._finalize_perception_run("failed", error_message=str(e))
            self._emit_event(
                EventType.ACQUISITION_FAILED,
                {
                    "error": str(e),
                },
            )

    async def _acquire_embryo(self, embryo_state: EmbryoState, round_time: datetime | None = None):
        """Acquire a single volume for one embryo

        Parameters
        ----------
        embryo_state : EmbryoState
            Embryo state to acquire
        round_time : datetime, optional
            Shared timestamp for all embryos in this round (keeps them in sync)
        """
        embryo_id = embryo_state.id
        embryo = self.experiment.embryos.get(embryo_id)

        if not embryo:
            embryo_state.is_complete = True
            embryo_state.completion_reason = "embryo_removed"
            return

        try:
            # Move to embryo position
            pos = embryo.stage_position
            if pos and pos.get("x") is not None:
                await self.client.move_to_position(pos["x"], pos["y"])

            # Get calibration parameters
            cal = embryo.calibration or {}
            galvo_amplitude = cal.get("galvo_amplitude", 0.5)
            galvo_center = cal.get("galvo_center", 0.0)
            piezo_amplitude = cal.get("piezo_amplitude", 25.0)
            piezo_center = cal.get("piezo_center", 50.0)

            # Acquire based on mode (volume or snap)
            acquisition_mode = getattr(embryo, "acquisition_mode", "volume")

            if acquisition_mode == "snap":
                # Single 2D lightsheet image
                result = await self.client.capture_lightsheet_image(
                    piezo_position=piezo_center,
                    galvo_position=galvo_center,
                )
                num_frames = 1
                exposure_ms = 50.0  # Default snap exposure
            else:
                # Full 3D volume (default)
                result = await self.client.acquire_volume(
                    num_slices=embryo.num_slices,
                    exposure_ms=embryo.exposure_ms,
                    galvo_amplitude=galvo_amplitude,
                    galvo_center=galvo_center,
                    piezo_amplitude=piezo_amplitude,
                    piezo_center=piezo_center,
                    laser_power_488_pct=getattr(embryo, "laser_power_488_pct", None),
                )
                num_frames = embryo.num_slices
                exposure_ms = embryo.exposure_ms

            if result.get("success"):
                # Update state
                embryo_state.timepoints_acquired += 1
                embryo_state.error_count = 0
                self._total_timepoints += 1

                # Use round_time for consistent timestamps across all embryos
                acquisition_timestamp = round_time if round_time else datetime.now()

                # Update embryo state
                embryo.timepoints_acquired = embryo_state.timepoints_acquired
                # Record light exposure (includes updating last_imaged)
                embryo.record_exposure(
                    exposure_ms=exposure_ms,
                    num_frames=num_frames,
                    timestamp=acquisition_timestamp,
                )

                # Note: VOLUME_ACQUIRED event is emitted by the callback (agent.on_volume_acquired)
                # to avoid duplicate events and include more metadata

                # Callback for volume/image processing
                volume_data = None
                volume_uids = None  # Track UIDs from storage for perception events
                if self.on_volume_callback:
                    # Get data - 'volume' for volume mode, 'image' for snap mode
                    data = (
                        result.get("volume")
                        if acquisition_mode == "volume"
                        else result.get("image")
                    )
                    if data is not None:
                        # Ensure data is numpy array
                        if not isinstance(data, np.ndarray):
                            data = np.array(data)
                        # For snap mode (2D), add Z dimension so store_volume works
                        if acquisition_mode == "snap" and data.ndim == 2:
                            data = data[np.newaxis, ...]  # Add Z dimension: (Y,X) -> (1,Y,X)
                        volume_data = data
                        # Pass volume_path if available (zero-copy from device)
                        volume_path = result.get("volume_path")
                        # Callback may return UIDs from storage
                        callback_result = await self.on_volume_callback(
                            embryo_id,
                            embryo_state.timepoints_acquired,
                            data,
                            volume_path=volume_path,
                        )
                        if isinstance(callback_result, dict):
                            volume_uids = callback_result

                # Fire perception as a background task instead of awaiting
                # it. This unblocks the next embryo's acquisition so the
                # filmstrip gets fresh volumes every few seconds, not every
                # 20-40 seconds. Perception still runs to completion and
                # still emits DETECTOR_EVALUATED events; it just no longer
                # gates the acquisition loop.
                if self.perceiver and volume_data is not None:
                    perception_task = asyncio.create_task(
                        self._run_perception(
                            embryo_id=embryo_id,
                            timepoint=embryo_state.timepoints_acquired,
                            volume=volume_data,
                            embryo_state=embryo_state,
                            volume_uids=volume_uids,
                        )
                    )
                    self._perception_tasks.add(perception_task)
                    # Remove the task from the set when it finishes so the
                    # set doesn't grow unbounded over a long timelapse.
                    perception_task.add_done_callback(self._perception_tasks.discard)

                # Check stop condition (based on acquisition state, not
                # perception - we'll re-check inside _run_perception too)
                await self._check_stop_condition(embryo_state)

                logger.debug(f"Acquired t={embryo_state.timepoints_acquired} for {embryo_id}")

            else:
                embryo_state.error_count += 1
                embryo_state.last_error = result.get("error", "Unknown error")

                # Log to global error log for cross-embryo correlation
                self.global_error_log.log_error(
                    round_number=self._current_round,
                    embryo_id=embryo_id,
                    timepoint=embryo_state.timepoints_acquired,
                    error_type="acquisition",
                    message=embryo_state.last_error,
                )

                # Stop after too many errors
                if embryo_state.error_count >= 3:
                    embryo_state.is_complete = True
                    embryo_state.completion_reason = f"errors: {embryo_state.last_error}"

        except Exception as e:
            embryo_state.error_count += 1
            embryo_state.last_error = str(e)
            logger.error(
                f"Acquisition failed for {embryo_id} t{embryo_state.timepoints_acquired}: "
                f"{e}\n{traceback.format_exc()}"
            )

            # Log to global error log
            self.global_error_log.log_error(
                round_number=self._current_round,
                embryo_id=embryo_id,
                timepoint=embryo_state.timepoints_acquired,
                error_type="acquisition_exception",
                message=str(e),
                exception=e,
            )

    async def _check_stop_condition(self, embryo_state: EmbryoState):
        """
        Check if ANY stop condition is met (OR logic for composite conditions).

        Supports both single conditions and composite conditions.
        """
        # Role-based no-object terminal: safety net independent of the
        # user-configured stop condition. Fires when an embryo has been
        # detected as "no_object" for N consecutive checks (threshold
        # from the role definition), the canonical signature of an
        # embryo that hatched / drifted out of the FOV.
        from gently.harness.roles import REGISTRY as _ROLE_REGISTRY

        role = _ROLE_REGISTRY.get(getattr(embryo_state, "role", "test"))
        if (
            role is not None
            and role.no_object_consecutive_terminal is not None
            and embryo_state.consecutive_no_object >= role.no_object_consecutive_terminal
        ):
            embryo_state.is_complete = True
            embryo_state.completion_reason = (
                f"no_object x {embryo_state.consecutive_no_object} consecutive "
                f"(role={role.name} threshold={role.no_object_consecutive_terminal}; "
                f"likely hatched / out of FOV)"
            )
            logger.info(f"Embryo {embryo_state.id} stopped: {embryo_state.completion_reason}")
            self._emit_event(
                EventType.EMBRYO_TERMINATED,
                {
                    "embryo_id": embryo_state.id,
                    "completion_reason": embryo_state.completion_reason,
                    "timepoints_acquired": embryo_state.timepoints_acquired,
                },
            )
            return

        # Check all conditions (primary + additional) with OR logic
        for cond in embryo_state.stop_condition.all_conditions():
            reason = self._evaluate_single_condition(cond, embryo_state)
            if reason:
                embryo_state.is_complete = True
                embryo_state.completion_reason = reason
                logger.info(f"Embryo {embryo_state.id} stopped: {reason}")
                self._emit_event(
                    EventType.EMBRYO_TERMINATED,
                    {
                        "embryo_id": embryo_state.id,
                        "completion_reason": reason,
                        "timepoints_acquired": embryo_state.timepoints_acquired,
                    },
                )
                return  # Stop on first matching condition

    def _evaluate_single_condition(
        self, cond: StopCondition, embryo_state: EmbryoState
    ) -> str | None:
        """
        Evaluate a single stop condition.

        Parameters
        ----------
        cond : StopCondition
            Condition to evaluate
        embryo_state : EmbryoState
            Current embryo state

        Returns
        -------
        str or None
            Completion reason if condition is met, None otherwise
        """
        if cond.condition_type == StopConditionType.MANUAL:
            # Never auto-stop
            return None

        elif cond.condition_type == StopConditionType.ALL_TEST_HATCHED:
            # Stop when every role='test' embryo in the active timelapse
            # has hatched (via Claude detector setting hatching_status).
            test_states = [
                e
                for e in self._embryo_states.values()
                if getattr(e, "role", "test") == "test" and not e.should_skip
            ]
            if not test_states:
                return None  # no test embryos → not applicable
            if all(e.hatching_status.get("hatched") for e in test_states):
                # Optional confirmation timepoints — fold into the
                # per-embryo machinery the existing stage-based code uses.
                if embryo_state.detection_triggered_at is None:
                    embryo_state.detection_triggered_at = embryo_state.timepoints_acquired
                    embryo_state.detection_type = "all_test_hatched"
                tps_since = embryo_state.timepoints_acquired - embryo_state.detection_triggered_at
                if tps_since >= cond.confirm_timepoints:
                    return "all test embryos hatched" + (
                        f" (+{cond.confirm_timepoints} confirm)"
                        if cond.confirm_timepoints > 0
                        else ""
                    )
            return None

        elif cond.condition_type == StopConditionType.FIXED_TIMEPOINTS:
            if embryo_state.timepoints_acquired >= cond.value:
                return f"reached {cond.value} timepoints"

        elif cond.condition_type == StopConditionType.DURATION:
            assert self._started_at is not None  # running timelapse
            elapsed_hours = (datetime.now() - self._started_at).total_seconds() / 3600
            if elapsed_hours >= cond.value:
                return f"reached {cond.value}h duration"

        elif cond.condition_type in (
            StopConditionType.STAGE_BASED,
            StopConditionType.HATCHING,
            StopConditionType.COMMA_STAGE,
        ):
            # Generic stage-based stop: check if current stage is in target set
            target = cond.target_stages or set()

            # Check perception system for current stage
            if self.perceiver:
                session = self.perceiver.get_session(embryo_state.id)
                if session:
                    current_stage = session.current_stage
                    if current_stage and current_stage in target:
                        # First time detecting target stage — record the timepoint
                        if embryo_state.detection_triggered_at is None:
                            embryo_state.detection_triggered_at = embryo_state.timepoints_acquired
                            embryo_state.detection_type = f"stage:{current_stage}"
                            logger.info(
                                f"Target stage '{current_stage}' detected for {embryo_state.id} "
                                f"at t{embryo_state.timepoints_acquired}, "
                                f"will acquire {cond.confirm_timepoints} more"
                                " confirmation timepoints"
                            )

                        # Check if we've acquired enough confirmation timepoints
                        timepoints_since_detection = (
                            embryo_state.timepoints_acquired - embryo_state.detection_triggered_at
                        )
                        if timepoints_since_detection >= cond.confirm_timepoints:
                            if cond.confirm_timepoints > 0:
                                return (
                                    f"stage '{current_stage}' detected + "
                                    f"{cond.confirm_timepoints} confirmation timepoints"
                                )
                            return f"target stage reached (current: {current_stage})"

                    # Also catch terminal-stage completion the upper block
                    # might miss: e.g. target={hatching} but perception now
                    # reports 'hatched'. Session has no is_complete() — we
                    # check terminal-stage membership directly.
                    organism = get_organism()
                    if (
                        current_stage
                        and current_stage in organism.TERMINAL_STAGES
                        and target & organism.TERMINAL_STAGES
                    ):
                        return f"terminal stage '{current_stage}' reached (perception)"

            # Fallback: check legacy hatching_status (for manual marking)
            organism = get_organism()
            if target & organism.TERMINAL_STAGES:
                embryo = self.experiment.embryos.get(embryo_state.id)
                if embryo:
                    hatched_via_status = embryo.hatching_status.get("hatched", False)
                    if hatched_via_status:
                        return "terminal stage detected (manual)"

        return None

    def get_status(self) -> TimelapseState:
        """
        Get current timelapse status

        Can be called anytime, even while timelapse is running.

        Returns
        -------
        TimelapseState
            Current state with all embryo details
        """
        # Async cadence: "next round" is meaningless; report the soonest
        # next_due_at across active embryos so UI consumers still get a
        # useful "next acquisition" estimate.
        next_round_time = None
        seconds_until_next = None

        if self._status == TimelapseStatus.RUNNING and self._embryo_states:
            due_times = [
                e.next_due_at
                for e in self._embryo_states.values()
                if not e.is_complete
                and not e.should_skip
                and getattr(e, "cadence_phase", "normal") != "paused"
                and e.next_due_at is not None
            ]
            if due_times:
                next_round_time = min(due_times)
                seconds_until_next = max(0, (next_round_time - datetime.now()).total_seconds())

        return TimelapseState(
            status=self._status,
            started_at=self._started_at,
            embryos=self._embryo_states.copy(),
            total_timepoints=self._total_timepoints,
            current_round=self._current_round,
            interval_seconds=self._base_interval_seconds,
            next_round_time=next_round_time,
            seconds_until_next_round=seconds_until_next,
            error_message=self._error_message,
        )

    async def add_embryo(
        self,
        embryo_id: str,
        stop_condition: str | None = None,
        condition_value: Any = None,
    ) -> str:
        """
        Add an embryo to a running timelapse (round-based architecture)

        Parameters
        ----------
        embryo_id : str
            Embryo to add
        stop_condition : str, optional
            Stop condition (defaults to same as other embryos, or "manual")
        condition_value : any, optional
            Value for stop condition

        Returns
        -------
        str
            Confirmation message
        """
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running. Use start_adaptive_timelapse first."

        if embryo_id in self._embryo_states:
            return f"Embryo '{embryo_id}' already in timelapse"

        # Validate embryo exists
        if embryo_id not in self.experiment.embryos:
            return f"Embryo '{embryo_id}' not found in experiment"

        # Determine stop condition - default to matching other embryos or manual
        if stop_condition:
            stop_cond = self._parse_stop_condition(stop_condition, condition_value)
        else:
            # Use the same stop condition as existing embryos
            existing_states = list(self._embryo_states.values())
            if existing_states:
                stop_cond = existing_states[0].stop_condition
            else:
                stop_cond = StopCondition.manual()

        # Add embryo to the timelapse: set runtime fields on its EmbryoState
        # and register the reference in _embryo_states.
        from gently.harness.roles import REGISTRY as ROLE_REGISTRY

        embryo = self.experiment.embryos[embryo_id]
        embryo.stop_condition = stop_cond
        embryo.is_complete = False
        embryo.completion_reason = None
        embryo.error_count = 0
        embryo.last_error = None
        embryo.detection_triggered_at = None
        embryo.detection_type = None
        embryo.no_object_since_timepoint = None

        # Async cadence init for the newcomer. The newcomer inherits the
        # timelapse's current base_interval (the user-specified cadence
        # at start time); falls back to the role's default if base is
        # somehow unset.
        if self._base_interval_seconds:
            embryo.interval_seconds = self._base_interval_seconds
        elif embryo.interval_seconds is None:
            role_def = ROLE_REGISTRY.get(embryo.role)
            embryo.interval_seconds = (
                role_def.default_cadence_seconds if role_def is not None else 300.0
            )
        embryo.cadence_phase = "normal"
        embryo.next_due_at = datetime.now()  # picked up on next loop tick

        self._embryo_states[embryo_id] = embryo

        logger.info(
            f"Added {embryo_id} to timelapse "
            f"(interval={embryo.interval_seconds}s, role={embryo.role})"
        )

        return (
            f"Added {embryo_id} to timelapse\n"
            f"  Stop condition: {stop_cond.condition_type.value}\n"
            f"  Interval: {embryo.interval_seconds}s (role: {embryo.role})\n"
            f"  Will be picked up on next due-loop tick."
        )

    async def remove_embryo(self, embryo_id: str, mark_complete: bool = True) -> str:
        """
        Remove an embryo from a running timelapse

        Parameters
        ----------
        embryo_id : str
            Embryo to remove
        mark_complete : bool
            If True, mark as complete (preserves history). If False, remove entirely.

        Returns
        -------
        str
            Confirmation message
        """
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running."

        if embryo_id not in self._embryo_states:
            return f"Embryo '{embryo_id}' not in timelapse"

        emb_state = self._embryo_states[embryo_id]
        timepoints = emb_state.timepoints_acquired

        if mark_complete:
            # Mark as complete so it's excluded from future rounds
            emb_state.is_complete = True
            emb_state.completion_reason = "removed by user"
            logger.info(f"Marked {embryo_id} as complete (removed from timelapse)")
            return (
                f"Removed {embryo_id} from timelapse\n"
                f"  Timepoints acquired: {timepoints}\n"
                f"  Status: marked complete"
            )
        else:
            # Remove entirely from tracking
            del self._embryo_states[embryo_id]
            logger.info(f"Removed {embryo_id} from timelapse tracking entirely")
            return (
                f"Removed {embryo_id} from timelapse\n"
                f"  Timepoints acquired: {timepoints}\n"
                f"  Status: removed from tracking"
            )

    def modify_interval(self, new_interval_seconds: float) -> str:
        """
        Broadcast a new acquisition interval to every embryo in this timelapse.

        Async cadence: each embryo carries its own ``interval_seconds`` and
        ``next_due_at``. This convenience method updates every embryo at once
        (including paused / fast / burst phases — they all get the new
        interval, though paused embryos stay paused until separately resumed).
        For per-embryo changes, use ``transition_cadence(embryo, ...)`` or
        the agent tool ``modify_parameters``.

        Parameters
        ----------
        new_interval_seconds : float
            New interval in seconds (must be >= 1)
        """
        if new_interval_seconds < 1:
            return "Error: Interval must be at least 1 second"

        old_base = self._base_interval_seconds
        self._base_interval_seconds = new_interval_seconds
        changed = []
        for eid, embryo in self._embryo_states.items():
            old_per = embryo.interval_seconds
            self.transition_cadence(
                embryo,
                new_interval_seconds=new_interval_seconds,
                reason="modify_interval (broadcast)",
            )
            changed.append((eid, old_per, new_interval_seconds))

        logger.info(
            "Broadcast interval: base %ss -> %ss; %d embryos rescheduled",
            old_base,
            new_interval_seconds,
            len(changed),
        )
        return (
            f"Interval changed to {new_interval_seconds}s across "
            f"{len(changed)} embryo(s); each rescheduled on next loop tick."
        )

    async def modify_embryo(
        self,
        embryo_id: str,
        stop_condition: str | None = None,
        condition_value: Any = None,
    ) -> str:
        """
        Modify parameters for a single embryo during timelapse

        Note: Interval is now global for all embryos. Use modify_interval() to change it.

        Parameters
        ----------
        embryo_id : str
            Embryo to modify
        stop_condition : str, optional
            New stop condition
        condition_value : any, optional
            Value for stop condition

        Returns
        -------
        str
            Confirmation message
        """
        if embryo_id not in self._embryo_states:
            return f"Embryo '{embryo_id}' not in timelapse"

        estate = self._embryo_states[embryo_id]

        changes = []

        if stop_condition is not None:
            new_cond = self._parse_stop_condition(stop_condition, condition_value)
            estate.stop_condition = new_cond
            changes.append(f"stop condition: {stop_condition}")

        if not changes:
            return (
                f"No changes specified for {embryo_id}."
                " Note: use modify_interval() to change acquisition interval."
            )

        return f"Modified {embryo_id}: {', '.join(changes)}"

    async def stop_embryo(self, embryo_id: str, reason: str = "user_request") -> str:
        """
        Stop imaging a specific embryo

        Parameters
        ----------
        embryo_id : str
            Embryo to stop
        reason : str
            Reason for stopping

        Returns
        -------
        str
            Confirmation message
        """
        if embryo_id not in self._embryo_states:
            return f"Embryo '{embryo_id}' not in timelapse"

        estate = self._embryo_states[embryo_id]
        estate.is_complete = True
        estate.completion_reason = reason

        return f"Stopped imaging {embryo_id} (reason: {reason})"

    async def stop(self, reason: str = "user_request") -> str:
        """
        Stop the entire timelapse

        Parameters
        ----------
        reason : str
            Reason for stopping

        Returns
        -------
        str
            Confirmation message
        """
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running"

        self._stop_requested = True

        # Cancel the task
        if self._acquisition_task:
            self._acquisition_task.cancel()
            try:
                await self._acquisition_task
            except asyncio.CancelledError:
                pass

        # Drain in-flight perception tasks. Acquisition is now fire-and-
        # forget, so at stop time there may still be a few VLM calls in
        # the air. Give them a bounded window to finish gracefully so
        # DETECTOR_EVALUATED events and trace files don't get dropped.
        if self._perception_tasks:
            pending = list(self._perception_tasks)
            logger.info(f"Waiting for {len(pending)} in-flight perception task(s) to finish...")
            try:
                done, still_pending = await asyncio.wait(pending, timeout=60.0)
                if still_pending:
                    logger.warning(
                        f"{len(still_pending)} perception task(s) did not finish "
                        f"in 60s - cancelling to unblock stop"
                    )
                    for t in still_pending:
                        t.cancel()
                    # Swallow cancellation errors
                    await asyncio.gather(*still_pending, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error draining perception tasks: {e}")
            self._perception_tasks.clear()

        self._status = TimelapseStatus.IDLE

        # Log trace file count
        if self._trace_dir:
            trace_count = len(list(self._trace_dir.glob("*.json")))
            logger.info(f"Trace files saved: {trace_count} files in {self._trace_dir}")

        self._finalize_perception_run("stopped")

        # Emit stop event for viz server and other listeners
        get_event_bus().publish(
            EventType.ACQUISITION_STOPPED,
            {
                "reason": reason,
                "total_timepoints": self._total_timepoints,
                "embryo_count": len(self._embryo_states),
            },
            source="timelapse_orchestrator",
        )

        return (
            f"Timelapse stopped (reason: {reason})."
            f" Acquired {self._total_timepoints} total timepoints."
        )

    async def pause(self) -> str:
        """Pause the timelapse"""
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running"

        self._pause_start = datetime.now()  # Track when pause started
        self._status = TimelapseStatus.PAUSED

        return "Timelapse paused. Use resume() to continue."

    async def resume(self) -> str:
        """Resume a paused timelapse"""
        if self._status != TimelapseStatus.PAUSED:
            return "Timelapse not paused"

        # Track pause duration to exclude from scheduling
        if self._pause_start:
            self._total_pause_duration += datetime.now() - self._pause_start
            self._pause_start = None

        self._status = TimelapseStatus.RUNNING

        return "Timelapse resumed."

    def add_interval_rule(self, rule: IntervalRule):
        """
        Add an interval adjustment rule

        Parameters
        ----------
        rule : IntervalRule
            Rule to add
        """
        self._interval_rules.append(rule)
        logger.info(f"Added interval rule: {rule.name}")

    def add_speedup_on_stage(
        self,
        stage_name: str,
        new_interval_seconds: float = 30.0,
        embryo_ids: list[str] | None = None,
    ):
        """
        Add a rule to speed up imaging when a stage is reached

        Parameters
        ----------
        stage_name : str
            Stage that triggers speedup (e.g., "3fold", "pretzel", "comma")
        new_interval_seconds : float
            New interval after stage reached
        embryo_ids : list, optional
            Only apply to these embryos (None = all)
        """
        rule = IntervalRule(
            name=f"speedup_on_{stage_name}",
            trigger_stage=stage_name,
            new_interval_seconds=new_interval_seconds,
            applies_to=embryo_ids,
            one_time=True,
        )
        self.add_interval_rule(rule)

    def add_pre_terminal_speedup(self, fast_interval: float = 30.0):
        """
        Add automatic speedup when the organism's pre-terminal stage is detected.

        Uses get_organism().PRE_TERMINAL_SPEEDUP_STAGE to determine the trigger
        stage (e.g., "pretzel" for C. elegans).

        Parameters
        ----------
        fast_interval : float
            Interval to use after pre-terminal stage detection (default 30s)
        """
        organism = get_organism()
        stage = organism.PRE_TERMINAL_SPEEDUP_STAGE
        self.add_speedup_on_stage(stage, fast_interval)
        logger.info(f"Added pre-terminal speedup: {fast_interval}s after {stage} detection")

    # Backward-compatible alias
    def add_pre_hatching_speedup(self, fast_interval: float = 30.0):
        """Alias for add_pre_terminal_speedup (backward compatibility)."""
        self.add_pre_terminal_speedup(fast_interval)

    # ------------------------------------------------------------------
    # Phase 5 convenience helpers — install canonical reactive rules
    # ------------------------------------------------------------------

    def add_power_rule(self, rule: PowerRule):
        """Register a PowerRule for adaptive laser-power control."""
        self._power_rules.append(rule)
        logger.info("Added power rule: %s", rule.name)

    def add_test_onset_speedup(
        self,
        *,
        fast_interval: float = 60.0,
        confirm_timepoints: int = 2,
        embryo_ids: list[str] | None = None,
    ):
        """Install the canonical 'TestEmbryo signal-onset → fast cadence' rule.

        Fires on the ``lit_up`` pseudo-stage emitted by the dopaminergic
        detector (intensity_level ≥ MEDIUM). Per-source-embryo: the cadence
        change applies only to the embryo whose detection triggered the
        match. ``applies_to`` filters which embryos the rule listens to.
        ``confirm_timepoints`` requires that many additional consecutive
        matches before firing, suppressing single-frame false positives.
        """
        if embryo_ids is None:
            embryo_ids = [
                eid
                for eid, e in self._embryo_states.items()
                if getattr(e, "role", "test") == "test"
            ]
        rule = IntervalRule(
            name="test_onset_speedup",
            trigger_stage="lit_up",
            new_interval_seconds=fast_interval,
            applies_to=embryo_ids or None,
            confirm_timepoints=confirm_timepoints,
            one_time=True,
        )
        self.add_interval_rule(rule)
        logger.info(
            "Test-onset speedup installed: %ss on signal onset for %s (confirm_timepoints=%d)",
            fast_interval,
            embryo_ids or "all test embryos",
            confirm_timepoints,
        )

    def add_burst_rule(self, rule):
        """Register a BurstRule for auto-queued burst acquisitions."""
        self._burst_rules.append(rule)
        logger.info("Added burst rule: %s", rule.name)

    def add_test_burst_on_good_structure(
        self,
        *,
        frames: int = 60,
        mode: str = "1hz",
        num_slices: int = 1,
        confirm_timepoints: int = 2,
        embryo_ids: list[str] | None = None,
    ):
        """Install the canonical 'TestEmbryo stably-bright structure → burst' rule.

        Fires when the dopaminergic detector reports structure_quality
        == GOOD (well-resolved neurite pattern) AND intensity_level in
        {MEDIUM, STRONG} (signal stably above background). One-time per
        embryo — enforced downstream by ``queue_burst`` via
        ``_burst_applied``. ``confirm_timepoints`` requires that many
        additional consecutive matches before firing.
        """
        if embryo_ids is None:
            embryo_ids = [
                eid
                for eid, e in self._embryo_states.items()
                if getattr(e, "role", "test") == "test"
            ]
        rule = BurstRule(
            name="test_burst_on_good_structure",
            trigger_detector="dopaminergic_signal",
            trigger_intensity_levels=["MEDIUM", "STRONG"],
            trigger_structure_qualities=["GOOD"],
            frames=frames,
            mode=mode,
            num_slices=num_slices,
            applies_to=embryo_ids or None,
            confirm_timepoints=confirm_timepoints,
        )
        self.add_burst_rule(rule)
        logger.info(
            "Test-burst rule installed: %d frames @ %s on stable structure for %s "
            "(confirm_timepoints=%d)",
            frames,
            mode,
            embryo_ids or "all test embryos",
            confirm_timepoints,
        )

    def add_test_saturation_rampdown(
        self,
        *,
        wavelength: int = 488,
        step_pct: float = 1.0,
        floor_pct: float = 2.0,
        ceiling_pct: float = 6.0,
        confirm_timepoints: int = 0,
        embryo_ids: list[str] | None = None,
    ):
        """Install the canonical 'TestEmbryo saturation → step laser down' rule.

        Sticky-monotonic downward ramp: fires on intensity_level == SATURATING
        from the dopaminergic detector, drops ``wavelength`` power by
        ``step_pct`` (default 1%) until ``floor_pct``. Never increases.
        Re-fires each round signal saturates (one_time=False) so the ramp
        can chase the growing signal.
        """
        if embryo_ids is None:
            embryo_ids = [
                eid
                for eid, e in self._embryo_states.items()
                if getattr(e, "role", "test") == "test"
            ]
        rule = PowerRule(
            name=f"test_saturation_rampdown_{wavelength}",
            wavelength=wavelength,
            trigger_detector="dopaminergic_signal",
            trigger_intensity_levels=["SATURATING"],
            step_pct=step_pct,
            floor_pct=floor_pct,
            ceiling_pct=ceiling_pct,
            direction="down",
            applies_to=embryo_ids or None,
            confirm_timepoints=confirm_timepoints,
            one_time=False,
        )
        self.add_power_rule(rule)
        logger.info(
            "Test-saturation rampdown installed: %dnm step=%.2f%% floor=%.2f%% for %s",
            wavelength,
            step_pct,
            floor_pct,
            embryo_ids or "all test embryos",
        )

    # ------------------------------------------------------------------
    # Phase 9: overnight persistence (timelapse.yaml)
    # ------------------------------------------------------------------

    def _session_storage_dir(self) -> Path | None:
        """Resolve the FileStore-indexed folder for this session.

        Falls back to ``<root>/sessions/<session_id>`` (the legacy bare-id
        layout used before this routing was fixed) so a running session
        whose timelapse.yaml predates the fix can still load. Returns
        None if no session is active.
        """
        if not (self._store and self._session_id):
            return None
        sd = self._store._session_dir(self._session_id)
        if sd is None:
            sd = self._store.root / "sessions" / self._session_id
        return sd

    def save_state(self) -> Path | None:
        """Write orchestrator runtime state to ``timelapse.yaml``.

        Captures per-embryo cadence state, installed rules, burst state,
        photodose budget — enough to rebuild the running orchestrator's
        state on a session resume. Best-effort: failures log a warning
        but don't raise.

        Returns the path written, or None if no session is active.
        """
        sd = self._session_storage_dir()
        if sd is None:
            return None
        try:
            import yaml

            path = sd / "timelapse.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            doc = self._serialize_runtime_state()
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
            logger.debug("Saved timelapse.yaml (%d embryos)", len(self._embryo_states))
            return path
        except Exception as e:
            logger.warning("save_state failed: %s", e)
            return None

    def load_state(self) -> str:
        """Read ``timelapse.yaml`` and restore orchestrator runtime state.

        Call from session-resume code AFTER ``self.experiment.embryos`` has
        been populated (their on-disk embryo.yaml carries the durable
        identity / role / position fields; this restores the transient
        cadence + rule state on top).

        Looks first at the FileStore-indexed folder, then at the legacy
        bare-session-id folder used by older saves.
        """
        if not (self._store and self._session_id):
            return "No session — cannot load state."
        try:
            import yaml

            candidates = []
            sd = self._store._session_dir(self._session_id)
            if sd is not None:
                candidates.append(sd / "timelapse.yaml")
            legacy = self._store.root / "sessions" / self._session_id / "timelapse.yaml"
            if legacy not in candidates:
                candidates.append(legacy)
            path = next((p for p in candidates if p.exists()), None)
            if path is None:
                return f"No timelapse.yaml at {candidates[0]}"
            with open(path, encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
        except Exception as e:
            return f"Failed to read timelapse.yaml: {e}"

        try:
            self._apply_runtime_state(doc)
        except Exception as e:
            return f"Failed to apply state: {e}"

        return (
            f"Restored timelapse state: {len(self._embryo_states)} embryos, "
            f"{len(self._interval_rules)} interval rules, "
            f"{len(self._power_rules)} power rules, "
            f"{len(self._exclusive_queue)} queued bursts"
        )

    def _serialize_runtime_state(self) -> dict[str, Any]:
        """Build the timelapse.yaml document."""

        def _iso(dt):
            return dt.isoformat() if dt is not None else None

        def _ser_stop_condition(sc):
            """Serialize a StopCondition (including composite OR-chains)."""
            if sc is None:
                return None
            try:
                return {
                    "spec": sc.describe(),
                    "condition_type": sc.condition_type.value,
                    "value": sc.value,
                    "target_stages": sorted(sc.target_stages) if sc.target_stages else None,
                    "confirm_timepoints": sc.confirm_timepoints,
                    "additional": [
                        {
                            "condition_type": c.condition_type.value,
                            "value": c.value,
                            "target_stages": sorted(c.target_stages) if c.target_stages else None,
                            "confirm_timepoints": c.confirm_timepoints,
                        }
                        for c in sc.additional_conditions
                    ],
                }
            except Exception:
                return None

        embryos = {}
        for eid, e in self._embryo_states.items():
            embryos[eid] = {
                "role": getattr(e, "role", "test"),
                "cadence_phase": getattr(e, "cadence_phase", "normal"),
                "interval_seconds": e.interval_seconds,
                "next_due_at": _iso(getattr(e, "next_due_at", None)),
                "laser_power_488_pct": e.laser_power_488_pct,
                "total_exposure_ms": e.total_exposure_ms,
                "timepoints_acquired": e.timepoints_acquired,
                "is_complete": e.is_complete,
                "completion_reason": e.completion_reason,
                "should_skip": e.should_skip,
                "skip_reason": e.skip_reason,
                "detection_triggered_at": e.detection_triggered_at,
                "detection_type": e.detection_type,
                "no_object_since_timepoint": e.no_object_since_timepoint,
                "hatching_status": dict(e.hatching_status) if e.hatching_status else {},
                # Stop condition is per-embryo (StopCondition with optional
                # composite OR-chain). Persisted so the strategy view can
                # show bounded vs open-ended and so resumes restore correctly.
                "stop_condition": _ser_stop_condition(getattr(e, "stop_condition", None)),
            }

        def _ser_interval_rule(r):
            return {
                "name": r.name,
                "trigger_detector": r.trigger_detector,
                "trigger_stage": r.trigger_stage,
                "new_interval_seconds": r.new_interval_seconds,
                "applies_to": list(r.applies_to) if r.applies_to else None,
                "confirm_timepoints": r.confirm_timepoints,
                "one_time": r.one_time,
            }

        def _ser_power_rule(r):
            return {
                "name": r.name,
                "wavelength": r.wavelength,
                "trigger_detector": r.trigger_detector,
                "trigger_intensity_levels": list(r.trigger_intensity_levels or []),
                "trigger_stage": r.trigger_stage,
                "step_pct": r.step_pct,
                "floor_pct": r.floor_pct,
                "ceiling_pct": r.ceiling_pct,
                "direction": r.direction,
                "applies_to": list(r.applies_to) if r.applies_to else None,
                "confirm_timepoints": r.confirm_timepoints,
                "one_time": r.one_time,
            }

        def _ser_burst_rule(r):
            return {
                "name": r.name,
                "trigger_detector": r.trigger_detector,
                "trigger_intensity_levels": list(r.trigger_intensity_levels or []) or None,
                "trigger_structure_qualities": list(r.trigger_structure_qualities or []) or None,
                "frames": r.frames,
                "mode": r.mode,
                "num_slices": r.num_slices,
                "applies_to": list(r.applies_to) if r.applies_to else None,
                "confirm_timepoints": r.confirm_timepoints,
            }

        return {
            "saved_at": datetime.now().isoformat(),
            "schema_version": 1,
            "status": self._status.value if hasattr(self._status, "value") else str(self._status),
            "started_at": _iso(self._started_at),
            "base_interval_seconds": self._base_interval_seconds,
            "current_round": self._current_round,
            "total_timepoints": self._total_timepoints,
            "dose_budget_base_ms": self._dose_budget_base_ms,
            "dose_budget_exceeded": sorted(self._dose_budget_exceeded),
            "burst_applied": sorted(self._burst_applied),
            "burst_in_progress": self._burst_in_progress,
            "exclusive_queue": [
                {
                    "kind": op.kind,
                    "request_id": op.request_id,
                    "target_embryo_id": op.target_embryo_id,
                    "frames": getattr(op, "frames", None),
                    "mode": getattr(op, "mode", None),
                    "num_slices": getattr(op, "num_slices", None),
                }
                for op in self._exclusive_queue
            ],
            "interval_rules": [_ser_interval_rule(r) for r in self._interval_rules],
            "power_rules": [_ser_power_rule(r) for r in self._power_rules],
            "burst_rules": [_ser_burst_rule(r) for r in self._burst_rules],
            "applied_rules": {eid: sorted(s) for eid, s in self._applied_rules.items()},
            "active_monitoring_modes": [m.name for m in self._active_monitoring_modes],
            "embryos": embryos,
        }

    def _apply_runtime_state(self, doc: dict[str, Any]) -> None:
        """Restore orchestrator state from a parsed timelapse.yaml dict."""
        from .exclusive import BurstAcquisition
        from .timelapse_models import BurstRule, IntervalRule, PowerRule, StopCondition

        def _parse_dt(s):
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except (TypeError, ValueError):
                return None

        self._base_interval_seconds = float(
            doc.get("base_interval_seconds", self._base_interval_seconds)
        )
        self._current_round = int(doc.get("current_round", self._current_round))
        self._total_timepoints = int(doc.get("total_timepoints", self._total_timepoints))
        self._dose_budget_base_ms = doc.get("dose_budget_base_ms")
        self._dose_budget_exceeded = set(doc.get("dose_budget_exceeded") or [])
        self._burst_applied = set(doc.get("burst_applied") or [])
        self._burst_in_progress = doc.get("burst_in_progress")
        if started := _parse_dt(doc.get("started_at")):
            self._started_at = started

        self._interval_rules = []
        for r in doc.get("interval_rules") or []:
            self._interval_rules.append(
                IntervalRule(
                    name=r["name"],
                    trigger_detector=r.get("trigger_detector"),
                    trigger_stage=r.get("trigger_stage"),
                    new_interval_seconds=float(r.get("new_interval_seconds", 30.0)),
                    applies_to=r.get("applies_to"),
                    confirm_timepoints=int(r.get("confirm_timepoints", 0)),
                    one_time=bool(r.get("one_time", True)),
                )
            )

        self._power_rules = []
        for r in doc.get("power_rules") or []:
            self._power_rules.append(
                PowerRule(
                    name=r["name"],
                    wavelength=int(r.get("wavelength", 488)),
                    trigger_detector=r.get("trigger_detector"),
                    trigger_intensity_levels=r.get("trigger_intensity_levels") or None,
                    trigger_stage=r.get("trigger_stage"),
                    step_pct=float(r.get("step_pct", 1.0)),
                    floor_pct=float(r.get("floor_pct", 2.0)),
                    ceiling_pct=float(r.get("ceiling_pct", 6.0)),
                    direction=r.get("direction", "down"),
                    applies_to=r.get("applies_to"),
                    confirm_timepoints=int(r.get("confirm_timepoints", 0)),
                    one_time=bool(r.get("one_time", False)),
                )
            )

        self._burst_rules = []
        for r in doc.get("burst_rules") or []:
            self._burst_rules.append(
                BurstRule(
                    name=r["name"],
                    trigger_detector=r.get("trigger_detector"),
                    trigger_intensity_levels=r.get("trigger_intensity_levels") or None,
                    trigger_structure_qualities=r.get("trigger_structure_qualities") or None,
                    frames=int(r.get("frames", 60)),
                    mode=r.get("mode", "1hz"),
                    num_slices=int(r.get("num_slices", 1)),
                    applies_to=r.get("applies_to"),
                    confirm_timepoints=int(r.get("confirm_timepoints", 0)),
                )
            )

        self._applied_rules = {
            eid: set(names) for eid, names in (doc.get("applied_rules") or {}).items()
        }

        self._exclusive_queue.clear()
        for op_doc in doc.get("exclusive_queue") or []:
            if op_doc.get("kind") == "burst":
                self._exclusive_queue.append(
                    BurstAcquisition(
                        target_embryo_id=op_doc["target_embryo_id"],
                        frames=int(op_doc.get("frames", 60)),
                        mode=op_doc.get("mode", "1hz"),
                        num_slices=int(op_doc.get("num_slices", 1)),
                        request_id=op_doc.get("request_id"),
                        temperature_provider=self._temperature_provider,
                    )
                )

        def _deser_stop_condition(d):
            """Rebuild a StopCondition from the dict written by _ser_stop_condition."""
            if not d:
                return None
            try:
                primary = StopCondition(
                    condition_type=StopConditionType(d["condition_type"]),
                    value=d.get("value"),
                    target_stages=set(d.get("target_stages") or []) or None,
                    confirm_timepoints=int(d.get("confirm_timepoints") or 0),
                )
                for ad in d.get("additional") or []:
                    primary.add_condition(
                        StopCondition(
                            condition_type=StopConditionType(ad["condition_type"]),
                            value=ad.get("value"),
                            target_stages=set(ad.get("target_stages") or []) or None,
                            confirm_timepoints=int(ad.get("confirm_timepoints") or 0),
                        )
                    )
                return primary
            except Exception:
                # Fall back to spec-string parse if shape changed across versions.
                spec = d.get("spec") if isinstance(d, dict) else None
                if isinstance(spec, str) and spec:
                    try:
                        return StopCondition.parse(spec)
                    except Exception:
                        return None
                return None

        # Per-embryo state: only restore fields for embryos that already
        # exist in the experiment (embryo.yaml is the durable identity).
        for eid, ed in (doc.get("embryos") or {}).items():
            embryo = self.experiment.embryos.get(eid)
            if embryo is None:
                continue
            for attr in (
                "cadence_phase",
                "interval_seconds",
                "laser_power_488_pct",
                "total_exposure_ms",
                "timepoints_acquired",
                "is_complete",
                "completion_reason",
                "should_skip",
                "skip_reason",
                "detection_triggered_at",
                "detection_type",
                "no_object_since_timepoint",
            ):
                if attr in ed:
                    setattr(embryo, attr, ed[attr])
            if "next_due_at" in ed:
                embryo.next_due_at = _parse_dt(ed.get("next_due_at"))
            if "hatching_status" in ed:
                embryo.hatching_status = dict(ed.get("hatching_status") or {})
            if "stop_condition" in ed:
                sc = _deser_stop_condition(ed.get("stop_condition"))
                if sc is not None:
                    embryo.stop_condition = sc
            # Re-register into the orchestrator's active set if not skipped.
            if not embryo.should_skip and not embryo.is_complete:
                self._embryo_states[eid] = embryo

    # ------------------------------------------------------------------
    # Phase 7: exclusive acquisitions (burst, ...)
    # ------------------------------------------------------------------

    def queue_burst(
        self,
        embryo_id: str,
        *,
        frames: int = 60,
        mode: str = "1hz",
        num_slices: int = 1,
        force: bool = False,
        laser_config: str | None = None,
        tactic_id: str | None = None,
    ) -> str:
        """Queue a burst acquisition for ``embryo_id``.

        Bursts are FIFO; only one runs at a time. ``one_time``-per-embryo
        semantics by default (``force=True`` overrides). The orchestrator
        pauses all other embryos while the burst is running, then
        restores them to ``normal`` cadence afterwards.

        Parameters
        ----------
        embryo_id : str
            Target embryo. Must be in the active timelapse.
        frames : int
            Number of frames (default 60 — one 1 Hz movie at 60s).
        mode : "1hz" | "asap"
            Cadence policy.
        num_slices : int
            Z-slices per frame. Default 1 (snap mode — best chance of
            sustaining 1 Hz on this hardware).
        force : bool
            If True, queue even if this embryo has already had a burst.
        """
        from .exclusive import BurstAcquisition

        if embryo_id not in self._embryo_states:
            return f"Embryo '{embryo_id}' not in active timelapse."
        if not force and embryo_id in self._burst_applied:
            return (
                f"Embryo '{embryo_id}' already had a burst this session. "
                f"Pass force=True to queue another."
            )
        if any(op.target_embryo_id == embryo_id for op in self._exclusive_queue):
            return f"Embryo '{embryo_id}' already has a queued burst."
        op = BurstAcquisition(
            target_embryo_id=embryo_id,
            frames=frames,
            mode=mode,
            num_slices=num_slices,
            temperature_provider=self._temperature_provider,
            laser_config=laser_config,
            tactic_id=tactic_id,
        )
        self._exclusive_queue.append(op)
        logger.info(
            "Queued burst for %s: frames=%d mode=%s num_slices=%d request_id=%s (queue depth=%d)",
            embryo_id,
            frames,
            mode,
            num_slices,
            op.request_id,
            len(self._exclusive_queue),
        )
        self._emit_event(
            EventType.BURST_QUEUED,
            {
                "embryo_id": embryo_id,
                "request_id": op.request_id,
                "position_in_queue": len(self._exclusive_queue),
                "frames": frames,
                "mode": mode,
            },
        )
        return (
            f"Burst queued for {embryo_id} (request_id={op.request_id}, "
            f"frames={frames}, mode={mode}, queue_depth={len(self._exclusive_queue)})"
        )

    # ------------------------------------------------------------------
    # Phase 6: calibration pipelines (capture on Calibration, apply on Test)
    # ------------------------------------------------------------------

    def run_calibration_pipelines(
        self,
        *,
        pipelines: list[str] | None = None,
        source_volumes: dict[str, Any] | None = None,
        embryo_bboxes: dict[str, Any] | None = None,
    ) -> str:
        """Run the named calibration pipelines on CalibrationEmbryo volumes
        and merge the result into ``self._calibration_data``.

        ``source_volumes`` shape depends on the pipeline:
        - TwoPointCalibration: ``{"dark": {eid: vol}, "flat": {eid: vol}}``
        - EdgeRoiCalibration: ``{eid: vol}`` (or rely on ``embryo_bboxes``)

        Detectors read the merged calibration dict from their context;
        the dopaminergic detector specifically picks ``dark``, ``flat``,
        ``edge_bbox`` out and applies them as preprocessing.
        """
        from gently.app.calibration import (
            aggregate_calibrations,
            get_calibration_pipeline,
        )

        if pipelines is None:
            pipelines = ["two_point", "edge_roi"]

        captured = []
        for pname in pipelines:
            pipeline = get_calibration_pipeline(pname)
            if pipeline is None:
                logger.warning("Unknown calibration pipeline: %s", pname)
                continue

            # Each pipeline gets the source_volumes shape it expects.
            ctx = {"embryo_bboxes": embryo_bboxes} if embryo_bboxes else {}
            try:
                data = pipeline.capture(source_volumes or {}, ctx)
                captured.append(data)
                logger.info(
                    "Calibration pipeline '%s' captured: keys=%s notes=%s",
                    pname,
                    sorted(data.payload.keys()),
                    data.notes,
                )
            except Exception as e:
                logger.warning("Calibration pipeline '%s' failed: %s", pname, e)

        if not captured:
            return "No calibration pipelines produced data."

        merged = aggregate_calibrations(captured)
        self._calibration_data = merged

        # Persist a tiny manifest. The actual numpy arrays could be saved
        # as .npy alongside if needed; for now just log what we captured.
        sd = self._session_storage_dir()
        if sd is not None:
            try:
                import yaml

                cal_dir = sd / "calibration"
                cal_dir.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "pipelines": [
                        {
                            "name": c.pipeline_name,
                            "captured_at": c.captured_at.isoformat(),
                            "source_embryo_ids": c.source_embryo_ids,
                            "payload_keys": sorted(c.payload.keys()),
                            "notes": c.notes,
                        }
                        for c in captured
                    ],
                }
                with open(cal_dir / "aggregate.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(manifest, f, sort_keys=False)
                # Save heavy arrays as .npy
                import numpy as _np

                for c in captured:
                    for key, value in c.payload.items():
                        if isinstance(value, _np.ndarray):
                            _np.save(cal_dir / f"{c.pipeline_name}_{key}.npy", value)
            except Exception as e:
                logger.warning("Failed to persist calibration manifest: %s", e)

        return (
            f"Calibration complete: {len(captured)} pipeline(s); "
            f"merged keys: {sorted(merged.keys())}"
        )

    def enable_expression_monitoring(
        self,
        *,
        fast_interval: float = 60.0,
        rampdown_step_pct: float = 1.0,
        rampdown_floor_pct: float = 2.0,
        rampdown_ceiling_pct: float = 6.0,
    ):
        """One-call activation of the dopaminergic-onset experiment's
        reactive package: onset speedup + sticky-downward power ramp.

        Equivalent to::

            orchestrator.enable_monitoring_mode("expression_monitoring")

        See ``gently.app.orchestration.monitoring_modes.ExpressionMonitoringMode``
        for the declarative form.
        """
        from .monitoring_modes import ExpressionMonitoringMode

        mode = ExpressionMonitoringMode(
            name="expression_monitoring",
            description="",
            fast_interval=fast_interval,
            rampdown_step_pct=rampdown_step_pct,
            rampdown_floor_pct=rampdown_floor_pct,
            rampdown_ceiling_pct=rampdown_ceiling_pct,
        )
        mode.activate(self)
        self._active_monitoring_modes.append(mode)
        return f"Activated monitoring mode '{mode.name}': {mode.description}"

    def enable_monitoring_mode(
        self,
        name: str,
        *,
        embryo_ids: list[str] | None = None,
        **mode_kwargs,
    ) -> str:
        """Activate a named MonitoringMode from the registry.

        ``name`` is a key in
        ``gently.app.orchestration.monitoring_modes.MONITORING_MODES``
        (e.g. ``"expression_monitoring"``, ``"pre_terminal_monitoring"``,
        ``"idle"``). Extra kwargs are forwarded to the mode's
        constructor (e.g. ``fast_interval=30.0``).
        """
        from .monitoring_modes import MONITORING_MODES

        factory = MONITORING_MODES.get(name)
        if factory is None:
            return (
                f"Unknown monitoring mode: {name!r}. Available: {sorted(MONITORING_MODES.keys())}"
            )
        mode = factory(**mode_kwargs) if mode_kwargs else factory()
        mode.activate(self, embryo_ids=embryo_ids)
        self._active_monitoring_modes.append(mode)
        return f"Activated monitoring mode '{mode.name}': {mode.description}"

    def _check_interval_rules(
        self,
        embryo_id: str,
        detector_name: str | None = None,
        stage: str | None = None,
        intensity_level: str | None = None,
        structure_quality: str | None = None,
    ):
        """
        Evaluate all adaptive rules (interval + power) against a fresh
        detection event and apply matching ones.

        Despite the legacy name, this method now drives **both**
        ``IntervalRule`` and ``PowerRule`` evaluation — Phase 5 reactive
        control. ``intensity_level`` and ``structure_quality`` are passed
        from ``_run_detector`` (Phase 2) so power rules can fire on
        ``SATURATING`` etc.; ``stage`` keeps the existing perception-driven
        cadence triggers working.
        """
        if embryo_id not in self._embryo_states:
            return

        estate = self._embryo_states[embryo_id]

        # Get already applied rules for this embryo
        if embryo_id not in self._applied_rules:
            self._applied_rules[embryo_id] = set()
        applied = self._applied_rules[embryo_id]

        # ---- Power rules (Phase 5 sticky-downward ramp etc.) ----
        if embryo_id not in self._power_rule_consecutive:
            self._power_rule_consecutive[embryo_id] = {}
        consec = self._power_rule_consecutive[embryo_id]

        for prule in self._power_rules:
            matches = prule.matches(
                embryo_id=embryo_id,
                detector_name=detector_name,
                stage=stage,
                intensity_level=intensity_level,
            )

            # Track consecutive matches for confirm_timepoints.
            if matches:
                consec[prule.name] = consec.get(prule.name, 0) + 1
            else:
                consec[prule.name] = 0
                continue

            if consec[prule.name] < max(1, prule.confirm_timepoints + 1):
                continue  # need more consecutive matches before applying

            if prule.one_time and prule.name in applied:
                continue

            current = estate.laser_power_488_pct if prule.wavelength == 488 else None
            if current is None:
                # Embryo doesn't have a per-embryo override yet — fall back
                # to the experiment-wide default (or 4.0 if nothing set).
                current = 4.0
            new_pct = prule.next_power(current)
            if abs(new_pct - current) < 1e-6:
                # At floor/ceiling — nothing to do.
                if prule.one_time:
                    applied.add(prule.name)
                continue

            if prule.wavelength == 488:
                estate.laser_power_488_pct = new_pct
            # (Future wavelengths: extend EmbryoState similarly.)
            self._emit_event(
                EventType.POWER_RAMP_STEP,
                {
                    "embryo_id": embryo_id,
                    "rule": prule.name,
                    "wavelength": prule.wavelength,
                    "old_pct": current,
                    "new_pct": new_pct,
                    "direction": prule.direction,
                    "intensity_level": intensity_level,
                },
            )
            # Discrete trigger-fired event so the strategy view can show
            # rule firings without inferring them from the power-step event.
            self._emit_event(
                EventType.TRIGGER_FIRED,
                {
                    "embryo_id": embryo_id,
                    "rule_name": prule.name,
                    "rule_kind": "power",
                    "trigger_detector": prule.trigger_detector,
                    "trigger_stage": prule.trigger_stage,
                    "trigger_intensity_level": intensity_level,
                    "applied": {
                        "wavelength": prule.wavelength,
                        "old_pct": current,
                        "new_pct": new_pct,
                        "direction": prule.direction,
                    },
                },
            )
            logger.info(
                "Applied PowerRule '%s' on %s: %dnm %.2f%% -> %.2f%% (direction=%s, intensity=%s)",
                prule.name,
                embryo_id,
                prule.wavelength,
                current,
                new_pct,
                prule.direction,
                intensity_level,
            )

            if prule.one_time:
                applied.add(prule.name)

        # ---- Interval rules: per-source-embryo scope ----
        # A match for embryo X applies the cadence change to embryo X only.
        # ``applies_to`` is the listen-filter (which embryos the rule watches);
        # it is NOT a fan-out target list.
        if embryo_id not in self._interval_rule_consecutive:
            self._interval_rule_consecutive[embryo_id] = {}
        iconsec = self._interval_rule_consecutive[embryo_id]

        for rule in self._interval_rules:
            # Skip if already applied for this embryo (one-time rules)
            if rule.one_time and rule.name in applied:
                iconsec[rule.name] = 0
                continue

            matches = rule.matches(
                embryo_id=embryo_id,
                detector_name=detector_name,
                stage=stage,
            )

            if matches:
                iconsec[rule.name] = iconsec.get(rule.name, 0) + 1
            else:
                iconsec[rule.name] = 0
                continue

            # Require N+1 consecutive matches before firing (N = confirm_timepoints).
            if iconsec[rule.name] < max(1, rule.confirm_timepoints + 1):
                continue

            target = self._embryo_states.get(embryo_id)
            if target is None or target.is_complete:
                continue

            old_interval = target.interval_seconds
            self.transition_cadence(
                target,
                new_interval_seconds=rule.new_interval_seconds,
                reason=f"rule:{rule.name}",
            )
            self._emit_event(
                EventType.TRIGGER_FIRED,
                {
                    "embryo_id": embryo_id,
                    "rule_name": rule.name,
                    "rule_kind": "interval",
                    "trigger_detector": rule.trigger_detector,
                    "trigger_stage": rule.trigger_stage,
                    "trigger_intensity_level": None,
                    "applied": {
                        "old_interval_s": old_interval,
                        "new_interval_s": rule.new_interval_seconds,
                        "one_time": rule.one_time,
                        "confirm_timepoints": rule.confirm_timepoints,
                    },
                },
            )
            logger.info(
                "Applied interval rule '%s' on %s: %ss -> %ss (confirm=%d)",
                rule.name,
                embryo_id,
                old_interval,
                rule.new_interval_seconds,
                rule.confirm_timepoints,
            )

            if rule.one_time:
                applied.add(rule.name)

        # ---- Burst rules: queue a one-shot burst on stable structure ----
        # queue_burst enforces one-time-per-embryo via _burst_applied, so
        # we don't need our own one_time flag.
        if embryo_id not in self._burst_rule_consecutive:
            self._burst_rule_consecutive[embryo_id] = {}
        bconsec = self._burst_rule_consecutive[embryo_id]

        for brule in self._burst_rules:
            if embryo_id in self._burst_applied:
                bconsec[brule.name] = 0
                continue

            matches = brule.matches(
                embryo_id=embryo_id,
                detector_name=detector_name,
                intensity_level=intensity_level,
                structure_quality=structure_quality,
            )

            if matches:
                bconsec[brule.name] = bconsec.get(brule.name, 0) + 1
            else:
                bconsec[brule.name] = 0
                continue

            if bconsec[brule.name] < max(1, brule.confirm_timepoints + 1):
                continue

            result = self.queue_burst(
                embryo_id,
                frames=brule.frames,
                mode=brule.mode,
                num_slices=brule.num_slices,
            )
            self._emit_event(
                EventType.TRIGGER_FIRED,
                {
                    "embryo_id": embryo_id,
                    "rule_name": brule.name,
                    "rule_kind": "burst",
                    "trigger_detector": brule.trigger_detector,
                    "trigger_intensity_level": intensity_level,
                    "trigger_structure_quality": structure_quality,
                    "applied": {
                        "frames": brule.frames,
                        "mode": brule.mode,
                        "num_slices": brule.num_slices,
                        "confirm_timepoints": brule.confirm_timepoints,
                        "queue_result": result,
                    },
                },
            )
            logger.info(
                "Applied burst rule '%s' on %s (intensity=%s structure=%s): %s",
                brule.name,
                embryo_id,
                intensity_level,
                structure_quality,
                result,
            )

    def _finalize_perception_run(self, status: str = "completed", error_message: str | None = None):
        """Mark the perception run as finished in FileStore."""
        if self._store and self._perception_run_id is not None:
            try:
                self._store.complete_perception_run(
                    self._perception_run_id,
                    status=status,
                    error_message=error_message,
                )
                logger.info(f"Perception run {self._perception_run_id} marked as {status}")
            except Exception as e:
                logger.warning(f"Failed to finalize perception run: {e}")

    def _emit_event(self, event_type: EventType, data: dict):
        """Emit event to event bus"""
        self._event_bus.publish(
            event_type=event_type,
            data=data,
            source="timelapse_orchestrator",
        )

    # How often to recheck embryos marked as no_object (in timepoints)
    NO_OBJECT_RECHECK_INTERVAL = 10

    def _serialize_result(self, embryo_id: str, timepoint: int, result, session=None) -> dict:
        """Serialize a perception result to a dict (shared by trace, store, and events)."""
        data = {
            "session_id": self._session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "timestamp": datetime.now().isoformat(),
            "predicted_stage": result.stage,
            "reasoning": result.reasoning,
            "raw_response": result.raw_response,
        }
        if session:
            data["stability"] = session.stability
        return data

    def _volume_to_b64(self, volume) -> tuple:
        """Extract View A from volume and return (view_a, image_b64) or (None, None)."""
        view_a = volume[0] if volume.ndim == 4 else volume

        if view_a.ndim == 3:
            z_depth, height, width = view_a.shape
            if width > height * 2:
                view_a = view_a[:, :, : width // 2]
            bounds = compute_crop_bounds(view_a)
            cropped = apply_crop_bounds(view_a, bounds)
            three_view_img, _ = projection_three_view(cropped)
            return view_a, image_to_base64(three_view_img)
        elif view_a.ndim == 2:
            img = normalize_to_uint8(view_a)
            return view_a, image_to_base64(img)
        else:
            logger.warning(f"Unexpected volume dimensions: {volume.ndim}")
            return None, None

    async def _run_detector(
        self,
        embryo_id: str,
        timepoint: int,
        volume,
        embryo_state: EmbryoState,
        detector_name: str,
        volume_uids: dict | None = None,
    ):
        """Run a role-declared Detector (Phase 2) and persist + emit results.

        This is the path for ``role=test`` (DopaminergicSignalDetector) and
        any other role that declares a non-"perception" detector. The
        existing PerceptionProxy could route through here too, but for
        backward compatibility we keep the original Perceiver code path
        live for ``role=calibration``.
        """
        from gently.app.detectors import get_detector

        detector = get_detector(
            detector_name,
            claude_client=self.claude_client,
            perceiver=self.perceiver,
        )
        if detector is None:
            logger.warning(
                "Unknown detector '%s' for embryo %s — skipping detection",
                detector_name,
                embryo_id,
            )
            return

        # Build context. Calibration params (Phase 6) get plumbed here once
        # captured at session start — for now pass None.
        context = {
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "claude": self.claude_client,
            "perceiver": self.perceiver,
            "calibration": getattr(self, "_calibration_data", None),
        }

        try:
            result = await detector.run(volume, context)
        except Exception as e:
            logger.error("Detector %s failed for %s: %s", detector_name, embryo_id, e)
            return

        # Capture commonly-needed findings up top so we don't repeat lookups.
        findings = result.findings or {}
        intensity_level = findings.get("intensity_level")
        structure_quality = findings.get("structure_quality")
        has_hatched = bool(findings.get("has_hatched"))

        # Mirror onto EmbryoState for the agent's prompt + rule machinery.
        try:
            embryo_state.cv_analyses.setdefault(detector_name, []).append(
                {
                    "timepoint": timepoint,
                    "intensity_level": intensity_level,
                    "structure_quality": structure_quality,
                    "has_hatched": has_hatched,
                    "reasoning": result.reasoning,
                }
            )
            # Keep a rolling cap so cv_analyses doesn't grow unbounded.
            if len(embryo_state.cv_analyses[detector_name]) > 200:
                embryo_state.cv_analyses[detector_name] = embryo_state.cv_analyses[detector_name][
                    -200:
                ]
            # If detector flagged hatched, update legacy field too.
            if has_hatched:
                embryo_state.hatching_status = {
                    "hatched": True,
                    "confidence": "claude",
                    "timepoint": timepoint,
                    "source": detector_name,
                }
        except Exception:
            pass

        # Persist trace JSON
        trace_data = {
            "session_id": self._session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "timestamp": result.timestamp.isoformat(),
            "detector": detector_name,
            "findings": findings,
            "reasoning": result.reasoning,
            "raw_response": result.raw_response,
            "elapsed_ms": result.elapsed_ms,
            "error": result.error,
        }
        if self._trace_dir:
            try:
                self._write_trace_file(embryo_id, timepoint, trace_data)
            except Exception as e:
                logger.warning(f"Failed to write trace file: {e}")

        # Persist into predictions.jsonl. predicted_stage is required by
        # the FileStore schema; for the dopaminergic detector we synthesize
        # a pseudo-stage so downstream consumers see something meaningful.
        # Only MEDIUM/STRONG → "lit_up" (the speedup trigger). WEAK is
        # informational only ("weak_signal"). UNCERTAIN means the
        # classifier refused to commit — emit "uncertain" so no rule fires.
        if intensity_level == "SATURATING":
            pseudo_stage = "lit_up_saturating"
        elif intensity_level in ("STRONG", "MEDIUM"):
            pseudo_stage = "lit_up"
        elif intensity_level == "WEAK":
            pseudo_stage = "weak_signal"
        elif intensity_level == "UNCERTAIN":
            pseudo_stage = "uncertain"
        elif has_hatched:
            pseudo_stage = "hatched"
        else:
            pseudo_stage = (
                "no_object" if intensity_level == "NONE" else (intensity_level or "unknown")
            )

        # Track consecutive no_object across roles — drives the role-based
        # terminal stop in _check_stop_condition.
        if pseudo_stage == "no_object":
            embryo_state.consecutive_no_object += 1
        else:
            embryo_state.consecutive_no_object = 0

        if self._store and self._perception_run_id and self._session_id:
            try:
                self._store.store_prediction(
                    run_id=self._perception_run_id,
                    session_id=self._session_id,
                    embryo_id=embryo_id,
                    timepoint=timepoint,
                    predicted_stage=pseudo_stage,
                    reasoning=result.reasoning,
                    trace_data=trace_data,
                )
            except Exception as e:
                logger.warning(f"Failed to store prediction: {e}")

        # Emit the detector-evaluated event so UI / rule machinery can
        # react. We surface findings alongside a "stage" so existing
        # listeners (e.g. _check_interval_rules using trigger_stage) keep
        # working without modification.
        #
        # For the two-stage dopaminergic detector, ``result.raw_response``
        # is a dict containing the perceiver's prose. Surface that prose
        # as ``description`` on the event so the reasoning panel can show
        # it alongside the classifier's one-line reasoning. Older
        # single-call detectors emit ``raw_response`` as a string —
        # ignore it then.
        description = None
        if isinstance(result.raw_response, dict):
            description = result.raw_response.get("description")
        event_data = {
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "detector_name": detector_name,
            "stage": pseudo_stage,
            "findings": findings,
            "reasoning": result.reasoning,
            "description": description,
            "intensity_level": intensity_level,
            "structure_quality": structure_quality,
            "has_hatched": has_hatched,
            "error": result.error,
        }
        if volume_uids:
            event_data["volume_uid"] = volume_uids.get("volume_uid")
            event_data["projection_uid"] = volume_uids.get("projection_uid")
        self._emit_event(EventType.DETECTOR_EVALUATED, event_data)
        # Dedicated Phase 10 event for the per-detector findings stream.
        self._emit_event(
            EventType.CLAUDE_DETECTOR_RESULT,
            {
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "detector_name": detector_name,
                "findings": findings,
                "reasoning": result.reasoning,
                "description": description,
            },
        )

        if has_hatched:
            self._emit_event(
                EventType.HATCHING_DETECTED,
                {
                    "embryo_id": embryo_id,
                    "timepoint": timepoint,
                    "detector_name": detector_name,
                    "stage": "hatched",
                },
            )

        # Drive Phase 5 reactive rules — pass both the pseudo-stage and
        # the detailed findings so rules can match either.
        self._check_interval_rules(
            embryo_id=embryo_id,
            detector_name=detector_name,
            stage=pseudo_stage,
            intensity_level=intensity_level,
            structure_quality=structure_quality,
        )

        logger.info(
            "[%s] T%d: detector=%s intensity=%s structure=%s hatched=%s",
            embryo_id,
            timepoint,
            detector_name,
            intensity_level,
            structure_quality,
            has_hatched,
        )

    async def _run_perception(
        self,
        embryo_id: str,
        timepoint: int,
        volume,
        embryo_state: EmbryoState,
        volume_uids: dict | None = None,
    ):
        """Run the per-role detector on the acquired volume and emit results.

        Routes by ``embryo.role`` via the detector registry:
        - calibration → PerceptionProxy (existing nuclear-marker classifier)
        - test       → DopaminergicSignalDetector (Claude vision)
        - unassigned → treated like test (safer default)

        Roles can declare ``detector_name=None`` to opt out of any detector.
        """
        # Role-routed detection — Phase 2 wiring. If the role declares a
        # detector other than the standard "perception", branch into the
        # ad-hoc detector path; otherwise fall through to the original
        # Perceiver-based flow below.
        from gently.harness.roles import REGISTRY as ROLE_REGISTRY

        role_def = ROLE_REGISTRY.get(getattr(embryo_state, "role", "test"))
        detector_name = role_def.detector_name if role_def else None

        if detector_name and detector_name != "perception":
            await self._run_detector(
                embryo_id=embryo_id,
                timepoint=timepoint,
                volume=volume,
                embryo_state=embryo_state,
                detector_name=detector_name,
                volume_uids=volume_uids,
            )
            return

        # Skip perception for no_object embryos (except periodic rechecks)
        if embryo_state.no_object_since_timepoint is not None:
            timepoints_since = timepoint - embryo_state.no_object_since_timepoint
            if timepoints_since % self.NO_OBJECT_RECHECK_INTERVAL != 0:
                next_recheck = embryo_state.no_object_since_timepoint + (
                    (timepoints_since // self.NO_OBJECT_RECHECK_INTERVAL + 1)
                    * self.NO_OBJECT_RECHECK_INTERVAL
                )
                self._emit_event(
                    EventType.DETECTOR_EVALUATED,
                    {
                        "embryo_id": embryo_id,
                        "timepoint": timepoint,
                        "detector_name": "perception",
                        "stage": "no_object",
                        "reasoning": (
                            f"Skipped (empty field). Rechecking in"
                            f" {next_recheck - timepoint} timepoints."
                        ),
                        "skipped": True,
                    },
                )
                return
            else:
                logger.info(f"Rechecking no_object embryo {embryo_id} at t={timepoint}")

        try:
            # Volume → projection → base64
            view_a, image_b64 = self._volume_to_b64(volume)
            if view_a is None:
                return

            # Run perception via gently_perception.Perceiver
            result = await self.perceiver(
                embryo_id=embryo_id,
                timepoint=timepoint,
                image_b64=image_b64,
                timestamp=datetime.now(),
            )

            # Track no_object state
            if result.stage == "no_object":
                embryo_state.consecutive_no_object += 1
                if embryo_state.no_object_since_timepoint is None:
                    embryo_state.no_object_since_timepoint = timepoint
                    logger.info(f"Embryo {embryo_id} marked as no_object at t={timepoint}")
            else:
                if embryo_state.no_object_since_timepoint is not None:
                    logger.info(
                        f"Embryo {embryo_id} object found at t={timepoint}, resuming perception"
                    )
                    embryo_state.no_object_since_timepoint = None
                embryo_state.consecutive_no_object = 0

            # Get session for temporal context
            session = self.perceiver.get_session(embryo_id)

            # Serialize result once, reuse for trace file, store, and events
            result_data = self._serialize_result(embryo_id, timepoint, result, session)

            # Persist trace to JSON file
            if self._trace_dir:
                try:
                    self._write_trace_file(embryo_id, timepoint, result_data)
                except Exception as e:
                    logger.warning(f"Failed to write trace file: {e}")

            # Persist prediction to FileStore
            if self._store and self._perception_run_id and self._session_id:
                try:
                    self._store.store_prediction(
                        run_id=self._perception_run_id,
                        session_id=self._session_id,
                        embryo_id=embryo_id,
                        timepoint=timepoint,
                        predicted_stage=result.stage,
                        reasoning=result.reasoning,
                        trace_data=result_data,
                    )
                except Exception as e:
                    logger.warning(f"Failed to store prediction: {e}")

            # Build and emit perception event
            event_data = {
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "detector_name": "perception",
                "stage": result.stage,
                "reasoning": result.reasoning,
            }
            if volume_uids:
                event_data["volume_uid"] = volume_uids.get("volume_uid")
                event_data["projection_uid"] = volume_uids.get("projection_uid")
            if session:
                event_data["stability"] = session.stability
                summary = session.summary()
                if summary.get("temporal"):
                    from dataclasses import asdict

                    event_data["temporal_analysis"] = asdict(summary["temporal"])

            self._emit_event(EventType.DETECTOR_EVALUATED, event_data)

            if result.stage in ("hatching", "hatched"):
                self._emit_event(
                    EventType.HATCHING_DETECTED,
                    {
                        "embryo_id": embryo_id,
                        "timepoint": timepoint,
                        "detector_name": "hatching",
                        "stage": result.stage,
                    },
                )

            # Check interval rules based on stage
            self._check_interval_rules(embryo_id=embryo_id, stage=result.stage)

            logger.info(
                f"[{embryo_id}] T{timepoint}: stage={result.stage}, "
                f"stability={session.stability if session else '?'}"
            )

        except Exception as e:
            logger.error(f"Perception failed for {embryo_id}: {e}")

    def _write_trace_file(self, embryo_id: str, timepoint: int, trace_data: dict) -> Path:
        """Write perception trace to JSON file."""
        filename = f"{embryo_id}_T{timepoint:04d}.json"
        assert self._trace_dir is not None
        file_path = self._trace_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Wrote trace: {file_path.name}")
        return file_path
