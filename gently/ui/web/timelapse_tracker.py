"""
Timelapse State Tracker for the Visualization Server
=====================================================

Tracks timelapse state from events for client synchronization.
"""

from datetime import datetime


class TimelapseStateTracker:
    """
    Tracks timelapse state from events for client synchronization.

    Maintains state from EventBus events so new WebSocket clients
    can receive current timelapse status on connect.

    Uses session_id to help clients identify session boundaries and
    clear stale data when a new experiment starts.
    """

    def __init__(self):
        self.session_id: str | None = None  # Unique ID per experiment
        self.status = "IDLE"  # IDLE, RUNNING, PAUSED, COMPLETED
        self.started_at: str | None = None
        self.embryos: dict[str, dict] = {}  # embryo_id -> state
        self.total_timepoints = 0
        self.base_interval = 120
        self.detection_reasoning: dict[str, list[dict]] = {}  # embryo_id -> list of detections
        self.projection_uids: dict[
            str, dict[int, str]
        ] = {}  # embryo_id -> {timepoint -> projection_uid}
        self.volume_paths: dict[str, dict[int, str]] = {}  # embryo_id -> {timepoint -> volume_path}

    def handle_event(self, event_type: str, data: dict):
        """Update state based on incoming event"""
        if event_type == "SESSION_STARTED":
            # New session - clear all state from previous session
            self.session_id = data.get("session_id")
            self.status = "IDLE"
            self.started_at = None
            self.embryos = {}
            self.detection_reasoning = {}
            self.projection_uids = {}
            self.volume_paths = {}
            self.total_timepoints = 0

        elif event_type == "SESSION_RESTORED":
            # Capture session ID when agent resumes a session
            self.session_id = data.get("session_id")

        elif event_type == "ACQUISITION_STARTED":
            # Use session_id from prior SESSION_STARTED/SESSION_RESTORED event
            # (session_id should already be set before acquisition starts)
            self.status = "RUNNING"
            self.started_at = datetime.now().isoformat()
            self.base_interval = data.get("interval_seconds", 120)
            self.embryos = {}
            self.detection_reasoning = {}
            self.projection_uids = {}
            self.volume_paths = {}
            self.total_timepoints = 0
            for eid in data.get("embryo_ids", []):
                self.embryos[eid] = {
                    "embryo_id": eid,
                    "stop_condition": data.get("stop_condition", "manual"),
                    "interval_seconds": self.base_interval,
                    "timepoints": 0,
                    "is_complete": False,
                    "first_acquired": None,
                    "last_acquired": None,
                    "detections": {},
                    "current_stage": None,  # Updated by perception system
                }
                self.detection_reasoning[eid] = []

        elif event_type == "VOLUME_ACQUIRED":
            eid = data.get("embryo_id")
            if eid:
                # Create embryo if not exists (late join)
                if eid not in self.embryos:
                    self.embryos[eid] = {
                        "embryo_id": eid,
                        "stop_condition": "unknown",
                        "interval_seconds": self.base_interval,
                        "timepoints": 0,
                        "is_complete": False,
                        "first_acquired": None,
                        "last_acquired": None,
                        "detections": {},
                        "current_stage": None,  # Updated by perception system
                    }
                    self.detection_reasoning[eid] = []
                    if self.status == "IDLE":
                        self.status = "RUNNING"
                        self.started_at = datetime.now().isoformat()

                now = datetime.now().isoformat()
                # timepoint is already the count (timepoints_acquired), not 0-indexed
                timepoint = data.get("timepoint", 1)
                self.embryos[eid]["timepoints"] = timepoint
                if self.embryos[eid]["first_acquired"] is None:
                    self.embryos[eid]["first_acquired"] = now
                self.embryos[eid]["last_acquired"] = now
                self.total_timepoints += 1

                # Track projection UID for image lookup
                projection_uid = data.get("projection_uid")
                if projection_uid:
                    if eid not in self.projection_uids:
                        self.projection_uids[eid] = {}
                    self.projection_uids[eid][timepoint] = projection_uid

                # Track volume path for direct file access (projection generation)
                volume_path = data.get("volume_path")
                if volume_path:
                    if eid not in self.volume_paths:
                        self.volume_paths[eid] = {}
                    self.volume_paths[eid][timepoint] = volume_path

        elif event_type == "ACQUISITION_COMPLETED":
            self.status = "COMPLETED"
            completed_at = datetime.now().isoformat()
            for embryo in self.embryos.values():
                embryo["is_complete"] = True
                embryo.setdefault("completed_at", completed_at)

        elif event_type == "ACQUISITION_STOPPED":
            self.status = "STOPPED"
            # Don't mark embryos as complete - they were stopped, not finished

        elif event_type == "EMBRYO_TERMINATED":
            # A single embryo's imaging was halted by the orchestrator
            # (no_object terminal, configured stop condition, errors, etc).
            # Carry the completion_reason through so the UI can show why.
            eid = data.get("embryo_id")
            if eid and eid in self.embryos:
                self.embryos[eid]["is_complete"] = True
                self.embryos[eid]["completion_reason"] = data.get("completion_reason")
                self.embryos[eid].setdefault("completed_at", datetime.now().isoformat())

        elif event_type == "DETECTOR_EVALUATED":
            # All detector/perception evaluations (with reasoning) - populates reasoning panel
            eid = data.get("embryo_id")
            if eid:
                timepoint = data.get("timepoint")
                # Look up projection UID for this timepoint
                projection_uid = None
                if eid in self.projection_uids and timepoint in self.projection_uids.get(eid, {}):
                    projection_uid = self.projection_uids[eid][timepoint]

                detection = {
                    "detector_name": data.get("detector_name", "unknown"),
                    "detected": data.get("detected", data.get("is_hatching", False)),
                    "confidence": data.get("confidence"),
                    "reasoning": data.get("reasoning"),
                    # Perceiver prose from the two-stage dopaminergic detector
                    # (None for legacy single-call detectors / perception).
                    "description": data.get("description"),
                    "timepoint": timepoint,
                    "volume_uid": data.get("volume_uid"),
                    "projection_uid": data.get("projection_uid")
                    or projection_uid,  # Use stored UID as fallback
                    "timestamp": datetime.now().isoformat(),
                    # Perception-specific fields
                    "stage": data.get("stage"),
                    "is_hatching": data.get("is_hatching", False),
                    # Full reasoning trace from VLM (for detail panel)
                    "reasoning_trace": data.get("reasoning_trace"),
                    "is_transitional": data.get("is_transitional"),
                    "transition_between": data.get("transition_between"),
                    "observed_features": data.get("observed_features"),
                    "shape": data.get("shape"),
                    # Two-stage dopaminergic classifier fields. These were
                    # being dropped on persistence, so after a reload the
                    # JSON view in the reasoning panel showed only
                    # `reasoning` even though they were present on the
                    # original live event.
                    "intensity_level": data.get("intensity_level"),
                    "structure_quality": data.get("structure_quality"),
                    "has_hatched": data.get("has_hatched"),
                    "findings": data.get("findings"),
                    "contrastive_reasoning": data.get("contrastive_reasoning"),
                    "temporal_analysis": data.get("temporal_analysis"),
                }
                if eid not in self.detection_reasoning:
                    self.detection_reasoning[eid] = []
                self.detection_reasoning[eid].append(detection)

                # Update embryo's current stage if perception result
                if data.get("stage") and eid in self.embryos:
                    self.embryos[eid]["current_stage"] = data.get("stage")

        elif event_type == "EMBRYO_DETECTED":
            # An embryo was marked / registered (typically from the map
            # view's marking flow). Plant it in our state with its
            # position + role so the device-map renderer can show it
            # before any acquisition has happened.
            eid = data.get("embryo_id")
            if eid:
                emb = self.embryos.setdefault(
                    eid,
                    {
                        "embryo_id": eid,
                        "timepoints": 0,
                        "is_complete": False,
                        "first_acquired": None,
                        "last_acquired": None,
                        "detections": {},
                        "current_stage": None,
                    },
                )
                if data.get("x") is not None:
                    emb["stage_x_um"] = data["x"]
                if data.get("y") is not None:
                    emb["stage_y_um"] = data["y"]
                if data.get("role"):
                    emb["role"] = data["role"]
                if "strain" in data:
                    emb["strain"] = data.get("strain")
                if data.get("uid"):
                    emb["uid"] = data["uid"]
                if data.get("user_label"):
                    emb["user_label"] = data["user_label"]
                if data.get("confidence") is not None:
                    emb["confidence"] = data["confidence"]

        elif event_type in ("DETECTION_TRIGGERED", "HATCHING_DETECTED"):
            # Positive detection events - update embryo status
            eid = data.get("embryo_id")
            if eid and eid in self.embryos:
                detector_name = data.get("detector_name", "unknown")
                self.embryos[eid]["detections"][detector_name] = {
                    "detected": True,
                    "confidence": data.get("confidence"),
                }
                if detector_name == "hatching":
                    self.embryos[eid]["is_complete"] = True
                    self.embryos[eid].setdefault("completed_at", datetime.now().isoformat())

        elif event_type == "VERIFICATION_STARTED":
            # Verification round started for embryo
            eid = data.get("embryo_id")
            if eid and eid in self.embryos:
                self.embryos[eid]["verification"] = {
                    "status": "running",
                    "consecutive_count": data.get("consecutive_count", 0),
                    "required_count": data.get("required_count", 5),
                    "strategies_complete": 0,
                    "total_strategies": 5,
                    "strategies": {},
                }

        elif event_type == "VERIFICATION_STRATEGY":
            # Individual strategy result
            eid = data.get("embryo_id")
            if eid and eid in self.embryos and "verification" in self.embryos[eid]:
                strategy = data.get("strategy")
                self.embryos[eid]["verification"]["strategies"][strategy] = {
                    "passed": data.get("passed"),
                    "summary": data.get("summary"),
                }

        elif event_type == "VERIFICATION_PROGRESS":
            # Progress update
            eid = data.get("embryo_id")
            if eid and eid in self.embryos and "verification" in self.embryos[eid]:
                self.embryos[eid]["verification"]["strategies_complete"] = data.get(
                    "strategies_complete", 0
                )
                self.embryos[eid]["verification"]["total_strategies"] = data.get(
                    "total_strategies", 5
                )

        elif event_type == "VERIFICATION_COMPLETED":
            # Final verification result
            eid = data.get("embryo_id")
            if eid and eid in self.embryos:
                self.embryos[eid]["verification"] = {
                    "status": "completed",
                    "consensus": data.get("consensus"),
                    "reasoning": data.get("reasoning"),
                    "strategies": data.get("strategies", {}),
                    "ensemble_votes": data.get("ensemble_votes"),
                    "duration_seconds": data.get("duration_seconds"),
                }
                # Update consecutive count display
                if data.get("consensus"):
                    current = self.embryos[eid].get("consecutive_verified", 0)
                    self.embryos[eid]["consecutive_verified"] = current + 1
                else:
                    self.embryos[eid]["consecutive_verified"] = 0

        elif event_type == "STATUS_CHANGED":
            if data.get("status"):
                self.status = data["status"]
            # Handle interval changes
            if data.get("embryo_id") and data.get("new_interval_seconds"):
                eid = data["embryo_id"]
                if eid in self.embryos:
                    self.embryos[eid]["interval_seconds"] = data["new_interval_seconds"]
            # Photodose-budget exceedance — surface as a pause reason.
            if data.get("change") == "photodose_budget_exceeded":
                eid = data.get("embryo_id")
                if eid and eid in self.embryos:
                    self.embryos[eid]["cadence_phase"] = "paused"
                    self.embryos[eid]["photodose_paused"] = True
                    self.embryos[eid]["dose_budget_ms"] = data.get("budget_ms")
                    self.embryos[eid]["total_exposure_ms"] = data.get("total_exposure_ms")
            # Role re-assignment via assign_embryo_roles agent tool.
            if data.get("change") == "role_assigned":
                eid = data.get("embryo_id")
                if eid:
                    emb = self.embryos.setdefault(
                        eid,
                        {
                            "embryo_id": eid,
                            "timepoints": 0,
                            "is_complete": False,
                            "detections": {},
                            "current_stage": None,
                        },
                    )
                    if data.get("new_role"):
                        emb["role"] = data["new_role"]

        # -- Phase 10: async timelapse events --------------------------

        elif event_type == "EMBRYO_CADENCE_CHANGED":
            eid = data.get("embryo_id")
            if eid:
                emb = self.embryos.setdefault(
                    eid,
                    {
                        "embryo_id": eid,
                        "timepoints": 0,
                        "is_complete": False,
                        "detections": {},
                        "current_stage": None,
                    },
                )
                if data.get("new_phase") is not None:
                    emb["cadence_phase"] = data["new_phase"]
                if data.get("new_interval_s") is not None:
                    emb["interval_seconds"] = data["new_interval_s"]
                if data.get("next_due_at"):
                    emb["next_due_at"] = data["next_due_at"]
                emb["last_cadence_change_reason"] = data.get("reason")

        elif event_type == "POWER_RAMP_STEP":
            eid = data.get("embryo_id")
            if eid:
                emb = self.embryos.setdefault(
                    eid,
                    {
                        "embryo_id": eid,
                        "timepoints": 0,
                        "is_complete": False,
                        "detections": {},
                        "current_stage": None,
                    },
                )
                wavelength = data.get("wavelength", 488)
                if wavelength == 488:
                    emb["laser_power_488_pct"] = data.get("new_pct")
                emb.setdefault("power_history", []).append(
                    {
                        "wavelength": wavelength,
                        "old_pct": data.get("old_pct"),
                        "new_pct": data.get("new_pct"),
                        "direction": data.get("direction"),
                        "rule": data.get("rule"),
                        "intensity_level": data.get("intensity_level"),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                # cap history per embryo
                if len(emb["power_history"]) > 200:
                    emb["power_history"] = emb["power_history"][-200:]

        elif event_type == "CLAUDE_DETECTOR_RESULT":
            eid = data.get("embryo_id")
            if eid:
                emb = self.embryos.setdefault(
                    eid,
                    {
                        "embryo_id": eid,
                        "timepoints": 0,
                        "is_complete": False,
                        "detections": {},
                        "current_stage": None,
                    },
                )
                findings = data.get("findings") or {}
                emb["last_intensity_level"] = findings.get("intensity_level")
                emb["last_structure_quality"] = findings.get("structure_quality")
                emb["last_detector_name"] = data.get("detector_name")
                if findings.get("has_hatched"):
                    emb["hatched"] = True

        elif event_type in (
            "BURST_QUEUED",
            "BURST_START",
            "BURST_FRAME",
            "BURST_COMPLETE",
        ):
            eid = data.get("embryo_id")
            if eid:
                emb = self.embryos.setdefault(
                    eid,
                    {
                        "embryo_id": eid,
                        "timepoints": 0,
                        "is_complete": False,
                        "detections": {},
                        "current_stage": None,
                    },
                )
                emb.setdefault("burst", {})
                burst_state = emb["burst"]
                if event_type == "BURST_QUEUED":
                    burst_state["status"] = "queued"
                    burst_state["request_id"] = data.get("request_id")
                    burst_state["queue_position"] = data.get("position_in_queue")
                    burst_state["frames"] = data.get("frames")
                    burst_state["mode"] = data.get("mode")
                elif event_type == "BURST_START":
                    burst_state["status"] = "running"
                    burst_state["request_id"] = data.get("request_id")
                    burst_state["frames"] = data.get("frames")
                    burst_state["mode"] = data.get("mode")
                    burst_state["current_frame"] = 0
                    burst_state["started_at"] = datetime.now().isoformat()
                    emb["cadence_phase"] = "burst"
                elif event_type == "BURST_FRAME":
                    burst_state["current_frame"] = data.get("frame_idx", 0)
                    burst_state["total_frames"] = data.get("total_frames")
                elif event_type == "BURST_COMPLETE":
                    burst_state["status"] = "complete"
                    burst_state["frames_captured"] = data.get("frames_captured")
                    burst_state["duration_s"] = data.get("duration_s")
                    burst_state["sustained_hz"] = data.get("sustained_hz")
                    burst_state["mp4_path"] = data.get("mp4_path")
                    if emb.get("cadence_phase") == "burst":
                        emb["cadence_phase"] = "normal"

    def to_dict(self) -> dict:
        """Serialize for WebSocket transmission"""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "started_at": self.started_at,
            "embryos": self.embryos,
            "total_timepoints": self.total_timepoints,
            "base_interval": self.base_interval,
            "detection_reasoning": self.detection_reasoning,
        }

    def reset(self):
        """Clear state for new timelapse"""
        self.__init__()

    def seed_from_experiment(self, experiment) -> int:
        """Seed embryo positions / roles directly from an
        ``ExperimentState``. Belt-and-suspenders for the event-bus
        replay path: if the tracker started AFTER ``add_embryo`` calls
        fired (e.g. session resume happens in agent ``__init__`` but
        ``start_viz_server`` runs later), the in-memory event history
        may have rolled past those events. This pulls the current
        truth from the agent's experiment directly.

        Returns the number of embryos seeded.
        """
        if experiment is None:
            return 0
        seeded = 0
        for eid, emb in (getattr(experiment, "embryos", {}) or {}).items():
            pos = getattr(emb, "stage_position", None) or {}
            x = pos.get("x") if isinstance(pos, dict) else None
            y = pos.get("y") if isinstance(pos, dict) else None
            if x is None or y is None:
                continue
            self.handle_event(
                "EMBRYO_DETECTED",
                {
                    "embryo_id": eid,
                    "uid": getattr(emb, "uid", None),
                    "x": x,
                    "y": y,
                    "role": getattr(emb, "role", "test"),
                    "strain": getattr(emb, "strain", None),
                    "user_label": getattr(emb, "user_label", None),
                    "confidence": getattr(emb, "detection_confidence", None),
                },
            )
            seeded += 1
        return seeded
