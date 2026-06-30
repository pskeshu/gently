"""
Timelapse Orchestration Tools

Tools for managing adaptive timelapse acquisitions.
"""

from gently.harness.tools.helpers import (
    ctx_get,
    get_embryo_or_error,
    require_agent,
    require_developmental_tracker,
    require_timelapse_orchestrator,
)
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="generate_bluesky_plan",
    description="Generate a Bluesky acquisition plan from a scientific goal",
    category=ToolCategory.EXPERIMENT,
)
async def generate_bluesky_plan(
    goal: str,
    embryo_ids: list[str],
    plan_type: str = "adaptive_timelapse",
    parameters: dict | None = None,
    context: dict | None = None,
) -> str:
    """Generate Bluesky plan"""
    agent = ctx_get(context, "agent")

    if not agent:
        return "Error: No agent context"

    try:
        result = await agent._generate_bluesky_plan(
            goal=goal,
            embryo_ids=embryo_ids,
            plan_type=plan_type,
            parameters=parameters or {},
        )
        return result

    except Exception as e:
        return f"Error generating plan: {str(e)}"


@tool(
    name="start_adaptive_timelapse",
    description=(
        "Start an adaptive timelapse that runs in the background. Agent remains responsive"
        " while acquisition continues. "
        "Pass `monitoring_mode='expression_monitoring'` for fluorescent-reporter experiments"
        " to install reactive cadence + power rules at startup. "
        "Other monitoring_mode values: 'pre_terminal_monitoring' (hatching-timing"
        " experiments), 'idle' (plain imaging, no reactive rules)."
    ),
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def start_adaptive_timelapse(
    embryo_ids: list[str] | None = None,
    stop_condition: str = "manual",
    interval_seconds: float = 120.0,
    condition_value: int | None = None,
    monitoring_mode: str | None = None,
    tactic_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Start adaptive timelapse in background.

    ``tactic_id`` (optional) links this start to a tactic in the session
    Operation Plan: on success the tactic is transitioned to 'active', giving
    lifecycle symmetry with stop/pause/enable_monitoring_mode/queue_burst.
    """
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    try:
        result = await orchestrator.start(
            embryo_ids=embryo_ids,
            stop_condition=stop_condition,
            base_interval_seconds=interval_seconds,
            condition_value=condition_value,
        )

        # Best-effort auto-link to a matching plan item (Feature 3)
        try:
            cs = getattr(agent, "context_store", None)
            if cs:
                from .plan_execution_tools import try_auto_link_plan_item

                session_id = getattr(agent, "session_id", None)
                if session_id:
                    linked = try_auto_link_plan_item(
                        cs,
                        session_id,
                        stop_condition,
                        interval_seconds,
                    )
                    if linked:
                        result += f"\n(Auto-linked to plan item: '{linked}')"
        except Exception:
            pass  # Never block timelapse

        # Optionally install a monitoring mode at startup. The mode IS the
        # reactive control — without one, embryos stay at base interval
        # regardless of what detectors observe.
        if monitoring_mode and monitoring_mode != "idle":
            try:
                mode_result = orchestrator.enable_monitoring_mode(monitoring_mode)
                result += f"\n{mode_result}"
            except Exception as e:
                result += f"\n(Failed to enable monitoring mode '{monitoring_mode}': {e})"

        # Lifecycle symmetry: mark the linked tactic active (best-effort).
        if tactic_id:
            try:
                cs = getattr(agent, "context_store", None)
                session_id = getattr(agent, "session_id", None)
                if cs and session_id:
                    cs.transition_tactic(session_id, tactic_id, "active")
            except Exception:
                pass  # never block the timelapse

        return result
    except Exception as e:
        return f"Error starting timelapse: {str(e)}"


@tool(
    name="get_timelapse_status",
    description="Get current status of the running timelapse including per-embryo progress",
    category=ToolCategory.EXPERIMENT,
)
def get_timelapse_status(context: dict | None = None) -> str:
    """Get timelapse status"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    state = orchestrator.get_status()
    status_dict = state.to_dict()

    lines = [f"Timelapse Status: {status_dict['status'].upper()}", ""]

    if status_dict["started_at"]:
        lines.append(f"Started: {status_dict['started_at']}")
        lines.append(f"Duration: {status_dict['duration_minutes']:.1f} minutes")
        lines.append(f"Total timepoints acquired: {status_dict['total_timepoints']}")
        lines.append("")

    lines.append(f"Active embryos: {status_dict['active_embryos']}")
    lines.append(f"Completed embryos: {status_dict['completed_embryos']}")
    lines.append("")

    if status_dict.get("seconds_until_next_round") is not None:
        lines.append(f"Next acquisition in {status_dict['seconds_until_next_round']:.0f}s")
        lines.append("")

    if status_dict["embryo_details"]:
        lines.append("Embryo Details:")
        for eid, details in status_dict["embryo_details"].items():
            status_marker = "[done]" if details["is_complete"] else "[active]"
            lines.append(f"  {status_marker} {eid}: t={details['timepoints']}")
            if details["is_complete"]:
                lines.append(f"      Completed: {details['completion_reason']}")

    if status_dict["error"]:
        lines.append("")
        lines.append(f"Error: {status_dict['error']}")

    return "\n".join(lines)


@tool(
    name="modify_timelapse_embryo",
    description=(
        "Modify parameters for a specific embryo during a running timelapse."
        " Note: interval is now global - use modify_timelapse_interval to change it."
    ),
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def modify_timelapse_embryo(
    embryo_id: str,
    stop_condition: str | None = None,
    condition_value: int | None = None,
    context: dict | None = None,
) -> str:
    """Modify embryo parameters during timelapse (stop condition only - interval is global)"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.modify_embryo(
            embryo_id=embryo_id,
            stop_condition=stop_condition,
            condition_value=condition_value,
        )
        return result
    except Exception as e:
        return f"Error modifying embryo: {str(e)}"


@tool(
    name="add_embryo_to_timelapse",
    description=(
        "Add an embryo to an already running timelapse. The embryo will use the global"
        " interval and join on the next round."
    ),
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def add_embryo_to_timelapse(
    embryo_id: str,
    stop_condition: str | None = None,
    condition_value: int | None = None,
    context: dict | None = None,
) -> str:
    """Add an embryo to a running timelapse (uses global interval)"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return "No timelapse running. Use start_adaptive_timelapse first."

    try:
        result = await orchestrator.add_embryo(
            embryo_id=embryo_id,
            stop_condition=stop_condition,
            condition_value=condition_value,
        )
        return result
    except Exception as e:
        return f"Error adding embryo: {str(e)}"


@tool(
    name="stop_timelapse_embryo",
    description="Stop imaging a specific embryo in the timelapse (other embryos continue)",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def stop_timelapse_embryo(
    embryo_id: str, reason: str = "user_request", context: dict | None = None
) -> str:
    """Stop imaging a specific embryo"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.stop_embryo(embryo_id, reason)
        return result
    except Exception as e:
        return f"Error stopping embryo: {str(e)}"


@tool(
    name="stop_timelapse",
    description="Stop the entire timelapse acquisition",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def stop_timelapse(
    reason: str = "user_request",
    tactic_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Stop entire timelapse"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.stop(reason)
        # Mark the linked plan tactic done (guarded no-op when absent).
        if tactic_id:
            cs = getattr(agent, "context_store", None)
            session = getattr(agent, "session_id", None)
            if cs and session:
                cs.transition_tactic(session, tactic_id, "done")
        return result
    except Exception as e:
        return f"Error stopping timelapse: {str(e)}"


@tool(
    name="pause_timelapse",
    description="Pause the timelapse acquisition (can be resumed later)",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def pause_timelapse(
    tactic_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Pause timelapse"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.pause()
        # Mark the linked plan tactic paused (guarded no-op when absent).
        if tactic_id:
            cs = getattr(agent, "context_store", None)
            session = getattr(agent, "session_id", None)
            if cs and session:
                cs.transition_tactic(session, tactic_id, "paused")
        return result
    except Exception as e:
        return f"Error pausing timelapse: {str(e)}"


@tool(
    name="resume_timelapse",
    description="Resume a paused timelapse",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def resume_timelapse(context: dict | None = None) -> str:
    """Resume timelapse"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return "No timelapse to resume."

    try:
        result = await orchestrator.resume()
        return result
    except Exception as e:
        return f"Error resuming timelapse: {str(e)}"


@tool(
    name="add_stop_condition",
    description=(
        "Add an additional stop condition to a running timelapse (OR logic). E.g., add"
        " 'hatching' condition to a timelapse running with a duration limit."
    ),
    category=ToolCategory.EXPERIMENT,
)
def add_stop_condition(embryo_id: str, condition: str, context: dict | None = None) -> str:
    """
    Add an additional stop condition to an embryo in a running timelapse.

    The new condition is added with OR logic - the embryo will stop when
    ANY condition is met (the original OR the new one).

    Parameters
    ----------
    embryo_id : str
        ID of the embryo to modify
    condition : str
        Stop condition to add. Supported formats:
        - "hatching" - stop when hatching detected
        - "comma" or "comma_stage" - stop at comma stage
        - "duration:10" or "duration:10h" - stop after 10 hours
        - "timepoints:100" - stop after 100 timepoints

    Examples
    --------
    - Timelapse started with "duration:10h", add hatching detection:
      add_stop_condition(embryo_id="embryo1", condition="hatching")
      Result: Embryo stops on hatching OR 10h, whichever comes first

    Returns
    -------
    str
        Confirmation message with new stop condition description
    """
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return "No timelapse running."

    # Parse the new condition
    from gently.app.orchestration.timelapse import StopCondition

    try:
        new_condition = StopCondition.parse(condition)
    except ValueError as e:
        return f"Invalid stop condition: {e}"

    # Find the embryo state in the orchestrator
    embryo_state = orchestrator._embryo_states.get(embryo_id)
    if not embryo_state:
        available = list(orchestrator._embryo_states.keys())
        return f"Embryo '{embryo_id}' not found in timelapse. Available: {available}"

    # Add the condition
    embryo_state.stop_condition.add_condition(new_condition)

    # Get updated description
    new_desc = embryo_state.stop_condition.describe()

    return f"Added stop condition '{condition}' to {embryo_id}\nStop conditions: {new_desc}"


@tool(
    name="add_interval_speedup_rule",
    description=(
        "Add a rule to automatically speed up imaging when a developmental stage is"
        " reached (e.g., 'speed up to 30s when 3fold stage detected')"
    ),
    category=ToolCategory.EXPERIMENT,
)
def add_interval_speedup_rule(
    trigger_stage: str,
    new_interval_seconds: float = 30.0,
    embryo_ids: list[str] | None = None,
    context: dict | None = None,
) -> str:
    """Add interval speedup rule based on developmental stage"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    orchestrator.add_speedup_on_stage(
        stage_name=trigger_stage,
        new_interval_seconds=new_interval_seconds,
        embryo_ids=embryo_ids,
    )

    msg = (
        f"Added interval rule: speed up to {new_interval_seconds}s when"
        f" '{trigger_stage}' stage is reached"
    )
    if embryo_ids:
        msg += f" (for embryos: {', '.join(embryo_ids)})"

    return msg


@tool(
    name="enable_pre_hatching_speedup",
    description=(
        "Enable automatic speedup when embryos approach hatching (triggers when 3-fold"
        " stage is detected by the perception system)"
    ),
    category=ToolCategory.EXPERIMENT,
)
def enable_pre_hatching_speedup(
    fast_interval_seconds: float = 30.0, context: dict | None = None
) -> str:
    """Enable pre-hatching speedup based on developmental stage"""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    # Add speedup rule for pre-terminal stage (uses perception system + organism config)
    orchestrator.add_pre_terminal_speedup(fast_interval_seconds)

    from gently.organisms import get_organism

    organism = get_organism()
    trigger_stage = organism.PRE_TERMINAL_SPEEDUP_STAGE

    return (
        f"Enabled pre-hatching speedup:\n"
        f"  - Perception system will detect developmental stages\n"
        f"  - When {trigger_stage} stage detected, interval will change to"
        f" {fast_interval_seconds}s\n"
        f"  - This helps capture hatching at high temporal resolution"
    )


@tool(
    name="classify_embryo_stage",
    description="Use Claude Vision to classify the current developmental stage of an embryo",
    category=ToolCategory.ANALYSIS,
)
async def classify_embryo_stage(embryo_id: str, context: dict | None = None) -> str:
    """Classify embryo stage"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    if not embryo.recent_images:
        return f"No images available for {embryo_id}. Acquire a volume first."

    latest = embryo.recent_images[-1]

    # Initialize tracker if needed
    if not hasattr(agent, "developmental_tracker") or agent.developmental_tracker is None:
        from ..developmental_tracker import DevelopmentalTracker

        agent.developmental_tracker = DevelopmentalTracker(
            claude_client=agent.claude,
            model=agent.model,
        )

    recent = []
    for img in embryo.recent_images[-5:]:
        recent.append(
            {
                "timepoint": img.timepoint,
                "b64_image": img.max_projection_b64,
            }
        )

    result = agent.developmental_tracker.classify_stage(
        image_b64=latest.max_projection_b64,
        embryo_id=embryo.id,
        timepoint=latest.timepoint,
        recent_images=recent,
    )

    lines = [
        f"Stage Classification for {embryo.id}:",
        f"  Stage: {result.stage.value}",
        f"  Confidence: {result.confidence}",
        f"  Reasoning: {result.reasoning}",
    ]

    if result.predicted_minutes_to_hatching is not None:
        hours = result.predicted_minutes_to_hatching / 60
        lines.append(
            f"  Predicted time to hatching: ~{hours:.1f} hours"
            f" ({result.predicted_minutes_to_hatching} min)"
        )

    return "\n".join(lines)


@tool(
    name="get_stage_history",
    description="Get the developmental stage progression history for an embryo",
    category=ToolCategory.ANALYSIS,
)
def get_stage_history(embryo_id: str, context: dict | None = None) -> str:
    """Get stage history"""
    agent, err = require_agent(context)
    if err:
        return err

    # Prefer the live perception session (the orchestrator's Perceiver, which the
    # agent shares). The DevelopmentalTracker below is only populated by manual
    # classify_embryo_stage calls, so it is usually empty in autonomous runs.
    perceiver = getattr(agent, "perceiver", None)
    session = perceiver.get_session(embryo_id) if perceiver else None
    if session is not None and getattr(session, "current_stage", None):
        s = session.summary()
        lines = [
            f"Stage progression for {embryo_id} (live perception):",
            f"  Current stage: {s.get('current_stage')} (stable for {s.get('stability', 0)} obs)",
            f"  Observations: {s.get('observation_count', 0)}",
        ]
        seq = s.get("stage_sequence") or []
        if seq:
            lines.append(f"  Trajectory: {' -> '.join(seq)}")
        t = s.get("temporal")  # TemporalContext dataclass or None
        if t is not None:
            exp = getattr(t, "expected_duration_min", None)
            seg = f"  Time in current stage: {getattr(t, 'time_in_stage_min', 0.0):.0f} min"
            if exp:
                seg += f" (expected ~{exp:.0f} min)"
            lines.append(seg)
            if getattr(t, "is_potentially_arrested", False):
                lines.append("  ** potentially ARRESTED **")
        return "\n".join(lines)

    tracker, err = require_developmental_tracker(agent)
    if err:
        return err

    summary = tracker.get_progression_summary(embryo_id)

    if summary["observations"] == 0:
        return f"No stage classifications for {embryo_id}. Use classify_embryo_stage first."

    lines = [
        f"Stage Progression for {embryo_id}:",
        f"  Current stage: {summary['current_stage']} ({summary['current_confidence']})",
        f"  Observations: {summary['observations']}",
        f"  Stages observed: {', '.join(summary['stages_observed'])}",
    ]

    if summary["predicted_minutes_to_hatching"] is not None:
        hours = summary["predicted_minutes_to_hatching"] / 60
        lines.append(f"  Predicted time to hatching: ~{hours:.1f} hours")

    return "\n".join(lines)


def _perceiver_hatching_estimate(session) -> float | None:
    """Estimate minutes until the 'hatching' stage from the perception session.

    Uses gently_perception's own organism stage ordering + typical durations, so
    no DevelopmentalStage enum mapping is needed. Returns None when unknown
    (no_object / off-vocabulary stage), 0.0 when already hatching/hatched.
    """
    try:
        from gently_perception.organism import CELEGANS
    except Exception:
        return None
    stage = getattr(session, "current_stage", None)
    if not stage or stage == "no_object":
        return None
    stages = list(CELEGANS.stages)
    durations = dict(CELEGANS.stage_durations)
    if stage in ("hatching", "hatched"):
        return 0.0
    if stage not in stages or "hatching" not in stages:
        return None
    idx = stages.index(stage)
    target = stages.index("hatching")
    if idx >= target:
        return 0.0
    # Remaining time in the current stage (expected minus already-elapsed).
    elapsed = 0.0
    t = session.summary().get("temporal")
    if t is not None:
        elapsed = getattr(t, "time_in_stage_min", 0.0) or 0.0
    remaining = max(0.0, durations.get(stage, 0.0) - elapsed)
    # Plus the full expected duration of each stage between current and hatching.
    for s in stages[idx + 1 : target]:
        remaining += durations.get(s, 0.0)
    return remaining


@tool(
    name="predict_hatching",
    description=(
        "Predict time-to-hatching for an embryo with confidence intervals based on"
        " developmental stage"
    ),
    category=ToolCategory.ANALYSIS,
)
def predict_hatching(
    embryo_id: str | None = None, all_embryos: bool = False, context: dict | None = None
) -> str:
    """Predict hatching time with confidence intervals"""
    agent, err = require_agent(context)
    if err:
        return err

    # Prefer the live perception session; the DevelopmentalTracker is usually
    # empty in autonomous runs (only manual classify_embryo_stage feeds it).
    perceiver = getattr(agent, "perceiver", None)

    def _perc_line(eid: str):
        session = perceiver.get_session(eid) if perceiver else None
        if session is None or not getattr(session, "current_stage", None):
            return None
        stage = session.current_stage
        if stage in ("hatching", "hatched"):
            return f"  {eid}: stage={stage} (hatching now / already hatched)"
        est = _perceiver_hatching_estimate(session)
        if est is None:
            return f"  {eid}: stage={stage} (time-to-hatching unknown)"
        return f"  {eid}: stage={stage}, ~{est / 60:.1f}h to hatching ({est:.0f} min)"

    if perceiver is not None:
        if all_embryos:
            ids = list(agent.experiment.embryos.keys())
            perc = [_perc_line(e) for e in ids]
            if any(perc):
                out = ["Hatching predictions (live perception):", ""]
                out += [p for p in perc if p]
                missing = [e for e, p in zip(ids, perc, strict=False) if not p]
                if missing:
                    out.append("")
                    out.append(f"(no perception yet for: {', '.join(missing)})")
                return "\n".join(out)
        elif embryo_id:
            line = _perc_line(embryo_id)
            if line:
                return f"Hatching prediction for {embryo_id} (live perception):\n{line}"

    tracker, err = require_developmental_tracker(agent)
    if err:
        return err

    if all_embryos:
        embryo_ids = list(agent.experiment.embryos.keys())
        predictions = tracker.get_all_predictions(embryo_ids)

        if not predictions:
            return "No predictions available. Use classify_embryo_stage on embryos first."

        lines = ["Hatching Predictions:", ""]

        for eid, pred in predictions.items():
            lines.append(f"  {eid}:")
            lines.append(f"    Current stage: {pred.current_stage.value}")
            lines.append(
                f"    Predicted: {pred.predicted_hours:.1f}h ({pred.predicted_minutes} min)"
            )
            lines.append(f"    Range: {pred.range_hours[0]:.1f} - {pred.range_hours[1]:.1f}h")
            lines.append(f"    Confidence: {pred.confidence}")
            lines.append("")

        lines.append("Development Rates:")
        for eid in predictions.keys():
            rate = tracker.estimate_development_rate(eid)
            if rate:
                rate_pct = (rate - 1.0) * 100
                speed = "faster" if rate > 1.0 else "slower"
                lines.append(f"  {eid}: {abs(rate_pct):.1f}% {speed} than standard")

        return "\n".join(lines)

    else:
        if not embryo_id:
            return "Specify embryo_id or set all_embryos=True"

        pred = tracker.get_hatching_prediction(embryo_id)
        if not pred:
            return f"No prediction available for {embryo_id}. Use classify_embryo_stage first."

        lines = [
            f"Hatching Prediction for {embryo_id}:",
            f"  Current stage: {pred.current_stage.value}",
            f"  Predicted time to hatching: {pred.predicted_hours:.1f} hours"
            f" ({pred.predicted_minutes} min)",
            f"  Confidence interval: {pred.range_hours[0]:.1f} - {pred.range_hours[1]:.1f} hours",
            f"  Classification confidence: {pred.confidence}",
        ]

        rate = tracker.estimate_development_rate(embryo_id)
        if rate:
            rate_pct = (rate - 1.0) * 100
            speed = "faster" if rate > 1.0 else "slower"
            lines.append(f"  Development rate: {abs(rate_pct):.1f}% {speed} than standard")

        return "\n".join(lines)


@tool(
    name="set_autonomy",
    description="""Set the autonomy mode of the decision-moment wake-router (default OFF).
Modes:
  'off'  — never act on its own; only respond to your messages.
  'ask'  — on a notable event (stage transition, arrest, hatching, termination,
           errors) the agent PROPOSES a change and waits for you to Approve /
           Modify / Skip in the chat before acting.
  'auto' — the agent adapts acquisition on its own (still bounded by device
           limits; a few irreversible actions always require your confirmation).
You can switch modes mid-run. Use when the user says "enable autopilot/autonomous",
"ask me before changing things", "go fully autonomous", or "turn off autonomy".""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("Ask me before adapting", {"mode": "ask"}),
        ToolExample("Go fully autonomous", {"mode": "auto"}),
        ToolExample("Turn off autonomy", {"mode": "off"}),
    ],
)
def set_autonomy(
    mode: str | None = None, enabled: bool | None = None, context: dict | None = None
) -> str:
    """Set the wake-router mode (off/ask/auto). `enabled` kept for back-compat."""
    agent, err = require_agent(context)
    if err:
        return err
    router = getattr(agent, "wake_router", None)
    if router is None:
        return "Autonomy is not available (wake-router failed to initialize)."
    if mode is not None:
        m = str(mode).strip().lower()
        if m not in ("off", "ask", "auto"):
            return "mode must be 'off', 'ask', or 'auto'."
        router.set_mode(m)
    elif enabled is not None:
        router.set_enabled(bool(enabled))
    else:
        return "Specify mode ('off', 'ask', or 'auto')."
    cur = router.mode
    if cur == "auto":
        return (
            "Autonomy set to AUTO. I'll wake on stage transitions, arrest, "
            "hatching, termination, and errors and adapt acquisition on my own "
            "(irreversible actions still need your okay). Say 'ask mode' or "
            "'turn off autonomy' to change."
        )
    if cur == "ask":
        return (
            "Autonomy set to ASK. On a notable event I'll propose a change and "
            "wait for your Approve / Modify / Skip before doing anything."
        )
    return "Autonomy OFF. I'll only act when you message me."


# ---------------------------------------------------------------------------
# Live cadence / dose modulation — direct knobs for a running timelapse.
# ---------------------------------------------------------------------------


@tool(
    name="modify_timelapse_interval",
    description="""Change the base acquisition interval for ALL embryos on a running
timelapse, effective immediately.
Re-anchors every embryo's next acquisition to now + the new interval and notifies the UI.
Lower interval = more frequent imaging = more photodose; raise it to be gentler.
Use when the user says "image every N minutes/seconds now", "speed up/slow down the whole run".
For a single embryo use set_embryo_cadence instead.""",
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample("Image every 2 minutes now", {"new_interval_seconds": 120}),
        ToolExample("Slow everything down to 10 minutes", {"new_interval_seconds": 600}),
    ],
)
def modify_timelapse_interval(new_interval_seconds: float, context: dict | None = None) -> str:
    """Globally re-anchor the timelapse interval (live)."""
    agent, err = require_agent(context)
    if err:
        return err
    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err
    return orchestrator.modify_interval(new_interval_seconds)


@tool(
    name="set_embryo_cadence",
    description="""Change ONE embryo's acquisition cadence on a running timelapse, effective
immediately. Set new_interval_seconds to re-anchor that embryo's next acquisition to now +
interval (lower = more frequent = more dose).
Set new_phase to 'normal' to resume a paused embryo, or 'paused' to pause it.
NOTE: re-issuing the SAME interval with the SAME phase is a no-op (it won't re-anchor).
Use for per-embryo tuning, e.g. speed up the one that's developing fastest.""",
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            "Image embryo_2 every minute",
            {"embryo_id": "embryo_2", "new_interval_seconds": 60},
        ),
        ToolExample("Resume embryo_3", {"embryo_id": "embryo_3", "new_phase": "normal"}),
    ],
)
def set_embryo_cadence(
    embryo_id: str,
    new_interval_seconds: float | None = None,
    new_phase: str | None = None,
    context: dict | None = None,
) -> str:
    """Per-embryo cadence change routed through the re-anchoring path."""
    agent, err = require_agent(context)
    if err:
        return err
    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err
    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err
    if new_interval_seconds is None and new_phase is None:
        return "Specify new_interval_seconds and/or new_phase."
    if new_interval_seconds is not None and new_interval_seconds < 1:
        return "Interval must be >= 1 second."
    if new_phase is not None and new_phase not in ("normal", "fast", "burst", "paused"):
        return "new_phase must be one of: normal, fast, burst, paused."
    # Detect the no-op (transition_cadence silently does nothing, and would NOT
    # re-anchor next_due_at, if neither interval nor phase actually changes).
    cur_interval = getattr(embryo, "interval_seconds", None)
    cur_phase = getattr(embryo, "cadence_phase", None)
    interval_change = new_interval_seconds is not None and new_interval_seconds != cur_interval
    phase_change = new_phase is not None and new_phase != cur_phase
    if not interval_change and not phase_change:
        shown = f"{cur_interval:.0f}s" if cur_interval is not None else "default"
        return f"{embryo.id}: no change (already interval={shown}, phase={cur_phase})."
    orchestrator.transition_cadence(
        embryo,
        new_interval_seconds=new_interval_seconds if interval_change else None,
        new_phase=new_phase if phase_change else None,
        reason="agent:set_embryo_cadence",
    )
    bits = []
    if interval_change:
        bits.append(f"interval={new_interval_seconds:.0f}s")
    if phase_change:
        bits.append(f"phase={new_phase}")
    due = getattr(embryo, "next_due_at", None)
    tail = f"; next acquisition ~{due.strftime('%H:%M:%S')}" if due else ""
    return f"{embryo.id}: {', '.join(bits)}{tail}"


@tool(
    name="set_photodose_budget",
    description="""Set or clear the per-embryo photodose budget (a hard cap on cumulative
laser exposure). base_dose_budget_ms is the ceiling for a 1x-role (test) embryo;
calibration embryos get 10x.
When an embryo's cumulative exposure exceeds its budget it is auto-PAUSED to protect the sample.
Pass null/None to DISABLE the cap. Raising the budget also resumes embryos that were paused
for the old cap.
Use to enforce gentleness on precious samples, or to lift the cap when the user okays more dose.""",
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample("Cap each embryo at 5 seconds of light", {"base_dose_budget_ms": 5000}),
        ToolExample("Remove the photodose cap", {"base_dose_budget_ms": None}),
    ],
)
def set_photodose_budget(
    base_dose_budget_ms: float | None = None,
    resume_paused: bool = True,
    context: dict | None = None,
) -> str:
    """Set/clear the photodose budget; optionally resume budget-paused embryos."""
    agent, err = require_agent(context)
    if err:
        return err
    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err
    # Capture who was budget-paused BEFORE set_photodose_budget clears the set,
    # so we only resume embryos paused for the budget (not manual pauses/bursts).
    prev_exceeded = set(getattr(orchestrator, "_dose_budget_exceeded", set()) or set())
    msg = orchestrator.set_photodose_budget(base_dose_budget_ms)
    resumed = []
    if resume_paused:
        states = getattr(orchestrator, "_embryo_states", {}) or {}
        try:
            from gently.harness.roles import REGISTRY as ROLE_REGISTRY
        except Exception:
            ROLE_REGISTRY = {}
        for eid in prev_exceeded:
            e = states.get(eid)
            if e is None or getattr(e, "cadence_phase", None) != "paused":
                continue
            # Only resume if the embryo is now UNDER the new budget (or the cap
            # was disabled); otherwise it would just immediately re-pause.
            if base_dose_budget_ms is not None:
                rdef = (
                    ROLE_REGISTRY.get(getattr(e, "role", "test"))
                    if hasattr(ROLE_REGISTRY, "get")
                    else None
                )
                mult = getattr(rdef, "photodose_budget_multiplier", 1.0) if rdef else 1.0
                if (getattr(e, "total_exposure_ms", 0.0) or 0.0) > base_dose_budget_ms * mult:
                    continue
            orchestrator.transition_cadence(
                e, new_phase="normal", reason="agent:budget change resume"
            )
            resumed.append(eid)
    if resumed:
        msg += f" Resumed: {', '.join(sorted(resumed))}."
    return msg


@tool(
    name="get_photodose_status",
    description="""Report each embryo's cumulative light exposure vs its photodose budget,
and which are paused over budget.
Use to reason about gentleness before/after changing the budget, power, or cadence.""",
    category=ToolCategory.ANALYSIS,
    examples=[ToolExample("How much light has each embryo gotten?", {})],
)
def get_photodose_status(context: dict | None = None) -> str:
    """Read-only photodose / budget status across embryos."""
    agent, err = require_agent(context)
    if err:
        return err
    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err
    base = getattr(orchestrator, "_dose_budget_base_ms", None)
    exceeded: set[str] = getattr(orchestrator, "_dose_budget_exceeded", set()) or set()
    states = getattr(orchestrator, "_embryo_states", {}) or {}
    if base is None:
        lines = ["Photodose budget: DISABLED (no cap).", ""]
    else:
        lines = [f"Photodose budget: {base:.0f} ms base (scaled per role).", ""]
    try:
        from gently.harness.roles import REGISTRY as ROLE_REGISTRY
    except Exception:
        ROLE_REGISTRY = {}
    for eid in sorted(states):
        e = states[eid]
        used = getattr(e, "total_exposure_ms", 0.0) or 0.0
        role = getattr(e, "role", "test")
        if base is not None:
            rdef = ROLE_REGISTRY.get(role) if hasattr(ROLE_REGISTRY, "get") else None
            mult = getattr(rdef, "photodose_budget_multiplier", 1.0) if rdef else 1.0
            cap = base * mult
            pct = (used / cap * 100.0) if cap else 0.0
            flag = "  [PAUSED: over budget]" if eid in exceeded else ""
            lines.append(f"  {eid} ({role}): {used:.0f}/{cap:.0f} ms ({pct:.0f}%){flag}")
        else:
            lines.append(f"  {eid} ({role}): {used:.0f} ms used")
    if len(lines) == 2:
        lines.append("  (no embryos)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reactive monitoring modes (Phase 5) — high-level "install canonical
# detector → cadence + power reactive rules" entry points. Without one of
# these installed, embryos stay at base interval regardless of what
# detectors see.
# ---------------------------------------------------------------------------


@tool(
    name="enable_monitoring_mode",
    description=(
        "Install a named reactive monitoring mode on the running timelapse. "
        "The mode IS the reactive control — without one, embryos stay at base "
        "interval no matter what detectors see. WHEN TO USE: call once, "
        "immediately after start_adaptive_timelapse + assign_embryo_roles. "
        "Safe to re-call (modes are additive). "
        "Modes: "
        "'expression_monitoring' — for fluorescent-reporter onset experiments "
        "(GFP, mCherry, dat-1::GFP, dopaminergic, anything where signal turns on); "
        "installs WEAK+ -> 60s speedup + SATURATING -> 488 rampdown on test-role "
        "embryos. "
        "'pre_terminal_monitoring' — for hatching-timing / pre-hatch dynamics "
        "experiments; speeds up cadence on the organism's pre-terminal stage. "
        "'idle' — plain exploratory imaging with no reactive rules."
    ),
    category=ToolCategory.EXPERIMENT,
)
def enable_monitoring_mode(
    mode_name: str,
    tactic_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Install a named reactive monitoring mode on the orchestrator."""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    try:
        result = orchestrator.enable_monitoring_mode(mode_name)
        # Flip the matching plan tactic to active (guarded no-op when absent).
        if tactic_id:
            cs = getattr(agent, "context_store", None)
            session = getattr(agent, "session_id", None)
            if cs and session:
                cs.transition_tactic(session, tactic_id, "active")
        return result
    except Exception as e:
        return f"Error enabling monitoring mode '{mode_name}': {str(e)}"


@tool(
    name="add_test_onset_speedup",
    description=(
        "Manual / fine-grained version of the onset-speedup rule installed by "
        "enable_monitoring_mode('expression_monitoring'). Fires on the dopaminergic "
        "detector's 'lit_up' pseudo-stage (intensity_level >= WEAK) and switches "
        "matching embryos to `fast_interval` seconds. One-time per embryo. "
        "WHEN TO USE: only if you need a different fast_interval than the mode's "
        "default (60s), or you want to apply it to a specific embryo subset rather "
        "than all test-role embryos. Otherwise prefer enable_monitoring_mode."
    ),
    category=ToolCategory.EXPERIMENT,
)
def add_test_onset_speedup(
    fast_interval: float = 60.0,
    embryo_ids: list[str] | None = None,
    context: dict | None = None,
) -> str:
    """Install the canonical signal-onset cadence speedup rule."""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    try:
        orchestrator.add_test_onset_speedup(
            fast_interval=fast_interval,
            embryo_ids=embryo_ids,
        )
        target = ", ".join(embryo_ids) if embryo_ids else "all test-role embryos"
        return (
            f"Installed test-onset speedup: switch to {fast_interval}s interval "
            f"on signal onset for {target}."
        )
    except Exception as e:
        return f"Error installing test-onset speedup: {str(e)}"


@tool(
    name="add_test_saturation_rampdown",
    description=(
        "Manual / fine-grained version of the 488 rampdown rule installed by "
        "enable_monitoring_mode('expression_monitoring'). Sticky-monotonic downward "
        "ramp: when the dopaminergic detector reports SATURATING, drops 488 laser "
        "power by `step_pct` (never increases) until `floor_pct`. Re-fires each "
        "round signal saturates so the ramp can chase growing signal. "
        "WHEN TO USE: only if the mode's defaults (step 1.0%, floor 2.0%, "
        "ceiling 6.0%) don't fit. Otherwise prefer enable_monitoring_mode."
    ),
    category=ToolCategory.EXPERIMENT,
)
def add_test_saturation_rampdown(
    step_pct: float = 1.0,
    floor_pct: float = 2.0,
    ceiling_pct: float = 6.0,
    confirm_timepoints: int = 0,
    embryo_ids: list[str] | None = None,
    context: dict | None = None,
) -> str:
    """Install the canonical 488 saturation rampdown power rule."""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    try:
        orchestrator.add_test_saturation_rampdown(
            step_pct=step_pct,
            floor_pct=floor_pct,
            ceiling_pct=ceiling_pct,
            confirm_timepoints=confirm_timepoints,
            embryo_ids=embryo_ids,
        )
        target = ", ".join(embryo_ids) if embryo_ids else "all test-role embryos"
        return (
            f"Installed 488 saturation rampdown: step={step_pct}%, "
            f"floor={floor_pct}%, ceiling={ceiling_pct}%, "
            f"confirm_timepoints={confirm_timepoints} for {target}."
        )
    except Exception as e:
        return f"Error installing test-saturation rampdown: {str(e)}"


@tool(
    name="queue_burst",
    description=(
        "Queue a burst acquisition for one embryo. Captures `frames` rapid "
        "acquisitions at '1hz' (one frame per second, fits into multi-embryo "
        "cadence) or 'asap' (back-to-back, faster but fully exclusive). Produces "
        "a max-projection MP4. Other embryos pause while burst runs (their "
        "next_due_at keeps advancing so they catch up). "
        "WHEN TO USE: when something interesting just happened on one embryo and "
        "you want a high-frame-rate video — neuron firing, division, structural "
        "transition. Or for ground-truthing detector findings (force a burst to "
        "capture the moment the detector flagged GOOD structure). "
        "One-time per embryo by default; pass force=True to queue another."
    ),
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
def queue_burst(
    embryo_id: str,
    frames: int = 60,
    mode: str = "1hz",
    num_slices: int = 1,
    force: bool = False,
    tactic_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Queue an exclusive burst acquisition for one embryo."""
    agent, err = require_agent(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(agent)
    if err:
        return err

    try:
        result = orchestrator.queue_burst(
            embryo_id=embryo_id,
            frames=frames,
            mode=mode,
            num_slices=num_slices,
            force=force,
            tactic_id=tactic_id,
        )
        # Flip the matching plan tactic to active only when the burst was actually queued.
        # orchestrator.queue_burst returns "Burst queued for ..." on success and a
        # human-readable rejection sentence on soft-reject (embryo absent, already had
        # a burst, already has a queued burst).  Gate on the success prefix so a
        # soft-reject does NOT phantom-flip the tactic to active.
        if tactic_id and isinstance(result, str) and result.startswith("Burst queued for "):
            cs = getattr(agent, "context_store", None)
            session = getattr(agent, "session_id", None)
            if cs and session:
                cs.transition_tactic(session, tactic_id, "active")
        return result
    except Exception as e:
        return f"Error queueing burst for {embryo_id}: {str(e)}"
