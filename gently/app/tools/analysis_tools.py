"""
Analysis and VLM Tools

Tools for analyzing embryo images using Claude Vision.
"""

from gently.harness.tools.helpers import get_embryo_or_error, require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="analyze_volume",
    description="Analyze an embryo volume using Claude Vision API",
    category=ToolCategory.ANALYSIS,
)
async def analyze_volume(
    embryo_id: str,
    analysis_prompt: str,
    use_recent_context: bool = False,
    timepoint: int | None = None,
    context: dict | None = None,
) -> str:
    """Analyze embryo volume with Claude Vision"""
    agent, err = require_agent(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    try:
        result = await agent._analyze_with_vision(
            embryo_id=embryo.id,
            prompt=analysis_prompt,
            use_context=use_recent_context,
            timepoint=timepoint,
        )
        return result
    except Exception as e:
        return f"Error analyzing volume: {str(e)}"


@tool(
    name="get_recent_perceptions",
    description="""Get the latest perception state for one embryo or all embryos:
current developmental stage, how many consecutive observations it has held that stage
(stability), a possible-arrest signal, the recent stage trajectory, and the
perceiver's reasoning. Source: the LIVE perception loop (reads accumulated state,
does not trigger a fresh capture).
Use when the user asks "what stage is embryo X", "is anything stuck/arrested",
"how are the embryos developing", or before deciding whether to adapt acquisition.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("What stage is embryo_1 at?", {"embryo_id": "embryo_1"}),
        ToolExample("How is everything developing?", {}),
        ToolExample("Is anything arrested?", {}),
    ],
)
def get_recent_perceptions(
    embryo_id: str | None = None,
    n: int = 5,
    context: dict | None = None,
) -> str:
    """Read live per-embryo perception state from the perception sessions.

    All reads here (get_session / summary / attribute access) are synchronous and
    side-effect-free — they never trigger a VLM call.
    """
    agent, err = require_agent(context)
    if err:
        return err

    perceiver = getattr(agent, "perceiver", None)
    if perceiver is None:
        return "Perception system not available."

    def _one(eid: str) -> str:
        try:
            session = perceiver.get_session(eid)
        except Exception as e:
            return f"{eid}: perception read failed ({e})"
        if session is None or not getattr(session, "current_stage", None):
            return f"{eid}: no perceptions recorded yet"
        summary = session.summary()
        lines = [
            f"{eid}: stage={summary.get('current_stage')} "
            f"(stable for {summary.get('stability', 0)} obs, "
            f"{summary.get('observation_count', 0)} total)"
        ]
        seq = summary.get("stage_sequence") or []
        if seq:
            lines.append(f"  trajectory: {' -> '.join(seq)}")
        temporal = summary.get("temporal")  # TemporalContext dataclass or None
        if temporal is not None:
            tmin = getattr(temporal, "time_in_stage_min", 0.0)
            exp = getattr(temporal, "expected_duration_min", None)
            seg = f"  time in stage: {tmin:.0f} min"
            if exp:
                seg += (
                    f" (expected ~{exp:.0f} min, {getattr(temporal, 'overtime_ratio', 0.0):.1f}x)"
                )
            lines.append(seg)
            if getattr(temporal, "is_potentially_arrested", False):
                lines.append("  ** potentially ARRESTED **")
        observations = getattr(session, "observations", None) or []
        if observations and n > 0:
            recent = observations[-n:]
            lines.append(f"  recent observations (last {len(recent)}):")
            for o in recent:
                reason = (getattr(o, "reasoning", "") or "").strip().replace("\n", " ")
                if len(reason) > 160:
                    reason = reason[:159] + "…"
                lines.append(
                    f"    t{getattr(o, 'timepoint', '?')}: {getattr(o, 'stage', '?')} - {reason}"
                )
        return "\n".join(lines)

    if embryo_id:
        return _one(embryo_id)

    embryos = getattr(agent.experiment, "embryos", {}) or {}
    if not embryos:
        return "No embryos in the experiment."
    out = ["Perception state (all embryos):", ""]
    for eid in sorted(embryos):
        out.append(_one(eid))
        out.append("")
    return "\n".join(out).rstrip()


@tool(
    name="get_detection_summary",
    description="Get summary of all detections across all embryos",
    category=ToolCategory.DETECTION,
)
def get_detection_summary(context: dict) -> str:
    """Get detection summary"""
    agent, err = require_agent(context)
    if err:
        return err

    lines = ["Detection Summary:", ""]

    for embryo_id, embryo in agent.experiment.embryos.items():
        if embryo.detection_results:
            lines.append(f"* {embryo_id}:")
            for det_name, results in embryo.detection_results.items():
                if results:
                    latest = results[-1]
                    lines.append(
                        f"  - {det_name}: {latest.get('detected', False)}"
                        f" at t={latest.get('timepoint', '?')}"
                    )
            lines.append("")

    if len(lines) == 2:
        return "No detections recorded yet."

    return "\n".join(lines)
