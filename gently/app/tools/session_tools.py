"""
Session and Interaction Tools

Tools for session statistics, interaction logging, and experiment comparison.
"""

from gently.harness.tools.helpers import (
    get_embryo_or_error,
    require_agent,
    require_interaction_logger,
)
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="assess_image_quality",
    description=(
        "Assess image quality metrics (focus, brightness, noise) and suggest parameter adjustments"
    ),
    category=ToolCategory.ANALYSIS,
)
async def assess_image_quality(
    embryo_id: str | None = None, suggest_parameters: bool = True, context: dict | None = None
) -> str:
    """Assess image quality and suggest improvements"""
    agent, err = require_agent(context)
    if err:
        return err

    if embryo_id:
        embryo, err = get_embryo_or_error(agent, embryo_id)
        if err:
            return err
        if not embryo.recent_images:
            return f"No images available for {embryo_id}"
        latest = embryo.recent_images[-1]
        image_b64 = latest.max_projection_b64
        source = f"{embryo_id} (timepoint {latest.timepoint})"
    else:
        return "Please specify an embryo_id"

    quality_prompt = """Assess the image quality of this microscopy image.

Evaluate the following aspects (rate each as GOOD, ACCEPTABLE, or POOR):

1. FOCUS: Is the specimen in sharp focus? Look for:
   - Clear cell boundaries
   - Sharp edges
   - No blur or haze

2. BRIGHTNESS: Is the illumination appropriate?
   - Too dim: important features not visible
   - Too bright: saturation/clipping
   - Good: full dynamic range used

3. CONTRAST: Can you distinguish structures?
   - Cell membranes visible?
   - Internal structures visible?

4. NOISE: Is the image noisy?
   - Grainy appearance?
   - Speckles or artifacts?

5. FIELD OF VIEW: Is the embryo well-positioned?
   - Centered in frame?
   - Full embryo visible?

Respond in this format:
FOCUS: [GOOD/ACCEPTABLE/POOR] - [brief reason]
BRIGHTNESS: [GOOD/ACCEPTABLE/POOR] - [brief reason]
CONTRAST: [GOOD/ACCEPTABLE/POOR] - [brief reason]
NOISE: [GOOD/ACCEPTABLE/POOR] - [brief reason]
FIELD_OF_VIEW: [GOOD/ACCEPTABLE/POOR] - [brief reason]
OVERALL: [GOOD/ACCEPTABLE/POOR]

SUGGESTIONS: [List specific parameter adjustments if any aspect is POOR]"""

    try:
        response = agent.claude.messages.create(
            model=agent.model,
            max_tokens=800,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": quality_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
        )

        assessment = response.content[0].text

        lines = [
            f"Image Quality Assessment for {source}:",
            "",
            assessment,
        ]

        if suggest_parameters:
            suggestions = []
            assessment_lower = assessment.lower()

            if "focus" in assessment_lower and "poor" in assessment_lower:
                suggestions.append("- Consider re-running piezo-galvo calibration")
                suggestions.append("- Check if embryo has moved out of focus plane")

            if "brightness" in assessment_lower and "poor" in assessment_lower:
                if "dim" in assessment_lower or "dark" in assessment_lower:
                    suggestions.append("- Increase exposure_ms (current -> +50%)")
                    suggestions.append("- Check laser power settings")
                elif "bright" in assessment_lower or "saturated" in assessment_lower:
                    suggestions.append("- Decrease exposure_ms (current -> -30%)")

            if "noise" in assessment_lower and "poor" in assessment_lower:
                suggestions.append("- Increase exposure_ms to improve signal-to-noise")
                suggestions.append("- Consider averaging multiple frames")

            if suggestions:
                lines.append("")
                lines.append("Parameter Suggestions:")
                lines.extend(suggestions)

        return "\n".join(lines)

    except Exception as e:
        return f"Error assessing quality: {str(e)}"


@tool(
    name="get_session_stats",
    description=(
        "Get statistics for the current agent session including interactions,"
        " corrections, and tool usage"
    ),
    category=ToolCategory.DATA,
)
def get_session_stats(context: dict | None = None) -> str:
    """Get session statistics from interaction logger"""
    agent, err = require_agent(context)
    if err:
        return err

    logger, err = require_interaction_logger(agent)
    if err:
        return err

    stats = logger.get_session_stats()

    lines = [
        "Session Statistics:",
        f"  Total interactions: {stats['total_interactions']}",
        f"  Tool calls: {stats['tool_calls']}",
        f"  Corrections detected: {stats['corrections']}",
        f"  Errors: {stats['errors']}",
        f"  Correction rate: {stats['correction_rate']:.1%}",
    ]

    return "\n".join(lines)


@tool(
    name="compare_embryo_development",
    description="Compare developmental progress across multiple embryos in the current experiment",
    category=ToolCategory.ANALYSIS,
)
def compare_embryo_development(
    embryo_ids: list[str] | None = None, context: dict | None = None
) -> str:
    """Compare embryo development"""
    agent, err = require_agent(context)
    if err:
        return err

    if embryo_ids:
        embryos = [agent.experiment.get_embryo_by_any_name(eid) for eid in embryo_ids]
        embryos = [e for e in embryos if e is not None]
    else:
        embryos = list(agent.experiment.embryos.values())

    if not embryos:
        return "No embryos found."

    lines = ["Embryo Development Comparison:", ""]

    lines.append(
        f"{'Embryo':<15} {'Timepoints':<12} {'Stage':<15} {'Hatching Est.':<15} {'Status'}"
    )
    lines.append("-" * 70)

    for embryo in embryos:
        stage = "unknown"
        hatching_est = "N/A"

        if hasattr(agent, "developmental_tracker") and agent.developmental_tracker:
            current = agent.developmental_tracker.get_current_stage(embryo.id)
            if current:
                stage = current.stage.value
                if current.predicted_minutes_to_hatching:
                    hours = current.predicted_minutes_to_hatching / 60
                    hatching_est = f"~{hours:.1f}h"

        if embryo.should_skip:
            status = f"skipped ({embryo.skip_reason})"
        elif embryo.hatching_status and embryo.hatching_status.get("detected"):
            status = "HATCHED"
        else:
            status = "active"

        lines.append(
            f"{embryo.id:<15} {embryo.timepoints_acquired:<12} {stage:<15}"
            f" {hatching_est:<15} {status}"
        )

    active = sum(1 for e in embryos if not e.should_skip)
    hatched = sum(1 for e in embryos if e.hatching_status and e.hatching_status.get("detected"))

    lines.append("")
    lines.append(f"Summary: {active} active, {hatched} hatched, {len(embryos) - active} skipped")

    return "\n".join(lines)


@tool(
    name="analyze_corrections",
    description=(
        "Analyze user corrections from interaction logs to identify patterns in agent mistakes"
    ),
    category=ToolCategory.DATA,
)
def analyze_corrections(limit: int = 50, context: dict | None = None) -> str:
    """Analyze correction patterns"""
    agent, err = require_agent(context)
    if err:
        return err

    logger, err = require_interaction_logger(agent)
    if err:
        return err

    interactions = logger.load_session_interactions()

    if not interactions:
        return "No interactions recorded yet."

    corrections = [i for i in interactions if i.was_corrected]

    if not corrections:
        return f"No corrections detected in {len(interactions)} interactions."

    lines = [
        f"Correction Analysis ({len(corrections)} corrections in"
        f" {len(interactions)} interactions):",
        "",
    ]

    indicator_counts: dict[str, int] = {}
    tool_corrections: dict[str, int] = {}

    for corr in corrections[:limit]:
        for indicator in corr.correction_indicators:
            indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

        for tc in corr.tool_calls:
            tool_corrections[tc.tool_name] = tool_corrections.get(tc.tool_name, 0) + 1

    lines.append("Common correction indicators:")
    for indicator, count in sorted(indicator_counts.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"  '{indicator}': {count} times")

    lines.append("")
    lines.append("Tools frequently followed by corrections:")
    for tool_name, count in sorted(tool_corrections.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"  {tool_name}: {count} times")

    lines.append("")
    lines.append("Recent correction examples:")
    for corr in corrections[-3:]:
        lines.append(f"  - '{corr.user_prompt[:50]}...'")
        if corr.correction_prompt:
            lines.append(f"    Correction: '{corr.correction_prompt[:50]}...'")

    return "\n".join(lines)


@tool(
    name="export_interaction_log",
    description="Export interaction logs for external analysis",
    category=ToolCategory.DATA,
)
def export_interaction_log(format: str = "summary", context: dict | None = None) -> str:
    """Export interaction log"""
    agent, err = require_agent(context)
    if err:
        return err

    logger, err = require_interaction_logger(agent)
    if err:
        return err

    if format == "jsonl_path":
        return f"Interaction log file: {logger.log_file}"

    interactions = logger.load_session_interactions()

    if not interactions:
        return "No interactions recorded."

    if format == "summary":
        stats = logger.get_session_stats()
        lines = [
            f"Session: {logger.session_id}",
            f"Log file: {logger.log_file}",
            f"Total interactions: {stats['total_interactions']}",
            f"Tool calls: {stats['tool_calls']}",
            f"Corrections: {stats['corrections']} ({stats['correction_rate']:.1%})",
            f"Errors: {stats['errors']}",
        ]
        return "\n".join(lines)

    elif format == "detailed":
        lines = [f"Detailed Log ({len(interactions)} interactions):", ""]

        for i, inter in enumerate(interactions[-20:], 1):
            lines.append(f"{i}. [{inter.timestamp.strftime('%H:%M:%S')}]")
            lines.append(f"   User: {inter.user_prompt[:60]}...")
            lines.append(f"   Tools: {[tc.tool_name for tc in inter.tool_calls]}")
            if inter.was_corrected:
                lines.append(f"   CORRECTED: {inter.correction_indicators}")
            lines.append("")

        return "\n".join(lines)

    return f"Unknown format: {format}. Use 'summary', 'detailed', or 'jsonl_path'"


@tool(
    name="import_embryos_from_session",
    description="""Import embryos (positions, calibration, settings) from another session into
the current experiment.
Use when user wants to start a fresh session but keep embryo positions from a previous
session (e.g., "import embryos from last session", "load embryos from session X").
This imports positions and calibration data but NOT conversation history or detection
results - it's a fresh start with known embryos.
Use list_sessions or /sessions first to find the session_id to import from.""",
    category=ToolCategory.DATA,
    examples=[
        ToolExample("Import embryos from session abc123", {"session_id": "abc123"}),
        ToolExample(
            "Load embryos from previous session, replacing current ones",
            {"session_id": "abc123", "clear_existing": True},
        ),
    ],
)
def import_embryos_from_session(
    session_id: str, clear_existing: bool = False, context: dict | None = None
) -> str:
    """
    Import embryos from another session.

    Parameters
    ----------
    session_id : str
        Session ID to import embryos from (use list_sessions to find IDs)
    clear_existing : bool
        If True, replace all current embryos. If False, add to existing (skip duplicates).
    context : dict
        Execution context
    """
    agent, err = require_agent(context)
    if err:
        return err

    result = agent.import_embryos_from_session(session_id=session_id, clear_existing=clear_existing)

    if not result.get("success"):
        return f"Import failed: {result.get('error', 'Unknown error')}"

    lines = [
        f"✓ Imported embryos from session {session_id}",
        f"  Imported: {len(result['imported'])} embryo(s)",
    ]

    if result["imported"]:
        lines.append(f"    {', '.join(result['imported'])}")

    if result["skipped"]:
        lines.append(f"  Skipped (already exist): {len(result['skipped'])}")
        lines.append(f"    {', '.join(result['skipped'])}")

    if result.get("errors"):
        lines.append(f"  Errors: {len(result['errors'])}")
        for err in result["errors"]:
            lines.append(f"    - {err}")

    return "\n".join(lines)


@tool(
    name="list_sessions",
    description="""List available sessions with their IDs, embryo counts, message counts,
and last active times.
Use when user asks "show sessions", "what sessions exist", or needs to pick a session to
resume or import from. Returns ALL sessions — do NOT filter by embryo count. Sessions are
valuable for conversation history too.""",
    category=ToolCategory.DATA,
    examples=[
        ToolExample("Show available sessions", {}),
        ToolExample("List recent sessions", {"limit": 5}),
    ],
)
def list_sessions(limit: int = 20, context: dict | None = None) -> str:
    """
    List available sessions.

    Parameters
    ----------
    limit : int
        Maximum number of sessions to show (default: 20)
    context : dict
        Execution context
    """
    agent, err = require_agent(context)
    if err:
        return err

    all_sessions = agent.list_sessions()

    if not all_sessions:
        return "No sessions found."

    sessions = all_sessions[:limit]
    total = len(all_sessions)

    lines = [f"Available Sessions ({total} total):", ""]
    lines.append(f"{'ID':<40} {'Embryos':<8} {'Messages':<10} {'Last Active'}")
    lines.append("-" * 80)

    for s in sessions:
        session_id = s.get("session_id", "unknown")[:38]
        embryo_count = s.get("embryo_count", 0)
        msg_count = s.get("message_count", 0)
        last_active = s.get("last_active", "")
        if last_active:
            # Format datetime string
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(last_active.replace("Z", "+00:00"))
                last_active = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                last_active = last_active[:16]

        lines.append(f"{session_id:<40} {embryo_count:<8} {msg_count:<10} {last_active}")

    if total > limit:
        lines.append(f"  ... and {total - limit} more (use limit= to see more)")

    lines.append("")
    lines.append("To resume a session (full history + state): /resume <session_id>")
    lines.append(
        "To import only embryo positions into current session:"
        " import_embryos_from_session(session_id)"
    )

    return "\n".join(lines)
