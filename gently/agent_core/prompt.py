"""
Prompt construction for the agent.

Builds system prompts and context for different thinking modes.
Includes escalation context, trigger data, and expectation status.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from ..context import Context, Observation, Expectation, Watchpoint
from .types import ThinkTrigger, ThinkingMode


# ---------------------------------------------------------------------------
# Agent Identity
# ---------------------------------------------------------------------------

AGENT_IDENTITY = """\
You are the reasoning core of a microscopy research assistant. You help a biologist
image C. elegans embryos through development.

You think continuously - not just when asked. You hold the researcher's intentions,
build understanding from observations, make predictions, and notice when things
need attention.

## Your capabilities

You can:
- OBSERVE: Look at embryos, check their state
- ACT: Move the stage, acquire images, configure the microscope
- PERCEIVE: Classify developmental stages, detect features
- SPEAK: Say something to the researcher (if present)
- ASK: Ask the researcher a question
- NOTIFY: Send a notification (if away)
- UPDATE_CONTEXT: Update your understanding, add observations, set expectations

## How you think

Each cycle, you:
1. Consider current context and what triggered this thought
2. Decide what matters right now
3. Take actions and/or update your understanding

You don't need to act every cycle. Sometimes just noting an observation is enough.
"""


# ---------------------------------------------------------------------------
# Response Format
# ---------------------------------------------------------------------------

RESPONSE_FORMAT = """\
## Response Format

<reasoning>
Your thinking process
</reasoning>

<observations>
- observation: "what you noticed"
  significance: high|medium|low
  relates_to: [goals, embryos]
</observations>

<actions>
- action: observe|image|move|speak|ask|notify
  params: {...}
  reason: "why"
</actions>

<context_updates>
- update: expectation|watchpoint|learning|understanding
  content: {...}
</context_updates>
"""


# ---------------------------------------------------------------------------
# Formatting Functions
# ---------------------------------------------------------------------------

def format_context(context: Context) -> str:
    """Format context for the prompt."""
    sections = []

    # Intentions
    sections.append("## Current Context")
    sections.append("")
    sections.append("### Intentions")

    if context.active_campaigns:
        sections.append("**Active Campaigns:**")
        for c in context.active_campaigns:
            progress = f" ({c.progress})" if c.progress else ""
            sections.append(f"- {c.description}{progress}")
    else:
        sections.append("No active campaigns.")

    if context.intentions.current_focus:
        sections.append(f"\n**Current Focus:** {context.intentions.current_focus}")

    if context.intentions.session_intent:
        intent = context.intentions.session_intent
        sections.append(f"\n**Session Intent:** {intent.planned_intent or 'Not specified'}")

    # Understanding
    sections.append("\n### Understanding")

    if context.understanding.embryo_states:
        sections.append("**Embryo States:**")
        for eid, state in context.understanding.embryo_states.items():
            stage = state.current_stage or "unknown"
            confidence = f" ({state.stage_confidence.value})" if state.stage_confidence else ""
            flags = []
            if state.is_hatched:
                flags.append("hatched")
            if state.needs_attention:
                flags.append(f"needs attention: {state.attention_reason}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            sections.append(f"- {eid}: {stage}{confidence}{flag_str}")
    else:
        sections.append("No embryo states tracked.")

    if context.understanding.learnings:
        sections.append("\n**Learnings:**")
        for learning in context.understanding.learnings[:5]:  # Limit to 5
            sections.append(f"- {learning.content} ({learning.confidence.value})")

    # Expectations
    if context.pending_expectations:
        sections.append("\n### Expectations")
        now = datetime.now()
        for exp in context.pending_expectations[:5]:  # Limit to 5
            time_str = exp.expected_time.strftime("%H:%M")
            delta = exp.expected_time - now
            minutes_until = delta.total_seconds() / 60
            if minutes_until > 0:
                urgency = f"in {minutes_until:.0f}min"
            else:
                urgency = f"OVERDUE by {abs(minutes_until):.0f}min"
            sections.append(f"- {exp.target}: {exp.prediction} (by {time_str}, {urgency})")

    # Attention
    if context.active_watchpoints:
        sections.append("\n### Watching")
        for wp in context.active_watchpoints:
            sections.append(f"- {wp.target}: {wp.condition} ({wp.priority.value})")

    if context.attention.open_questions:
        sections.append("\n### Open Questions")
        for q in context.attention.open_questions[:3]:  # Limit to 3
            sections.append(f"- {q.content}")

    # Recent observations
    if context.observations:
        sections.append("\n### Recent Observations")
        for obs in context.observations[-5:]:  # Last 5
            time_str = obs.timestamp.strftime("%H:%M:%S")
            sections.append(f"- [{time_str}] {obs.content}")

    return "\n".join(sections)


def format_world_state(world) -> str:
    """Format world state for the prompt."""
    sections = []

    sections.append("## World State")
    sections.append(f"**Time:** {world.current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append(f"**User Present:** {'Yes' if world.user_present else 'No'}")

    if world.session_id:
        sections.append(f"**Session:** {world.session_id}")

    if world.microscope_status:
        sections.append(f"**Microscope:** {world.microscope_status.get('status', 'unknown')}")

    if hasattr(world, 'context_richness'):
        sections.append(f"**Context Richness:** {world.context_richness:.2f}")

    if world.recent_events:
        sections.append("\n**Recent Events:**")
        for event in world.recent_events[-5:]:  # Last 5
            time_str = event.timestamp.strftime("%H:%M:%S")
            data_summary = ""
            if event.data:
                # Include key data fields concisely
                key_fields = {k: v for k, v in event.data.items()
                              if k in ("embryo_id", "stage", "status", "message")}
                if key_fields:
                    data_summary = f" ({', '.join(f'{k}={v}' for k, v in key_fields.items())})"
            sections.append(f"- [{time_str}] {event.event_type.name}{data_summary}")

    return "\n".join(sections)


def format_trigger_context(trigger: ThinkTrigger, trigger_data: Optional[Dict] = None) -> str:
    """Format trigger-specific context for the prompt."""
    if not trigger_data:
        return ""

    sections = []
    sections.append("\n## Trigger Details")

    if trigger == ThinkTrigger.SURPRISE:
        sections.append("**Surprise detected!**")
        if "expected" in trigger_data and "actual" in trigger_data:
            sections.append(f"- Expected: {trigger_data['expected']}")
            sections.append(f"- Actual: {trigger_data['actual']}")
        if "embryo_id" in trigger_data:
            sections.append(f"- Embryo: {trigger_data['embryo_id']}")

    elif trigger == ThinkTrigger.ESCALATION:
        sections.append("**Escalated from quick scan.**")
        if "reason" in trigger_data:
            sections.append(f"- Reason: {trigger_data['reason']}")
        if "context" in trigger_data:
            sections.append(f"- Context: {trigger_data['context']}")
        sections.append("\nThe quick scan flagged something that needs deeper thinking.")
        sections.append("Please investigate more thoroughly than a routine check.")

    elif trigger == ThinkTrigger.EXPECTATION:
        exp_type = trigger_data.get("type", "unknown")
        if exp_type == "approaching":
            sections.append("**Expectation approaching deadline.**")
            sections.append(f"- Target: {trigger_data.get('target', '?')}")
            sections.append(f"- Prediction: {trigger_data.get('prediction', '?')}")
            minutes = trigger_data.get("time_until_minutes", 0)
            sections.append(f"- Time until: {minutes:.0f} minutes")
            sections.append("\nCheck if this is on track. Should we adjust the expectation?")
        elif exp_type == "expired":
            sections.append("**Expectation expired without resolution!**")
            sections.append(f"- Target: {trigger_data.get('target', '?')}")
            sections.append(f"- Prediction: {trigger_data.get('prediction', '?')}")
            minutes = trigger_data.get("overdue_minutes", 0)
            sections.append(f"- Overdue by: {minutes:.0f} minutes")
            sections.append("\nInvestigate: Was the prediction wrong? Did we miss something?")

    elif trigger == ThinkTrigger.WATCHPOINT:
        sections.append("**Watchpoint triggered.**")
        for key in ("embryo_id", "condition", "feature", "message"):
            if key in trigger_data:
                sections.append(f"- {key}: {trigger_data[key]}")

    elif trigger == ThinkTrigger.USER:
        sections.append("**User interaction.**")
        if "message" in trigger_data:
            sections.append(f'- User said: "{trigger_data["message"]}"')

    return "\n".join(sections)


def format_ask(trigger: ThinkTrigger, mode: ThinkingMode, trigger_data: Optional[Dict] = None) -> str:
    """Format the thinking prompt based on trigger and mode."""
    trigger_context = format_trigger_context(trigger, trigger_data)

    if mode == ThinkingMode.FAST:
        return f"""\
## Think (quick scan)

Trigger: {trigger.value}
{trigger_context}

Quick check: Anything need immediate attention?
- Scan expectations and watchpoints
- Note anything surprising
- Take action only if urgent
- If something needs deeper investigation, say so in your reasoning

Keep it brief.
"""

    elif mode == ThinkingMode.MODERATE:
        return f"""\
## Think

Trigger: {trigger.value}
{trigger_context}

What matters right now?
- Check if any expectations are confirmed or surprised
- Look for patterns in recent observations
- Consider what to watch for next
- Take actions if needed
- Update context as appropriate

Provide clear reasoning.
"""

    else:  # DEEP
        return f"""\
## Think Deeply

Trigger: {trigger.value}
{trigger_context}

This is a good moment to step back and think carefully.

Consider:
- How do recent events relate to our goals and campaigns?
- Are there patterns emerging that we should formalize as learnings?
- What should we expect to happen next, and when?
- Are there questions we should be investigating?
- Should we adjust our focus or watchpoints?

Provide thorough reasoning. It's OK to take your time and think carefully.
"""


# ---------------------------------------------------------------------------
# Main Prompt Builder
# ---------------------------------------------------------------------------

def build_agent_prompt(
    context: Context,
    world,
    trigger: ThinkTrigger,
    mode: ThinkingMode,
    trigger_data: Optional[Dict] = None,
) -> str:
    """
    Build the full prompt for the agent.

    Parameters
    ----------
    context : Context
        Agent's current context
    world : WorldState
        Current world state
    trigger : ThinkTrigger
        What triggered this think
    mode : ThinkingMode
        Thinking depth
    trigger_data : dict, optional
        Additional data about the trigger

    Returns
    -------
    str
        Complete prompt
    """
    parts = [
        AGENT_IDENTITY,
        format_context(context),
        format_world_state(world),
        format_ask(trigger, mode, trigger_data),
        RESPONSE_FORMAT,
    ]
    return "\n\n---\n\n".join(parts)


def build_system_prompt() -> str:
    """Build the system prompt (identity only)."""
    return AGENT_IDENTITY
