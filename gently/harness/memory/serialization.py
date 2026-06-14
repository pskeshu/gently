"""
Context serialization — convert context to/from prompt format.

Provides utilities for formatting context for LLM prompts and
parsing context updates from LLM responses.
"""

import json
from typing import Any

from .model import (
    Campaign,
    Context,
    EmbryoUnderstanding,
    Expectation,
    ExpectationStatus,
    Learning,
    Observation,
    PlannedSession,
    PlannedSessionStatus,
    Project,
    Question,
    SessionIntent,
    Watchpoint,
)


def context_to_dict(context: Context) -> dict[str, Any]:
    """
    Serialize a Context to a dictionary.

    Useful for JSON export or API responses.
    """
    return {
        "intentions": {
            "campaigns": [_campaign_to_dict(c) for c in context.intentions.campaigns],
            "projects": [_project_to_dict(p) for p in context.intentions.projects],
            "planned_sessions": [
                _planned_session_to_dict(ps) for ps in context.intentions.planned_sessions
            ],
            "current_focus": context.intentions.current_focus,
            "session_intent": _session_intent_to_dict(context.intentions.session_intent)
            if context.intentions.session_intent
            else None,
        },
        "understanding": {
            "embryo_states": {
                eid: _embryo_to_dict(e) for eid, e in context.understanding.embryo_states.items()
            },
            "learnings": [
                _learning_to_dict(learning) for learning in context.understanding.learnings
            ],
        },
        "observations": [_observation_to_dict(o) for o in context.observations],
        "expectations": [_expectation_to_dict(e) for e in context.expectations],
        "attention": {
            "watchpoints": [_watchpoint_to_dict(w) for w in context.attention.watchpoints],
            "open_questions": [_question_to_dict(q) for q in context.attention.open_questions],
        },
    }


def context_to_json(context: Context, indent: int = 2) -> str:
    """Serialize a Context to JSON string."""
    return json.dumps(context_to_dict(context), indent=indent, default=str)


def context_summary(context: Context) -> str:
    """
    Generate a brief human-readable summary of the context.

    Useful for logging and debugging.
    """
    lines = []

    # Campaigns
    active_campaigns = [c for c in context.intentions.campaigns if c.status.value == "active"]
    if active_campaigns:
        lines.append(f"Campaigns: {len(active_campaigns)} active")
        for c in active_campaigns[:2]:
            progress = f" ({c.progress})" if c.progress else ""
            lines.append(f"  - {c.display_name}{progress}")

    # Planned sessions
    upcoming = [
        ps
        for ps in context.intentions.planned_sessions
        if ps.status == PlannedSessionStatus.PLANNED
    ]
    if upcoming:
        lines.append(f"Planned sessions: {len(upcoming)} upcoming")
        for ps in upcoming[:2]:
            when = ps.scheduled_date or "(unscheduled)"
            lines.append(f"  - {ps.display_title} [{when}]")

    # Focus
    if context.intentions.current_focus:
        lines.append(f"Focus: {context.intentions.current_focus}")

    # Embryos
    embryos = context.understanding.embryo_states
    if embryos:
        tracked = [e for e in embryos.values() if e.is_tracked]
        hatched = [e for e in embryos.values() if e.is_hatched]
        attention = [e for e in embryos.values() if e.needs_attention]
        lines.append(
            f"Embryos: {len(tracked)} tracked, {len(hatched)} hatched,"
            f" {len(attention)} need attention"
        )

    # Expectations
    pending = [e for e in context.expectations if e.status == ExpectationStatus.PENDING]
    if pending:
        lines.append(f"Expectations: {len(pending)} pending")

    # Watchpoints
    active_wp = [w for w in context.attention.watchpoints if w.status.value == "active"]
    if active_wp:
        lines.append(f"Watchpoints: {len(active_wp)} active")

    # Questions
    open_q = [
        q for q in context.attention.open_questions if q.status.value in ("open", "investigating")
    ]
    if open_q:
        lines.append(f"Questions: {len(open_q)} open")

    # Recent observations
    if context.observations:
        lines.append(f"Observations: {len(context.observations)} recent")

    return "\n".join(lines) if lines else "Empty context"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _campaign_to_dict(c: Campaign) -> dict[str, Any]:
    return {
        "id": c.id,
        "description": c.description,
        "shorthand": c.shorthand,
        "summary": c.summary,
        "target": c.target,
        "progress": c.progress,
        "parent_id": c.parent_id,
        "status": c.status.value,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
    }


def _project_to_dict(p: Project) -> dict[str, Any]:
    return {
        "id": p.id,
        "description": p.description,
        "campaign_id": p.campaign_id,
        "status": p.status.value,
        "created_at": p.created_at.isoformat(),
        "updated_at": p.updated_at.isoformat(),
    }


def _planned_session_to_dict(ps: PlannedSession) -> dict[str, Any]:
    return {
        "id": ps.id,
        "title": ps.title,
        "notes": ps.notes,
        "scheduled_date": ps.scheduled_date,
        "scheduled_time": ps.scheduled_time,
        "estimated_duration_minutes": ps.estimated_duration_minutes,
        "acquisition_params": ps.acquisition_params,
        "source_session_id": ps.source_session_id,
        "status": ps.status.value,
        "session_id": ps.session_id,
        "campaign_ids": ps.campaign_ids,
        "created_at": ps.created_at.isoformat(),
        "updated_at": ps.updated_at.isoformat(),
    }


def _session_intent_to_dict(s: SessionIntent) -> dict[str, Any]:
    return {
        "session_id": s.session_id,
        "planned_intent": s.planned_intent,
        "actual_summary": s.actual_summary,
        "campaign_ids": s.campaign_ids,
        "created_at": s.created_at.isoformat(),
        "completed_at": s.completed_at.isoformat() if s.completed_at else None,
    }


def _learning_to_dict(learning: Learning) -> dict[str, Any]:
    return {
        "id": learning.id,
        "content": learning.content,
        "confidence": learning.confidence.value,
        "basis": learning.basis,
        "created_at": learning.created_at.isoformat(),
    }


def _embryo_to_dict(e: EmbryoUnderstanding) -> dict[str, Any]:
    return {
        "embryo_id": e.embryo_id,
        "current_stage": e.current_stage,
        "stage_confidence": e.stage_confidence.value if e.stage_confidence else None,
        "health_assessment": e.health_assessment,
        "notes": e.notes,
        "last_observed": e.last_observed.isoformat() if e.last_observed else None,
        "is_tracked": e.is_tracked,
        "is_hatched": e.is_hatched,
        "needs_attention": e.needs_attention,
        "attention_reason": e.attention_reason,
    }


def _observation_to_dict(o: Observation) -> dict[str, Any]:
    return {
        "id": o.id,
        "timestamp": o.timestamp.isoformat(),
        "type": o.type,
        "content": o.content,
        "embryo_id": o.embryo_id,
        "significance": o.significance.value,
        "session_id": o.session_id,
        "gently_refs": o.gently_refs,
        "relates_to": o.relates_to,
    }


def _expectation_to_dict(e: Expectation) -> dict[str, Any]:
    return {
        "id": e.id,
        "target": e.target,
        "prediction": e.prediction,
        "expected_time": e.expected_time.isoformat(),
        "uncertainty": e.uncertainty,
        "basis": e.basis,
        "status": e.status.value,
        "created_at": e.created_at.isoformat(),
        "resolved_at": e.resolved_at.isoformat() if e.resolved_at else None,
    }


def _watchpoint_to_dict(w: Watchpoint) -> dict[str, Any]:
    return {
        "id": w.id,
        "target": w.target,
        "condition": w.condition,
        "priority": w.priority.value,
        "status": w.status.value,
        "created_at": w.created_at.isoformat(),
    }


def _question_to_dict(q: Question) -> dict[str, Any]:
    return {
        "id": q.id,
        "content": q.content,
        "status": q.status.value,
        "resolution": q.resolution,
        "created_at": q.created_at.isoformat(),
        "resolved_at": q.resolved_at.isoformat() if q.resolved_at else None,
    }
