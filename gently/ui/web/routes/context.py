"""Context (shared-visibility) routes.

Exposes the agent's "mind" — its open questions (uncertainty), active
watchpoints (attention), and pending expectations (beliefs) — read by anyone,
resolvable only by the control holder. Live updates ride the CONTEXT_UPDATED
event the FileContextStore emits on the global bus, which the server already
broadcasts to /ws; the client just re-fetches /api/context on it (no polling).
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from gently.ui.web.auth import require_control

from .campaigns import _serialize


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _store():
        # Defensive: the store is wired after construction; tolerate cold start.
        return getattr(server, "context_store", None)

    def _members(project_id: str) -> set[str]:
        store = getattr(server, "gently_store", None)
        if store is None:
            return set()
        try:
            return {s.get("session_id") for s in store.list_sessions(project_id=project_id)}
        except Exception:
            return set()

    @router.get("/api/context")
    async def get_context(project_id: str | None = None):
        """The agent's mind; with ``project_id`` only what belongs to that project."""
        cs = _store()
        empty = {"available": False, "expectations": [], "watchpoints": [], "questions": []}
        if cs is None:
            return empty
        try:
            if project_id:
                members = _members(project_id)
                questions = [
                    q
                    for q in cs.list_questions(project_id=project_id, session_ids=members)
                    if q.status.value in ("open", "investigating")
                ]
                expectations = [
                    e
                    for e in cs.list_expectations(project_id=project_id, session_ids=members)
                    if e.status.value == "pending"
                ]
            else:
                questions = cs.get_open_questions()
                expectations = cs.get_pending_expectations()
            return {
                "available": True,
                "questions": [_serialize(q) for q in questions],
                "watchpoints": [_serialize(w) for w in cs.get_active_watchpoints()],
                "expectations": [_serialize(e) for e in expectations],
            }
        except Exception:
            return empty

    @router.post("/api/context/expectations", dependencies=[Depends(require_control)])
    async def add_expectation(request: Request):
        """A person's expectation, for a project or the live session: target,
        prediction, expected_time (ISO), uncertainty, basis."""
        body = await request.json()
        cs = _store()
        if cs is None:
            raise HTTPException(status_code=503, detail="context store unavailable")
        from gently.harness.memory.model import Expectation

        prediction = (body.get("prediction") or "").strip()
        if not prediction:
            raise HTTPException(status_code=400, detail="prediction required")
        when = body.get("expected_time")
        try:
            expected = datetime.fromisoformat(when) if when else datetime.now()
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="expected_time must be ISO 8601") from None
        exp = Expectation(
            id=str(uuid.uuid4())[:8],
            target=(body.get("target") or "project").strip(),
            prediction=prediction,
            expected_time=expected,
            uncertainty=body.get("uncertainty") or None,
            basis=body.get("basis") or None,
            session_id=body.get("session_id") or None,
            project_id=body.get("project_id") or None,
        )
        cs.add_expectation(exp)
        return {"ok": True, "expectation": _serialize(exp)}

    @router.post("/api/context/questions", dependencies=[Depends(require_control)])
    async def add_question(request: Request):
        body = await request.json()
        cs = _store()
        if cs is None:
            raise HTTPException(status_code=503, detail="context store unavailable")
        from gently.harness.memory.model import Question

        content = (body.get("content") or "").strip()
        if not content:
            raise HTTPException(status_code=400, detail="content required")
        q = Question(
            id=str(uuid.uuid4())[:8],
            content=content,
            session_id=body.get("session_id") or None,
            project_id=body.get("project_id") or None,
        )
        cs.add_question(q)
        return {"ok": True, "question": _serialize(q)}

    @router.post("/api/context/questions/{q_id}/resolve", dependencies=[Depends(require_control)])
    async def resolve_question(q_id: str, resolution: str = Body("", embed=True)):
        cs = _store()
        if cs is None:
            return {"ok": False, "error": "context store unavailable"}
        cs.resolve_question(q_id, resolution or "")
        return {"ok": True}

    @router.post(
        "/api/context/watchpoints/{wp_id}/resolve", dependencies=[Depends(require_control)]
    )
    async def resolve_watchpoint(wp_id: str):
        cs = _store()
        if cs is None:
            return {"ok": False, "error": "context store unavailable"}
        cs.resolve_watchpoint(wp_id)
        return {"ok": True}

    @router.post(
        "/api/context/expectations/{exp_id}/resolve", dependencies=[Depends(require_control)]
    )
    async def resolve_expectation(exp_id: str, status: str = Body("confirmed", embed=True)):
        cs = _store()
        if cs is None:
            return {"ok": False, "error": "context store unavailable"}
        from gently.harness.memory.model import ExpectationStatus

        try:
            st = ExpectationStatus(status)
        except ValueError:
            st = ExpectationStatus.CONFIRMED
        cs.resolve_expectation(exp_id, st)
        return {"ok": True}

    return router
