"""Context (shared-visibility) routes.

Exposes the agent's "mind" — its open questions (uncertainty), active
watchpoints (attention), and pending expectations (beliefs) — read by anyone,
resolvable only by the control holder. Live updates ride the CONTEXT_UPDATED
event the FileContextStore emits on the global bus, which the server already
broadcasts to /ws; the client just re-fetches /api/context on it (no polling).
"""

from fastapi import APIRouter, Body, Depends

from gently.ui.web.auth import require_control

from .campaigns import _serialize


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _store():
        # Defensive: the store is wired after construction; tolerate cold start.
        return getattr(server, "context_store", None)

    @router.get("/api/context")
    async def get_context():
        cs = _store()
        empty = {"available": False, "expectations": [], "watchpoints": [], "questions": []}
        if cs is None:
            return empty
        try:
            return {
                "available": True,
                "questions": [_serialize(q) for q in cs.get_open_questions()],
                "watchpoints": [_serialize(w) for w in cs.get_active_watchpoints()],
                "expectations": [_serialize(e) for e in cs.get_pending_expectations()],
            }
        except Exception:
            return empty

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
