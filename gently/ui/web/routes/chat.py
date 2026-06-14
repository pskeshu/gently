"""Per-timepoint VLM chat routes.

Lets the biologist continue the perception assessment as a conversation,
grounded in the same projection image and the original VLM reasoning.

Storage: ``embryos/{embryo_id}/traces/t{NNNN}_chat.jsonl`` — one JSON
object per line, ``{"role": "user"|"assistant", "ts": ..., "content": ...}``.
The seed (projection image + original assessment) is reconstructed on
every turn from the existing trace, so the chat file only stores real
user/assistant exchanges.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gently.settings import settings
from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)

# Per-timepoint VLM chat → perception tier (Opus 4.8); centralized, not hardcoded.
CHAT_MODEL = settings.models.perception
SYSTEM_PROMPT = (
    "You are helping a biologist interpret a microscopy perception "
    "assessment of a C. elegans embryo at a specific timepoint. You can "
    "see the three-view projection (XY top-left, YZ top-right, XZ "
    "bottom-left) the original assessment was based on. The original VLM "
    "assessment appears as your first assistant turn — refer back to it "
    "when relevant, but treat the biologist's follow-up questions as "
    "primary.\n\n"
    "Be precise and grounded in what is visible. If a question can't be "
    "answered from the image alone (e.g. needs fluorescence quantification, "
    "cell tracking, or external data), say so plainly rather than "
    "speculating. Keep replies concise — biologists are looking at live "
    "data and want fast, clear answers."
)


class ChatRequest(BaseModel):
    message: str


def _resolve_session_dir(server, sid: str) -> Path | None:
    store = getattr(server, "gently_store", None)
    if store is None:
        return None
    return store._session_dir(sid)


def _trace_path(server, sid: str, eid: str, tp: int) -> Path | None:
    sd = _resolve_session_dir(server, sid)
    if sd is None:
        return None
    return sd / "embryos" / eid / "traces" / f"t{tp:04d}.json"


def _chat_path(server, sid: str, eid: str, tp: int) -> Path | None:
    sd = _resolve_session_dir(server, sid)
    if sd is None:
        return None
    return sd / "embryos" / eid / "traces" / f"t{tp:04d}_chat.jsonl"


def _load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    turns: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed chat line in %s", path)
    return turns


def _append_turn(path: Path, role: str, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    turn = {
        "role": role,
        "ts": datetime.now().isoformat(),
        "content": content,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(turn, ensure_ascii=False) + "\n")


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/perception/chat/{sid}/{eid}/{tp}")
    async def get_chat(sid: str, eid: str, tp: int):
        """Return the persisted chat history for (session, embryo, timepoint).

        Empty list if no chat has been started yet.
        """
        path = _chat_path(server, sid, eid, tp)
        if path is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"turns": _load_history(path)}

    @router.post("/api/perception/chat/{sid}/{eid}/{tp}")
    async def post_chat(
        sid: str,
        eid: str,
        tp: int,
        body: ChatRequest,
        _control=Depends(require_control),  # noqa: B008
    ):
        """Append a user message and stream the assistant reply as SSE.

        Each SSE event is JSON: ``{"type": "delta", "text": "..."}`` for
        text deltas, ``{"type": "done"}`` when the response completes, or
        ``{"type": "error", "message": "..."}`` on failure.
        """
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="Empty message")

        trace_path = _trace_path(server, sid, eid, tp)
        chat_path = _chat_path(server, sid, eid, tp)
        if trace_path is None or chat_path is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if not trace_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No perception trace for T{tp}",
            )

        with open(trace_path, encoding="utf-8") as f:
            trace = json.load(f)
        stage = trace.get("predicted_stage", "unknown")
        reasoning = trace.get("reasoning", "")

        store = server.gently_store
        proj_b64 = store.get_projection_b64(sid, eid, tp)
        if not proj_b64:
            raise HTTPException(
                status_code=404,
                detail=f"No projection for T{tp}",
            )

        history = _load_history(chat_path)

        seed_user = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": proj_b64,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Please give a perception assessment of this "
                        f"C. elegans embryo at timepoint T{tp}, including "
                        f"the developmental stage and your reasoning."
                    ),
                },
            ],
        }
        seed_assistant = {
            "role": "assistant",
            "content": [{"type": "text", "text": f"Stage: {stage}\n\n{reasoning}"}],
        }

        messages: list[dict] = [seed_user, seed_assistant]
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": body.message}],
            }
        )

        # Persist the user's turn now so a dropped stream doesn't lose it.
        _append_turn(chat_path, "user", body.message)

        async def event_stream():
            import anthropic

            client = anthropic.AsyncAnthropic()
            accumulated: list[str] = []
            try:
                async with client.messages.stream(
                    model=CHAT_MODEL,
                    max_tokens=2048,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {
                                "type": "ephemeral",
                                "ttl": "1h",
                            },
                        }
                    ],
                    messages=messages,
                ) as stream:
                    async for text in stream.text_stream:
                        accumulated.append(text)
                        payload = json.dumps({"type": "delta", "text": text})
                        yield f"data: {payload}\n\n"

                full_reply = "".join(accumulated)
                if full_reply:
                    _append_turn(chat_path, "assistant", full_reply)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as exc:
                logger.exception("Chat stream failed")
                err = json.dumps({"type": "error", "message": str(exc)})
                yield f"data: {err}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router
