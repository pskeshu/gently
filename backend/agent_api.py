"""
Agent API endpoints for Microscopy Copilot

Provides WebSocket and REST endpoints for conversational AI interaction
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import json
from pathlib import Path

from gently.agent import MicroscopyCopilot


# Pydantic models for API
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    timestamp: str


class ExperimentStatusResponse(BaseModel):
    status: str
    embryo_count: int
    start_time: Optional[str]
    current_plan: Optional[str]
    embryos: Dict


# Global copilot instance
# This is initialized when the backend starts
_copilot: Optional[MicroscopyCopilot] = None


def init_copilot(
    storage_path: Path,
    run_engine=None,
    devices: Optional[Dict] = None
) -> MicroscopyCopilot:
    """
    Initialize the global copilot instance

    Parameters
    ----------
    storage_path : Path
        Where to store experiment data
    run_engine : RunEngine, optional
        Bluesky RunEngine
    devices : dict, optional
        Ophyd devices

    Returns
    -------
    MicroscopyCopilot
        Initialized copilot
    """
    global _copilot
    _copilot = MicroscopyCopilot(
        storage_path=storage_path,
        run_engine=run_engine,
        devices=devices
    )
    return _copilot


def get_copilot() -> MicroscopyCopilot:
    """Get the global copilot instance"""
    if _copilot is None:
        raise RuntimeError("Copilot not initialized. Call init_copilot() first.")
    return _copilot


# Create router
router = APIRouter(prefix="/api/agent", tags=["agent"])


# ============================================================================
# WebSocket Endpoint for Chat
# ============================================================================

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with copilot

    Protocol:
    - Client sends: {"type": "message", "content": "user message here"}
    - Server responds: {"type": "response", "content": "copilot response", "timestamp": "..."}
    - Server can also send: {"type": "status", "data": {...}} for experiment updates
    """
    await websocket.accept()
    copilot = get_copilot()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data.get("type") == "message":
                user_message = data.get("content", "")

                # Send acknowledgment
                await websocket.send_json({
                    "type": "ack",
                    "timestamp": str(asyncio.get_event_loop().time())
                })

                # Get response from copilot
                try:
                    response = await copilot.handle_message(user_message)

                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "content": response,
                        "timestamp": str(asyncio.get_event_loop().time())
                    })

                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing message: {str(e)}",
                        "timestamp": str(asyncio.get_event_loop().time())
                    })

            elif data.get("type") == "get_history":
                # Send conversation history
                await websocket.send_json({
                    "type": "history",
                    "messages": copilot.conversation_history
                })

    except WebSocketDisconnect:
        print("Agent chat WebSocket disconnected")
    except Exception as e:
        print(f"Agent WebSocket error: {e}")
        await websocket.close()


# ============================================================================
# REST Endpoints
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def post_chat_message(request: ChatRequest):
    """
    Send a message to the copilot (REST alternative to WebSocket)

    Parameters
    ----------
    request : ChatRequest
        Message to send

    Returns
    -------
    ChatResponse
        Copilot's response
    """
    copilot = get_copilot()

    try:
        response = await copilot.handle_message(request.message)
        return ChatResponse(
            response=response,
            timestamp=str(asyncio.get_event_loop().time())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_conversation_history():
    """Get full conversation history"""
    copilot = get_copilot()
    return {"messages": copilot.conversation_history}


@router.delete("/history")
async def clear_conversation_history():
    """Clear conversation history (start fresh)"""
    copilot = get_copilot()
    copilot.conversation_history = []
    return {"status": "cleared"}


@router.get("/status", response_model=ExperimentStatusResponse)
async def get_experiment_status():
    """Get current experiment status"""
    copilot = get_copilot()

    return ExperimentStatusResponse(
        status=copilot.experiment.acquisition_status,
        embryo_count=len(copilot.experiment.embryos),
        start_time=copilot.experiment.start_time.isoformat() if copilot.experiment.start_time else None,
        current_plan=copilot.experiment.current_plan_name,
        embryos=copilot.experiment.to_dict()['embryos']
    )


@router.post("/load-embryos")
async def load_embryos(database: Dict):
    """
    Load embryos from calibration database

    Parameters
    ----------
    database : dict
        Embryo database with positions and calibrations

    Returns
    -------
    dict
        Status and loaded embryo count
    """
    copilot = get_copilot()
    copilot.load_embryos_from_database(database)

    return {
        "status": "loaded",
        "embryo_count": len(copilot.experiment.embryos),
        "embryo_ids": list(copilot.experiment.embryos.keys())
    }


@router.get("/embryos/{embryo_id}")
async def get_embryo_status(embryo_id: str):
    """Get detailed status for specific embryo"""
    copilot = get_copilot()

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        raise HTTPException(status_code=404, detail=f"Embryo {embryo_id} not found")

    return embryo.to_dict()


@router.post("/embryos/{embryo_id}/skip")
async def skip_embryo(embryo_id: str, reason: str = "User requested"):
    """Skip an embryo"""
    copilot = get_copilot()

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        raise HTTPException(status_code=404, detail=f"Embryo {embryo_id} not found")

    embryo.should_skip = True
    embryo.skip_reason = reason

    return {"status": "skipped", "embryo_id": embryo.id, "reason": reason}


@router.post("/embryos/{embryo_id}/resume")
async def resume_embryo(embryo_id: str):
    """Resume a skipped embryo"""
    copilot = get_copilot()

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        raise HTTPException(status_code=404, detail=f"Embryo {embryo_id} not found")

    embryo.should_skip = False
    embryo.skip_reason = None

    return {"status": "resumed", "embryo_id": embryo.id}
