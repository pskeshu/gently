"""Device-layer supervision + launch gate routes.

Two surfaces, both backed by ``DeviceLayerSupervisor``:

- **Launch gate** (``GET /launch``) — the bare two-question start screen
  (hardware on/off, agent on/off) with its persisted choices.
- **Devices panel** (``/api/device-layer/*``) — runtime status, log tail, and
  Start / Stop controls that mirror the gate's hardware block.

Plus the whole-backend shutdown handshake (``POST /api/shutdown``, issue #85)
used by the desktop shell on window-close — loopback-only, stops the managed
device layer gracefully, then asks the launcher to exit.

Control-mutating endpoints (start/stop) are gated behind ``require_control``,
matching the rest of the hardware routes. Stop reuses the 409 + ``"blocked"``
mid-run guard pattern: if hardware looks active, the caller must confirm.

See ``docs/superpowers/specs/2026-07-02-unified-launcher-design.md`` (RFC #78).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from gently.ui.web.auth import require_control
from gently.ui.web.launch_prefs import detect_sam_device, load_prefs, save_prefs, stored_prefs

logger = logging.getLogger(__name__)


def get_supervisor(server):
    """Return the server's DeviceLayerSupervisor, creating one on first use.

    ``launch_gently`` attaches a supervisor to the viz server at boot, but we
    lazily construct one if it's absent so these routes work standalone (and so
    the panel degrades gracefully rather than 500-ing).
    """
    sup = getattr(server, "device_supervisor", None)
    if sup is None:
        from gently.app.device_supervisor import DeviceLayerSupervisor

        sup = DeviceLayerSupervisor()
        server.device_supervisor = sup
    return sup


def _hardware_active(server) -> bool:
    """Best-effort: is an acquisition / timelapse currently running?

    Used by the stop guard. Conservative — only reports active when we can see a
    clearly-running timelapse; unknown state reads as inactive so a stuck flag
    can't wedge the Stop button.
    """
    tracker = getattr(server, "timelapse_tracker", None)
    status = getattr(tracker, "status", None)
    return str(status).lower() in ("running", "acquiring", "active")


def create_router(server) -> APIRouter:
    router = APIRouter()

    # ── Launch gate ──────────────────────────────────────────────────────

    @router.get("/launch", response_class=HTMLResponse)
    async def launch_gate(request: Request):
        """The bare two-question launch screen (prefilled from last choice)."""
        return server.templates.TemplateResponse(request, "launch.html", {})

    @router.get("/api/launch/prefs")
    async def get_launch_prefs():
        """Persisted launch choices (hardware/agent toggles + advanced defaults).

        Adds `sam_detected` (auto-detected GPU/CPU) and `sam_device_raw` (the
        stored 'auto'/'cuda'/'cpu' before resolution) for the Settings UI.
        """
        prefs = load_prefs()
        prefs["sam_detected"] = detect_sam_device()
        prefs["sam_device_raw"] = stored_prefs().get("sam_device", "auto")
        return prefs

    @router.post("/api/launch/prefs", dependencies=[Depends(require_control)])
    async def set_launch_prefs(request: Request):
        """Persist launch choices; returns the merged result."""
        body = await request.json()
        return save_prefs(body if isinstance(body, dict) else {})

    @router.post("/api/launch/go", dependencies=[Depends(require_control)])
    async def launch_go(request: Request):
        """Submit the launch gate: persist the toggles, mark the gate passed
        (so / stops bouncing to /launch), and — if the microscope toggle is on —
        start the device layer.

        Returns fast: start() only *spawns* the child. MMCore init and its
        per-stage progress happen in the background and are polled via
        /api/device-layer/status, so the UI can hand off to the dashboard
        immediately and follow the boot there.
        """
        body = await _safe_json(request)
        prefs = save_prefs(body if isinstance(body, dict) else {})
        server.gate_passed = True
        device = None
        if prefs.get("hardware"):
            try:
                device = get_supervisor(server).start()
            except (FileNotFoundError, OSError) as e:
                logger.error("device layer start failed from launch gate: %s", e)
                return {"ok": True, "hardware": True, "error": str(e)}
        return {
            "ok": True,
            "hardware": bool(prefs.get("hardware")),
            "agent": bool(prefs.get("agent")),
            "device": device,
        }

    # ── Devices panel: device-layer supervision ──────────────────────────

    @router.get("/api/device-layer/status")
    async def device_layer_status():
        """Live status: running / stopped / external / crashed (+ short log tail)."""
        return get_supervisor(server).status()

    @router.get("/api/device-layer/log")
    async def device_layer_log(limit: int = 200):
        """Recent captured console output (oldest → newest) for the console view."""
        return {"lines": get_supervisor(server).log_tail(limit)}

    @router.post("/api/device-layer/start", dependencies=[Depends(require_control)])
    async def device_layer_start(request: Request):
        """Spawn (or adopt-if-external) the device layer per the request body.

        Body (all optional): ``{"sam_device": "cuda"|"cpu", "config_path": "..."}``.
        """
        body = await _safe_json(request)
        sup = get_supervisor(server)
        try:
            status = sup.start(
                sam_device=body.get("sam_device"),
                config_path=body.get("config_path"),
            )
        except (FileNotFoundError, OSError) as e:
            logger.error("device layer start failed: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)
        return status

    @router.post("/api/device-layer/stop", dependencies=[Depends(require_control)])
    async def device_layer_stop(request: Request):
        """Stop the managed device layer, with a mid-run confirmation guard.

        If hardware looks active and the body doesn't carry ``{"confirm": true}``,
        returns 409 ``{"blocked": true, ...}`` instead of stopping (same shape the
        thermalizer stop-guard uses). ``{"force": true}`` skips the grace period.
        """
        body = await _safe_json(request)
        sup = get_supervisor(server)

        if _hardware_active(server) and not body.get("confirm"):
            return JSONResponse(
                {
                    "blocked": True,
                    "reason": "hardware is active — an acquisition is running",
                    "hint": 'resend with {"confirm": true} to stop anyway',
                },
                status_code=409,
            )

        return sup.stop(force=bool(body.get("force")))

    # ── Whole-backend shutdown (desktop shell handshake, issue #85) ───────

    @router.post("/api/shutdown")
    async def shutdown_backend(request: Request):
        """Gracefully stop the entire backend (device layer + agent + server).

        Called by the desktop shell on window-close so the backend can drain
        state (e.g. session-replay final batches) and stop the device layer via
        its clean SIGTERM path *before* the shell's kill / Job Object floor.

        Loopback-only: the shell carries no auth cookie, so instead of
        ``require_control`` the guard is the connection source itself — only
        127.0.0.1/::1 may ask the process to die. Mid-run, the standard 409
        ``{"blocked": true}`` confirm guard applies (resend with
        ``{"confirm": true}``); ``{"force": true}`` hard-kills the device layer.
        """
        host = request.client.host if request.client else None
        if host not in ("127.0.0.1", "::1"):
            return JSONResponse(
                {"error": "shutdown may only be requested from localhost"},
                status_code=403,
            )

        body = await _safe_json(request)

        if _hardware_active(server) and not body.get("confirm"):
            return JSONResponse(
                {
                    "blocked": True,
                    "reason": "hardware is active — an acquisition is running",
                    "hint": 'resend with {"confirm": true} to stop anyway',
                },
                status_code=409,
            )

        # Stop a MANAGED device-layer child first (graceful path; no-ops for
        # external/absent). stop() blocks in proc.wait() under a lock, so keep
        # it off the event loop.
        sup = get_supervisor(server)
        try:
            await asyncio.to_thread(sup.stop, force=bool(body.get("force")))
        except Exception:
            logger.exception("device-layer stop during shutdown failed (continuing)")

        rs = getattr(server, "request_shutdown", None)
        if rs is None:
            return JSONResponse({"error": "shutdown not wired by launcher"}, status_code=501)
        # Small delay so this response flushes before uvicorn starts exiting.
        asyncio.get_running_loop().call_later(0.3, rs)
        return {"ok": True, "stopping": True}

    return router


async def _safe_json(request: Request) -> dict:
    """Parse a JSON body, tolerating an empty/absent one (returns {})."""
    try:
        body = await request.json()
    except (ValueError, TypeError):
        return {}
    return body if isinstance(body, dict) else {}
