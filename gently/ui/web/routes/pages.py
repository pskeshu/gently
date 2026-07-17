"""Page routes - HTML template rendering."""

import time

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from gently.settings import settings


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main SPA page.

        Viewing is open to everyone — the dashboard loads in view mode with no
        login. Signing in is an *elevation* to control (handled in-app via the
        chat window's "Sign in" affordance), not a gate on the page itself.

        The launch gate is the entry point: until it's submitted this session,
        every visit to / bounces to /launch (RFC #78 defer-init boot).
        """
        if not getattr(server, "gate_passed", False):
            return RedirectResponse("/launch", status_code=302)
        # The v2 landing ("what are we doing today?") is for STARTING fresh. Skip
        # it when resuming a session (one-shot flag from the resume route) or when
        # the live session already has work — so a resumed/underway session lands
        # straight in the workspace instead of bouncing through the welcome screen.
        # Resumed within the last few seconds? (time-window, not a one-shot, so
        # all clients the resume-broadcast reloads skip the landing together.)
        just_resumed = (time.monotonic() - getattr(server, "_resumed_at", 0.0)) < 15.0
        has_work = False
        try:
            bridge = getattr(server, "agent_bridge", None)
            agent = getattr(bridge, "agent", None) if bridge else None
            if agent is not None:
                has_work = len(agent.experiment.embryos) > 0
        except Exception:
            has_work = False
        show_landing = bool(settings.ui.ux_v2) and not (just_resumed or has_work)
        return server.templates.TemplateResponse(
            request,
            "index.html",
            {
                "active_section": "embryos",
                "is_live": True,
                "ux_v2": settings.ui.ux_v2,
                "show_landing": show_landing,
            },
        )

    # Standalone URLs redirect to SPA with hash fragment for tab routing
    @router.get("/review")
    async def review_page():
        return RedirectResponse("/#sessions", status_code=302)

    @router.get("/campaigns")
    async def campaigns_page():
        return RedirectResponse("/#plans", status_code=302)

    @router.get("/campaigns/{campaign_id}/review")
    async def plan_review_page(campaign_id: str):
        return RedirectResponse(f"/#plans:{campaign_id}", status_code=302)

    @router.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        """Serve the dashboard settings page"""
        return server.templates.TemplateResponse(
            request,
            "settings.html",
        )

    return router
