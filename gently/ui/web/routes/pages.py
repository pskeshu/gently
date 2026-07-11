"""Page routes - HTML template rendering."""

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
        """
        return server.templates.TemplateResponse(
            request,
            "index.html",
            {"active_section": "embryos", "is_live": True, "ux_v2": settings.ui.ux_v2},
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
