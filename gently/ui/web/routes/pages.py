"""Page routes - HTML template rendering."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main visualization page"""
        return server.templates.TemplateResponse(
            "index.html",
            {"request": request}
        )

    @router.get("/review", response_class=HTMLResponse)
    async def review_page(request: Request):
        """Serve the session review page"""
        return server.templates.TemplateResponse(
            "review.html",
            {"request": request}
        )

    @router.get("/campaigns", response_class=HTMLResponse)
    async def campaigns_page(request: Request):
        """Serve the campaigns/plans overview page"""
        return server.templates.TemplateResponse(
            "campaigns.html",
            {"request": request, "campaign_id": ""}
        )

    @router.get("/campaigns/{campaign_id}/review", response_class=HTMLResponse)
    async def plan_review_page(request: Request, campaign_id: str):
        """Serve the plan review page (same template, auto-opens campaign)"""
        return server.templates.TemplateResponse(
            "campaigns.html",
            {"request": request, "campaign_id": campaign_id}
        )

    @router.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        """Serve the dashboard settings page"""
        return server.templates.TemplateResponse(
            "settings.html",
            {"request": request}
        )

    return router
