"""Page routes - HTML template rendering."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main SPA page (redirect to login if accounts require it)."""
        from gently.ui.web.accounts import get_account_store
        from gently.ui.web.auth import current_username
        store = get_account_store()
        if store is not None and store.has_users() and not current_username(request):
            return RedirectResponse("/login", status_code=302)
        return server.templates.TemplateResponse(
            "index.html",
            {"request": request, "active_section": "embryos", "is_live": True}
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
            "settings.html",
            {"request": request}
        )

    return router
