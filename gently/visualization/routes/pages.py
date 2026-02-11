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

    return router
