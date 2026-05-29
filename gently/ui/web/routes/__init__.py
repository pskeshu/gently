"""
Route modules for the Visualization Server
============================================

Each module exposes a ``create_router(server)`` function that returns
a FastAPI ``APIRouter`` bound to the server instance.
"""

from .pages import create_router as create_pages_router
from .sessions import create_router as create_sessions_router
from .images import create_router as create_images_router
from .volumes import create_router as create_volumes_router
from .data import create_router as create_data_router
from .websocket import create_router as create_websocket_router
from .agent_ws import create_router as create_agent_ws_router
from .campaigns import create_router as create_campaigns_router
from .chat import create_router as create_chat_router
from .experiments import create_router as create_experiments_router
from .auth_routes import create_router as create_auth_router


def register_all_routes(server):
    """Register all route groups on the server's FastAPI app."""
    for factory in (
        create_pages_router,
        create_auth_router,
        create_sessions_router,
        create_campaigns_router,
        create_experiments_router,
        create_images_router,
        create_volumes_router,
        create_data_router,
        create_websocket_router,
        create_agent_ws_router,
        create_chat_router,
    ):
        router = factory(server)
        server.app.include_router(router)
