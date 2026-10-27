"""
Route modules for the Visualization Server
============================================

Each module exposes a ``create_router(server)`` function that returns
a FastAPI ``APIRouter`` bound to the server instance.
"""

from gently.tools.figure import create_router as create_figure_router
from gently.tools.pool import create_router as create_pool_router
from gently.tools.select import create_router as create_select_router
from gently.tools.twitch_hatch import create_router as create_twitch_hatch_router

from .agent_ws import create_router as create_agent_ws_router
from .auth_routes import create_router as create_auth_router
from .bf import create_router as create_bf_router
from .brightfield import create_router as create_brightfield_router
from .calibration_records import create_router as create_calibration_records_router
from .campaigns import create_router as create_campaigns_router
from .chat import create_router as create_chat_router
from .context import create_router as create_context_router
from .data import create_router as create_data_router
from .device_layer import create_router as create_device_layer_router
from .experiments import create_router as create_experiments_router
from .images import create_router as create_images_router
from .logs import create_router as create_logs_router
from .notebook import create_router as create_notebook_router
from .operation_plan import create_router as create_operation_plan_router
from .pages import create_router as create_pages_router
from .replay import create_router as create_replay_router
from .reveal import create_router as create_reveal_router
from .roles import create_router as create_roles_router
from .search import create_router as create_search_router
from .sessions import create_router as create_sessions_router
from .tactic_library import create_router as create_tactic_library_router
from .temperature import create_router as create_temperature_router
from .ui_prefs import create_router as create_ui_prefs_router
from .volumes import create_router as create_volumes_router
from .websocket import create_router as create_websocket_router


def register_all_routes(server):
    """Register all route groups on the server's FastAPI app."""
    for factory in (
        create_pages_router,
        create_auth_router,
        create_device_layer_router,
        create_sessions_router,
        create_campaigns_router,
        create_experiments_router,
        create_images_router,
        create_logs_router,
        create_volumes_router,
        create_data_router,
        create_websocket_router,
        create_agent_ws_router,
        create_chat_router,
        create_context_router,
        create_notebook_router,
        create_temperature_router,
        create_operation_plan_router,
        create_roles_router,
        create_tactic_library_router,
        create_bf_router,
        create_brightfield_router,
        create_calibration_records_router,
        create_replay_router,
        create_reveal_router,
        create_ui_prefs_router,
        create_search_router,
        create_figure_router,
        create_pool_router,
        create_select_router,
        create_twitch_hatch_router,
    ):
        router = factory(server)
        server.app.include_router(router)
