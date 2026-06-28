"""Embryo Role Registry route.

Returns the static REGISTRY of embryo roles from ``gently.harness.roles``.
Never raises a 500 — the registry is global and read-only.
"""

from fastapi import APIRouter

from gently.harness.roles import REGISTRY


def create_router(server) -> APIRouter:  # noqa: ARG001 (server not needed; registry is global)
    router = APIRouter()

    @router.get("/api/roles")
    async def get_roles():
        roles = [
            {
                "name": role.name,
                "description": role.description,
                "role_class": role.role_class,
                "ui_color": role.ui_color,
                "ui_icon": role.ui_icon,
                "default_cadence_seconds": role.default_cadence_seconds,
            }
            for role in REGISTRY.values()
        ]
        return {"roles": roles}

    return router
