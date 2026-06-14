"""Auth routes — login / logout / me, plus the login page.

Self-managed accounts (see gently/ui/web/accounts.py). Login issues a signed
session cookie; roles (viewer/operator/admin) gate control elsewhere via
gently.ui.web.auth.resolve_role and the /ws/agent control lock.
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from gently.ui.web.accounts import (
    _SESSION_TTL_SECONDS,
    CONTROL_ROLES,
    ROLES,
    get_account_store,
)
from gently.ui.web.auth import SESSION_COOKIE, current_username

logger = logging.getLogger(__name__)


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _secure(request: Request) -> bool:
        # Only mark the cookie Secure over HTTPS, else the browser drops it on
        # plain-HTTP LAN deployments.
        return request.url.scheme == "https"

    @router.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        store = get_account_store()
        if store is None or not store.has_users():
            return RedirectResponse("/", status_code=302)
        if current_username(request):
            return RedirectResponse("/", status_code=302)
        return server.templates.TemplateResponse(request, "login.html")

    @router.post("/api/auth/login")
    async def login(request: Request):
        store = get_account_store()
        if store is None or not store.has_users():
            return JSONResponse({"error": "accounts not configured"}, status_code=400)
        try:
            body = await request.json()
        except Exception:
            body = {}
        username = (body.get("username") or "").strip()
        password = body.get("password") or ""
        role = store.verify_password(username, password)
        if not role:
            host = request.client.host if request.client else "?"
            logger.warning("login failed for %r from %s", username, host)
            return JSONResponse({"error": "Invalid username or password"}, status_code=401)
        token = store.issue_session(username)
        resp = JSONResponse({"ok": True, "username": username, "role": role})
        resp.set_cookie(
            SESSION_COOKIE,
            token,
            httponly=True,
            samesite="lax",
            secure=_secure(request),
            max_age=_SESSION_TTL_SECONDS,
            path="/",
        )
        logger.info("login ok: %s (%s)", username, role)
        return resp

    @router.post("/api/auth/logout")
    async def logout(request: Request):
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(SESSION_COOKIE, path="/")
        return resp

    @router.get("/api/auth/me")
    async def me(request: Request):
        store = get_account_store()
        if store is None or not store.has_users():
            return JSONResponse({"accounts": False, "authenticated": False})
        username = current_username(request)
        if not username:
            return JSONResponse({"accounts": True, "authenticated": False})
        role = store.get_role(username)
        return JSONResponse(
            {
                "accounts": True,
                "authenticated": True,
                "username": username,
                "role": role,
                "can_control": role in CONTROL_ROLES,
            }
        )

    @router.post("/api/auth/users")
    async def create_user(request: Request):
        """Admin-only: provision a new account."""
        store = get_account_store()
        if store is None:
            return JSONResponse({"error": "accounts not configured"}, status_code=400)
        requester = current_username(request)
        if not requester or store.get_role(requester) != "admin":
            return JSONResponse({"error": "admin role required"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            body = {}
        new_user = (body.get("username") or "").strip()
        password = body.get("password") or ""
        role = body.get("role") or "viewer"
        if not new_user or not password:
            return JSONResponse({"error": "username and password required"}, status_code=400)
        if role not in ROLES:
            return JSONResponse({"error": f"role must be one of {list(ROLES)}"}, status_code=400)
        try:
            store.create_user(new_user, password, role)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        logger.info("admin %s created user %s (%s)", requester, new_user, role)
        return JSONResponse({"ok": True, "username": new_user, "role": role})

    return router
