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
        from gently.core.reveal import is_local
        from gently.ui.web.auth import unnamed

        # Where this browser is, and what it is called until it says
        # otherwise. Known whether or not there are accounts.
        host = request.client.host if request.client else None
        here = {"local": is_local(host), "unnamed": unnamed(host)}
        store = get_account_store()
        if store is None or not store.has_users():
            # No accounts: the person is whoever the landing page's cookie
            # names, shown by display name.
            from gently.ui.web.operators import names_of

            pid = current_username(request)
            operator = {"id": pid, "name": names_of().get(pid, pid)} if pid else None
            return JSONResponse(
                {"accounts": False, "authenticated": False, "operator": operator, **here}
            )
        username = current_username(request)
        if not username:
            return JSONResponse({"accounts": True, "authenticated": False, **here})
        role = store.get_role(username)
        return JSONResponse(
            {
                "accounts": True,
                "authenticated": True,
                "username": username,
                "role": role,
                "can_control": role in CONTROL_ROLES,
                **here,
            }
        )

    @router.get("/api/auth/users")
    async def list_users():
        """Who can be at the rig: the account names when there are accounts,
        else the operators this rig has seen, for choosing on the landing page."""
        from gently.ui.web.operators import list_operators

        colour = getattr(getattr(server, "manager", None), "_generate_color", None)
        paint = (lambda u: {**u, "color": colour(u["id"])}) if callable(colour) else (lambda u: u)
        store = get_account_store()
        if store is None or not store.has_users():
            people = list_operators(getattr(server, "gently_store", None))
            return JSONResponse({"accounts": False, "users": [paint(u) for u in people]})
        return JSONResponse(
            {
                "accounts": True,
                "users": [
                    paint({"id": u["username"], "name": u["username"]}) for u in store.list_users()
                ],
            }
        )

    @router.post("/api/auth/operator")
    async def set_operator(request: Request):
        """Who is here now, on a rig with no accounts: remember the name and
        set the cookie current_username reads."""
        from gently.ui.web.auth import OPERATOR_COOKIE
        from gently.ui.web.operators import remember_operator, touch_operator

        store = get_account_store()
        if store is not None and store.has_users():
            return JSONResponse({"error": "accounts are configured; sign in"}, status_code=409)
        body = await request.json()
        try:
            # A pick sends the id; a typed name makes or finds the person.
            who = (
                touch_operator(body["id"])
                if body.get("id")
                else remember_operator(body.get("name") or "")
            )
        except KeyError:
            return JSONResponse({"error": "no such person"}, status_code=404)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        resp = JSONResponse({"ok": True, **who})
        resp.set_cookie(OPERATOR_COOKIE, who["id"], max_age=365 * 86400, samesite="lax", path="/")
        # Becoming someone mid-session: the live session remembers them too.
        try:
            sid = server._current_session_id()
            gstore = getattr(server, "gently_store", None)
            if sid and gstore is not None:
                gstore.add_session_operator(sid, who["id"])
        except Exception:
            pass
        return resp

    @router.post("/api/auth/operator/rename")
    async def rename_operator_route(request: Request):
        """A display name changed, a typo or otherwise. The id, and so every
        record and cookie, stays."""
        from gently.ui.web.operators import rename_operator

        store = get_account_store()
        if store is not None and store.has_users():
            return JSONResponse({"error": "accounts are configured"}, status_code=409)
        body = await request.json()
        try:
            who = rename_operator(body.get("id") or "", body.get("name") or "")
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        return JSONResponse({"ok": True, **who})

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
