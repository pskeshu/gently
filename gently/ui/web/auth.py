"""Web-UI authorization roles.

Two roles:
  view    -- read-only. GET endpoints, SSE / WebSocket event streams.
  control -- can drive hardware (POST/PUT/DELETE). Localhost is always
             control; remote callers must present a matching token in the
             X-Gently-Token header (token read from GENTLY_CONTROL_TOKEN).

Routes that move hardware or mutate persistent state declare a dependency:

    from gently.ui.web.auth import require_control

    @router.post("/api/devices/foo")
    async def foo(_=Depends(require_control)):
        ...

Default-deny on control: if the token env var is unset, remote callers get
view-only access until the operator provisions a token. That matches the
"diSPIM computer alone gives control directions" intent while leaving room
for authenticated remote operators later.
"""

from __future__ import annotations

import logging
import os
from enum import Enum

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Header name used to upgrade a remote session to control role. Single shared
# token for now; per-user identities can be layered on later without changing
# this module's public surface.
_TOKEN_HEADER = "X-Gently-Token"
_TOKEN_ENV    = "GENTLY_CONTROL_TOKEN"


class Role(str, Enum):
    VIEW    = "view"
    CONTROL = "control"


def _configured_token() -> str | None:
    """Return the shared control token, or None if no token is provisioned.

    Read fresh each request so the operator can rotate the token without
    restarting the web server.
    """
    tok = os.environ.get(_TOKEN_ENV, "").strip()
    return tok or None


def resolve_role(request: Request) -> Role:
    """Determine the effective role for a request.

    Localhost is always control (the diSPIM box). Remote callers need to
    present X-Gently-Token matching GENTLY_CONTROL_TOKEN.
    """
    client = request.client
    host = client.host if client else None
    if host in _LOOPBACK_HOSTS:
        return Role.CONTROL

    token = _configured_token()
    if token is not None:
        supplied = request.headers.get(_TOKEN_HEADER, "").strip()
        if supplied and supplied == token:
            return Role.CONTROL

    return Role.VIEW


def require_control(request: Request) -> Role:
    """FastAPI dependency — 403 unless the caller has the control role.

    Logs the denied client host (without leaking the token) so the operator
    can spot if a remote browser is trying to drive hardware.
    """
    role = resolve_role(request)
    if role is Role.CONTROL:
        return role
    host = request.client.host if request.client else "unknown"
    logger.warning("control-route 403 for %s -> %s %s",
                   host, request.method, request.url.path)
    raise HTTPException(
        status_code=403,
        detail="control role required (this endpoint moves hardware or "
               "mutates persistent state; localhost has it by default, "
               "remote callers need X-Gently-Token)",
    )
