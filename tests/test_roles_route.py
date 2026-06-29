"""Tests for GET /api/roles — embryo role registry endpoint (Task 3, D2).

Verifies:
- Returns all roles in the registry including lineaging and role_class.
- Each entry has the expected fields.
- Never raises a 500 (graceful).
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.ui.web.routes.roles import create_router


def _client():
    """Build a minimal TestClient with only the roles router."""
    server = object()  # roles route never uses server attributes
    app = FastAPI()
    app.include_router(create_router(server))
    return TestClient(app)


def test_roles_returns_200():
    r = _client().get("/api/roles")
    assert r.status_code == 200


def test_roles_body_has_roles_key():
    body = _client().get("/api/roles").json()
    assert "roles" in body
    assert isinstance(body["roles"], list)


def test_roles_includes_all_registry_entries():
    """All four built-in roles must be present."""
    roles = _client().get("/api/roles").json()["roles"]
    names = {r["name"] for r in roles}
    assert names == {"unassigned", "test", "calibration", "lineaging"}


def test_roles_include_lineaging():
    roles = _client().get("/api/roles").json()["roles"]
    lineaging = next(r for r in roles if r["name"] == "lineaging")
    assert lineaging["role_class"] == "reference"
    assert lineaging["ui_color"] == "#33cc88"
    assert lineaging["ui_icon"] == "triangle"


def test_roles_entry_has_all_required_fields():
    """Every role entry exposes the required fields."""
    required = {
        "name",
        "description",
        "role_class",
        "ui_color",
        "ui_icon",
        "default_cadence_seconds",
    }
    roles = _client().get("/api/roles").json()["roles"]
    for role in roles:
        missing = required - role.keys()
        assert not missing, f"Role {role.get('name')!r} missing fields: {missing}"


def test_test_role_class_is_subject():
    roles = _client().get("/api/roles").json()["roles"]
    test_role = next(r for r in roles if r["name"] == "test")
    assert test_role["role_class"] == "subject"


def test_calibration_role_class_is_reference():
    roles = _client().get("/api/roles").json()["roles"]
    cal = next(r for r in roles if r["name"] == "calibration")
    assert cal["role_class"] == "reference"


def test_cadence_seconds_is_numeric():
    roles = _client().get("/api/roles").json()["roles"]
    for role in roles:
        assert isinstance(role["default_cadence_seconds"], (int, float))


def test_never_500():
    """The route must never raise a 500 regardless of server state."""
    r = _client().get("/api/roles")
    assert r.status_code != 500
