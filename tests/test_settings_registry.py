"""Settings are declared once, and drawn from the declaration.

The old Settings page was built control by control, and eight of its nineteen
view settings were read by nothing: saved faithfully, changing nothing. A
setting now says who reads it, and this fails if they do not.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.settings import settings
from gently.ui.web import auth
from gently.ui.web import settings_registry as registry
from gently.ui.web.routes import data as data_routes

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"
JS = WEB / "static" / "js"
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
EMBRYOS = (JS / "embryos.js").read_text(encoding="utf-8")
SETTINGS_JS = (JS / "settings.js").read_text(encoding="utf-8")
STORE_JS = (JS / "settings-store.js").read_text(encoding="utf-8")
APP_JS = (JS / "app.js").read_text(encoding="utf-8")

DEAD = (
    "criticalOvertimeRatio",
    "audioTick",
    "borderEncoding",
    "temperatureModel",
    "timeAxis",
    "imageSplitRatio",
    "autoAdvance",
    "showContrastive",
)


def _shipped() -> str:
    start = EMBRYOS.index("    dashboardConfig: {")
    return EMBRYOS[start : EMBRYOS.index("\n    },\n", start)]


# ── the registry itself ──────────────────────────────────────────────────


def test_keys_are_unique_and_every_category_exists():
    keys = [s.key for s in registry.SETTINGS]
    assert len(keys) == len(set(keys))
    known = {c["id"] for c in registry.CATEGORIES}
    assert {s.category for s in registry.SETTINGS} <= known
    assert [c["id"] for c in registry.CATEGORIES] == [
        "views",
        "alerts",
        "experiment",
        "microscope",
        "recording",
        "assistant",
        "system",
    ]


def test_every_category_has_something_in_it():
    used = {s.category for s in registry.SETTINGS}
    assert used == {c["id"] for c in registry.CATEGORIES}


@pytest.mark.parametrize("s", [s for s in registry.SETTINGS if s.readers], ids=lambda s: s.key)
def test_a_setting_is_read_where_it_says_it_is(s):
    for rel, needle in s.readers:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert needle in text, f"{s.key}: nothing in {rel} reads it ({needle!r})"


@pytest.mark.parametrize(
    "s",
    [s for s in registry.SETTINGS if s.type not in ("link", "custom", "readonly")],
    ids=lambda s: s.key,
)
def test_a_setting_that_can_be_changed_names_its_reader_and_its_store(s):
    assert s.readers, f"{s.key} can be changed and says nothing reads it"
    assert s.store, f"{s.key} can be changed and has nowhere to be kept"
    assert s.store == "theme" or s.store.split(":")[0] in ("prefs", "env", "launch")


@pytest.mark.parametrize(
    "s",
    [s for s in registry.SETTINGS if s.store.startswith("prefs:")],
    ids=lambda s: s.key,
)
def test_a_browser_preference_ships_with_a_value(s):
    """Its leaf is in the values embryos.js ships with, or it is the Atrium's or
    the agent panel's, which atrium.js and agent-chat.js read for themselves."""
    leaf = s.store.split(":", 1)[1].split(".")[-1]
    if leaf in ("atrium", "agentPanel"):
        return
    assert re.search(rf"\b{leaf}:", _shipped()), f"{s.key}: embryos.js ships no {leaf}"


def test_nothing_is_shipped_that_is_not_declared():
    declared = {
        s.store.split(":", 1)[1].split(".")[-1]
        for s in registry.SETTINGS
        if s.store.startswith("prefs:")
    }
    groups = {"board", "filmstrip", "vitals", "ambient"}
    shipped = set(re.findall(r"^\s+(\w+):", _shipped(), re.M)) - groups - {"dashboardConfig"}
    assert shipped <= declared, f"shipped but not in the registry: {sorted(shipped - declared)}"


@pytest.mark.parametrize("name", DEAD)
def test_the_eight_that_nothing_read_are_gone(name):
    assert name not in EMBRYOS
    assert name not in SETTINGS_JS
    assert not any(name in s.store for s in registry.SETTINGS)


def test_the_board_offers_the_columns_the_board_has():
    cols = next(s for s in registry.SETTINGS if s.key == "views.board.columns")
    offered = [v for v, _ in cols.choices]
    assert offered == ["stage", "clock", "stereo", "pace", "eta", "sparkline", "alert"]
    for col in offered:
        assert f"cols.includes('{col}')" in EMBRYOS, f"the board has no {col} column"
    assert "confidence" not in offered and "rate" not in offered


def test_a_restart_setting_reads_the_value_it_edits():
    for s in registry.env_settings():
        assert s.applies == registry.RESTART and s.reach == registry.RIG
        value = registry.current_value(s, settings)
        assert value is not None, f"{s.key}: {s.source} is not in gently.settings"


def test_no_secret_port_or_model_can_be_edited():
    for s in registry.env_settings():
        name = registry.env_name(s)
        for word in ("PORT", "HOST", "MODEL", "KEY", "TOKEN", "STORAGE", "PASSWORD"):
            assert word not in name, f"{name} must not be editable from the browser"


def test_recording_is_on_unless_switched_off():
    rec = registry.by_key()["recording.enabled"]
    assert rec.default is True and registry.env_name(rec) == "GENTLY_REPLAY"
    assert settings.ui.replay is True


# ── the routes ───────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(data_routes, "_HARDWARE_CONFIG_PATH", tmp_path / "hardware.yaml")
    app = FastAPI()
    app.include_router(data_routes.create_router(MagicMock()))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_the_schema_is_the_registry(client):
    d = client.get("/api/settings/schema").json()
    assert [c["id"] for c in d["categories"]] == [c["id"] for c in registry.CATEGORIES]
    assert [s["key"] for s in d["settings"]] == [s.key for s in registry.SETTINGS]
    by = {s["key"]: s for s in d["settings"]}
    assert by["recording.enabled"]["value"] is True
    assert by["recording.enabled"]["applies"] == "restart"
    assert by["views.film.thumbnailSize"]["choices"][1] == {"value": 56, "label": "Medium"}
    assert "value" not in by["views.film.thumbnailSize"], "a browser setting has no value here"
    assert by["experiment.plan"]["where"] == "Devices › Operate › Acquisition"
    assert by["system.version"]["value"].startswith(settings.__class__.__module__[:0] + "1.")


def test_the_overrides_route_writes_only_what_the_registry_keeps_there(client):
    editable = {i["env"] for i in client.get("/api/config/settings-overrides").json()["items"]}
    assert editable == {registry.env_name(s) for s in registry.env_settings()}
    assert (
        client.put("/api/config/settings-overrides", json={"GENTLY_VIZ_PORT": 1}).status_code == 400
    )
    r = client.put("/api/config/settings-overrides", json={"GENTLY_REPLAY_FIDELITY": "actions"})
    assert r.status_code == 200 and r.json()["restart_required"] is True
    by = {s["key"]: s for s in client.get("/api/settings/schema").json()["settings"]}
    assert by["recording.fidelity"]["overridden"] is True


# ── the tab ──────────────────────────────────────────────────────────────


def test_settings_is_a_tab_of_the_app():
    assert 'id="settings-content" class="tab-content"' in INDEX
    assert 'class="v2-nav-item" data-tab="settings"' in INDEX
    assert "SETTINGS: 'settings'" in (JS / "utils.js").read_text(encoding="utf-8")
    assert "if (tabName === TABS.SETTINGS && typeof SettingsTab !== 'undefined')" in APP_JS
    assert not (WEB / "templates" / "settings.html").exists()


def test_the_old_address_leads_into_the_app():
    from gently.ui.web.routes.pages import create_router

    app = FastAPI()
    app.include_router(create_router(MagicMock()))
    r = TestClient(app).get("/settings", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/#settings"


def test_the_store_is_loaded_before_anything_that_reads_it():
    assert INDEX.index("settings-store.js") < INDEX.index("/static/js/embryos.js")
    assert INDEX.index("settings-store.js") < INDEX.index("/static/js/settings.js")


def test_every_custom_block_is_in_the_page():
    for s in registry.SETTINGS:
        if s.type == "custom":
            assert f'id="{s.block}"' in INDEX, f"{s.key}: no block {s.block} in the page"


def test_the_views_read_through_the_store_and_hear_a_change():
    load = EMBRYOS[EMBRYOS.index("    loadDashboardConfig() {") :][:900]
    assert "SettingsStore.merged(this._shippedConfig)" in load
    assert "ClientEventBus.on('SETTINGS_CHANGED'" in EMBRYOS
    assert "SettingsStore.loadRigDefaults();" in EMBRYOS


def test_the_rigs_defaults_sit_under_the_browsers_choices():
    fn = STORE_JS[STORE_JS.index("    function merged(shipped) {") :][:200]
    assert "deepMerge(deepMerge(shipped || {}, _rig), local())" in fn


def test_settings_is_where_the_atrium_is_switched_off_so_it_opens_the_tabs():
    atrium = (JS / "atrium.js").read_text(encoding="utf-8")
    fn = atrium[atrium.index("    function wanted() {") :][:500]
    assert "location.hash" in fn and "return false;" in fn
    assert 'href="/#settings"' in atrium


def test_the_agent_panel_is_collapsed_unless_settings_says_otherwise():
    """The scope's operators asked for a quiet start: the chat folds until it is
    wanted. The registry's default and the fallback agent-chat.js reads with must
    agree, or Settings would show one thing and the page do another."""
    s = next(s for s in registry.SETTINGS if s.key == "assistant.panel")
    assert s.default == "collapsed"
    assert [c[0] for c in s.choices] == ["collapsed", "open", "remember"]
    assert s.store == "prefs:agentPanel" and s.applies == registry.LOAD
    chat = (JS / "agent-chat.js").read_text(encoding="utf-8")
    assert "SettingsStore.get('agentPanel', 'collapsed')" in chat
