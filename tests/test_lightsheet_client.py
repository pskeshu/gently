from gently.hardware.dispim.client import DiSPIMMicroscope


async def test_set_params_posts_body(monkeypatch):
    m = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    sent = {}

    async def fake_post(path, body):
        sent["path"] = path
        sent["body"] = body
        return {"params": body}

    m._api_post = fake_post  # confirm the real low-level POST helper name
    res = await m.set_lightsheet_live_params(galvo=1.0, piezo=42.0, exposure=15.0)
    assert sent["path"] == "/api/lightsheet/live/params"
    assert sent["body"] == {"galvo": 1.0, "piezo": 42.0, "exposure": 15.0}
    assert res == {"params": {"galvo": 1.0, "piezo": 42.0, "exposure": 15.0}}


def test_stream_lightsheet_is_async_generator():
    m = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    import inspect

    assert inspect.isasyncgenfunction(m.stream_lightsheet)


# ---------------------------------------------------------------------------
# set_laser_config / get_laser_configs
# ---------------------------------------------------------------------------


async def test_set_laser_config_posts_correct_path_and_body():
    """set_laser_config posts to /api/laser/config with {"config": <name>}."""
    m = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    sent = {}

    async def fake_post(path, body):
        sent["path"] = path
        sent["body"] = body
        return {"success": True, "config": body["config"]}

    m._api_post = fake_post
    res = await m.set_laser_config("ALL OFF")
    assert sent["path"] == "/api/laser/config"
    assert sent["body"] == {"config": "ALL OFF"}
    assert res["success"] is True
    assert res["config"] == "ALL OFF"


async def test_get_laser_configs_gets_correct_path():
    """get_laser_configs GETs /api/laser/configs."""
    m = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    fetched = {}

    async def fake_get(path):
        fetched["path"] = path
        return {"configs": ["488 only", "ALL OFF"]}

    m._api_get = fake_get
    res = await m.get_laser_configs()
    assert fetched["path"] == "/api/laser/configs"
    assert res == {"configs": ["488 only", "ALL OFF"]}
