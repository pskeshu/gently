import pytest
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
