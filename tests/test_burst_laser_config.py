"""Task 2: BurstAcquisition laser_config threading tests."""

from gently.app.orchestration.exclusive import BurstAcquisition


class FakeClient:
    def __init__(self):
        self.calls = []

    async def acquire_burst(self, **kw):
        self.calls.append(kw)
        return {"success": True, "request_id": "b1", "frames": []}


async def test_burst_passes_laser_config(monkeypatch):
    b = BurstAcquisition("emb1", frames=3, mode="1hz", num_slices=1, laser_config="ALL OFF")
    assert b._laser_config == "ALL OFF"


async def test_burst_laser_config_default_none():
    b = BurstAcquisition("emb1", frames=3, mode="1hz", num_slices=1)
    assert b._laser_config is None
