import pytest
from gently.app.orchestration.temperature_protocol import run_temp_change_burst_protocol
from gently.core.event_bus import EventType


class FakeClient:
    def __init__(self): self.laser=None; self.led=None; self.setpoint=None; self._poll=0
    async def set_laser_config(self, c): self.laser=c
    async def set_led(self, s): self.led=s
    async def set_temperature(self, t): self.setpoint=t
    async def get_temperature(self):
        self._poll += 1
        return {"state": "[ SYSTEM LOCKED ]" if self._poll >= 2 else "[ HEATING ]"}


class FakeOrch:
    def __init__(self, client): self._client=client; self._temperature_provider=lambda: None; self.events=[]
    @property
    def client(self): return self._client
    def _emit_event(self, et, data): self.events.append((et, data))


@pytest.mark.asyncio
async def test_phase_order_and_brightfield(monkeypatch):
    client = FakeClient(); orch = FakeOrch(client)
    bursts = []
    async def runner(b): bursts.append({"phase": getattr(b, "_phase", None), "laser": b._laser_config})
    res = await run_temp_change_burst_protocol(
        orch, "emb1", 25.0, frames=3, bursts_before=1, bursts_after=1,
        lock_timeout_s=5.0, poll_s=0.001, burst_runner=runner)
    assert client.laser == "ALL OFF" and client.led == "Open"
    assert client.setpoint == 25.0
    assert all(b["laser"] == "ALL OFF" for b in bursts)          # every burst brightfield
    assert len(bursts) >= 3                                       # before + >=1 during + after
    ets = [e[0] for e in orch.events]
    assert EventType.TEMP_PROTOCOL_STARTED in ets
    assert EventType.TEMPERATURE_SETPOINT_CHANGED in ets
    assert EventType.TEMP_PROTOCOL_COMPLETED in ets
    assert res["locked"] is True
