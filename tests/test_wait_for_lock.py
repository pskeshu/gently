from gently.app.orchestration.temperature_protocol import wait_for_temperature_lock


class FakeClient:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    async def get_temperature(self):
        i = min(self.calls, len(self.states) - 1)
        self.calls += 1
        return {"state": self.states[i]}


async def test_returns_true_when_locked():
    c = FakeClient(["[ IDLE ]", "[ HEATING ]", "[ SYSTEM LOCKED ]"])
    assert await wait_for_temperature_lock(c, timeout_s=5.0, poll_s=0.001) is True


async def test_returns_false_on_timeout():
    c = FakeClient(["[ HEATING ]"])
    assert await wait_for_temperature_lock(c, timeout_s=0.02, poll_s=0.001) is False
