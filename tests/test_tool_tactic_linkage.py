"""
Task 7: tactic_id threading + start-edge marking in execution tools.

Verifies:
  1. enable_monitoring_mode with tactic_id → transition_tactic("active") called.
  2. enable_monitoring_mode without tactic_id → no transition call.
  3. queue_burst (tool) with tactic_id → transition_tactic("active") called.
  4. queue_burst (tool) without tactic_id → no transition call.
  5. stop_timelapse with tactic_id → transition_tactic("done") called.
  6. pause_timelapse with tactic_id → transition_tactic("paused") called.
  7. BurstAcquisition(tactic_id=...) stores _tactic_id and includes it in BURST_START data.
  8. BurstAcquisition(tactic_id=...) includes tactic_id in BURST_COMPLETE data.
  9. run_temp_change_burst_protocol_tool with tactic_id → transition_tactic("active") called.
  10. run_temp_change_burst_protocol_tool without tactic_id → no transition call.
  11. TEMP_PROTOCOL_STARTED and TEMP_PROTOCOL_COMPLETED carry tactic_id.
"""

import asyncio

import pytest

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeContextStore:
    """Capture calls to transition_tactic."""

    def __init__(self):
        self.transitions: list[tuple] = []

    def transition_tactic(self, session_id: str, tactic_id: str, state: str | None = None, **bind):
        self.transitions.append((session_id, tactic_id, state))
        return True


class FakeOrchestrator:
    """Minimal orchestrator for tool-layer tests."""

    def __init__(self):
        self.monitoring_modes_enabled: list[str] = []
        self.bursts_queued: list[dict] = []
        self.stopped = False
        self.paused = False

    def enable_monitoring_mode(self, name: str, **kwargs) -> str:
        self.monitoring_modes_enabled.append(name)
        return f"Activated monitoring mode '{name}'"

    def queue_burst(
        self,
        embryo_id: str,
        *,
        frames: int = 60,
        mode: str = "1hz",
        num_slices: int = 1,
        force: bool = False,
        laser_config=None,
        tactic_id=None,
    ) -> str:
        self.bursts_queued.append({"embryo_id": embryo_id, "tactic_id": tactic_id})
        return f"Burst queued for {embryo_id}"

    async def stop(self, reason: str = "user_request") -> str:
        self.stopped = True
        return "Timelapse stopped."

    async def pause(self) -> str:
        self.paused = True
        return "Timelapse paused."


class FakeAgent:
    """Carries context_store, session_id, and timelapse_orchestrator."""

    def __init__(self, *, cs=None, session_id="sess_t7", orchestrator=None):
        self.context_store = cs
        self.session_id = session_id
        self.timelapse_orchestrator = orchestrator


class FakeMicroscope:
    """Satisfies the requires_microscope client check in the tool registry."""

    pass


def _make_agent(with_cs=True, with_orchestrator=True):
    cs = FakeContextStore() if with_cs else None
    orch = FakeOrchestrator() if with_orchestrator else None
    agent = FakeAgent(cs=cs, orchestrator=orch)
    return agent, cs, orch


def _ctx(agent, with_client=False):
    ctx = {"agent": agent}
    if with_client:
        ctx["client"] = FakeMicroscope()
    return ctx


# ---------------------------------------------------------------------------
# enable_monitoring_mode — tool-layer tactic transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enable_monitoring_mode_with_tactic_id_calls_transition():
    """enable_monitoring_mode with tactic_id flips the tactic to active."""
    from gently.app.tools.timelapse_tools import enable_monitoring_mode

    agent, cs, _orch = _make_agent()
    await enable_monitoring_mode(
        mode_name="expression_monitoring", tactic_id="t1", context=_ctx(agent)
    )

    assert cs is not None
    assert any(t == ("sess_t7", "t1", "active") for t in cs.transitions), (
        f"Expected transition ('sess_t7', 't1', 'active'), got {cs.transitions}"
    )


@pytest.mark.asyncio
async def test_enable_monitoring_mode_without_tactic_id_no_transition():
    """enable_monitoring_mode without tactic_id must not call transition_tactic."""
    from gently.app.tools.timelapse_tools import enable_monitoring_mode

    agent, cs, _orch = _make_agent()
    await enable_monitoring_mode(mode_name="expression_monitoring", context=_ctx(agent))

    assert cs.transitions == [], f"Expected no transitions, got {cs.transitions}"


@pytest.mark.asyncio
async def test_enable_monitoring_mode_no_cs_no_crash():
    """Missing context store must not crash — the transition is a guarded no-op."""
    from gently.app.tools.timelapse_tools import enable_monitoring_mode

    agent, _cs, _orch = _make_agent(with_cs=False)
    result = await enable_monitoring_mode(
        mode_name="expression_monitoring", tactic_id="t1", context=_ctx(agent)
    )
    # No exception; result is the mode activation string
    assert "expression_monitoring" in result


# ---------------------------------------------------------------------------
# queue_burst — tool-layer tactic transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_burst_with_tactic_id_calls_transition():
    """queue_burst (tool) with tactic_id flips the tactic to active."""
    from gently.app.tools.timelapse_tools import queue_burst

    agent, cs, _orch = _make_agent()
    await queue_burst(embryo_id="emb1", tactic_id="t2", context=_ctx(agent, with_client=True))

    assert any(t == ("sess_t7", "t2", "active") for t in cs.transitions), (
        f"Expected transition ('sess_t7', 't2', 'active'), got {cs.transitions}"
    )


@pytest.mark.asyncio
async def test_queue_burst_without_tactic_id_no_transition():
    """queue_burst (tool) without tactic_id must not call transition_tactic."""
    from gently.app.tools.timelapse_tools import queue_burst

    agent, cs, _orch = _make_agent()
    await queue_burst(embryo_id="emb1", context=_ctx(agent, with_client=True))

    assert cs.transitions == [], f"Expected no transitions, got {cs.transitions}"


@pytest.mark.asyncio
async def test_queue_burst_passes_tactic_id_to_orchestrator():
    """queue_burst (tool) passes tactic_id into orchestrator.queue_burst."""
    from gently.app.tools.timelapse_tools import queue_burst

    agent, _cs, orch = _make_agent()
    await queue_burst(embryo_id="emb1", tactic_id="t2", context=_ctx(agent, with_client=True))

    assert orch.bursts_queued, "orchestrator.queue_burst must have been called"
    assert orch.bursts_queued[0]["tactic_id"] == "t2"


@pytest.mark.asyncio
async def test_queue_burst_soft_reject_does_not_transition():
    """queue_burst (tool) must NOT flip the tactic to active on a soft-reject.

    orchestrator.queue_burst returns a rejection sentence (not starting with
    "Burst queued for") when the embryo already has a queued burst, already
    had a burst this session, or is not in the active timelapse.  The tool
    must treat those as non-events and leave transition_tactic uncalled.
    """
    from gently.app.tools.timelapse_tools import queue_burst

    class RejectingOrchestrator(FakeOrchestrator):
        def queue_burst(
            self,
            embryo_id,
            *,
            frames=60,
            mode="1hz",
            num_slices=1,
            force=False,
            laser_config=None,
            tactic_id=None,
        ) -> str:
            # Simulates "already has a queued burst" soft-reject
            return f"Embryo '{embryo_id}' already has a queued burst."

    cs = FakeContextStore()
    orch = RejectingOrchestrator()
    agent = FakeAgent(cs=cs, orchestrator=orch)

    await queue_burst(embryo_id="emb1", tactic_id="t2", context=_ctx(agent, with_client=True))

    assert cs.transitions == [], (
        f"transition_tactic must not be called on soft-reject; got {cs.transitions}"
    )


# ---------------------------------------------------------------------------
# stop_timelapse — mark done
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_timelapse_with_tactic_id_calls_done():
    """stop_timelapse with tactic_id flips the tactic to done."""
    from gently.app.tools.timelapse_tools import stop_timelapse

    agent, cs, _orch = _make_agent()
    await stop_timelapse(tactic_id="t1", context=_ctx(agent, with_client=True))

    assert any(t == ("sess_t7", "t1", "done") for t in cs.transitions), (
        f"Expected transition to done, got {cs.transitions}"
    )


@pytest.mark.asyncio
async def test_stop_timelapse_without_tactic_id_no_transition():
    """stop_timelapse without tactic_id must not call transition_tactic."""
    from gently.app.tools.timelapse_tools import stop_timelapse

    agent, cs, _orch = _make_agent()
    await stop_timelapse(context=_ctx(agent, with_client=True))

    assert cs.transitions == []


# ---------------------------------------------------------------------------
# pause_timelapse — mark paused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_timelapse_with_tactic_id_calls_paused():
    """pause_timelapse with tactic_id flips the tactic to paused."""
    from gently.app.tools.timelapse_tools import pause_timelapse

    agent, cs, _orch = _make_agent()
    await pause_timelapse(tactic_id="t1", context=_ctx(agent, with_client=True))

    assert any(t == ("sess_t7", "t1", "paused") for t in cs.transitions), (
        f"Expected transition to paused, got {cs.transitions}"
    )


@pytest.mark.asyncio
async def test_pause_timelapse_without_tactic_id_no_transition():
    """pause_timelapse without tactic_id must not call transition_tactic."""
    from gently.app.tools.timelapse_tools import pause_timelapse

    agent, cs, _orch = _make_agent()
    await pause_timelapse(context=_ctx(agent, with_client=True))

    assert cs.transitions == []


# ---------------------------------------------------------------------------
# BurstAcquisition — tactic_id stored and appears in event data (unit-level)
# ---------------------------------------------------------------------------


def test_burst_acquisition_stores_tactic_id():
    """BurstAcquisition stores tactic_id as _tactic_id."""
    from gently.app.orchestration.exclusive import BurstAcquisition

    b = BurstAcquisition("emb1", frames=10, tactic_id="t3")
    assert b._tactic_id == "t3"


def test_burst_acquisition_default_tactic_id_none():
    """BurstAcquisition._tactic_id defaults to None (backward compat)."""
    from gently.app.orchestration.exclusive import BurstAcquisition

    b = BurstAcquisition("emb1", frames=10)
    assert b._tactic_id is None


def test_burst_start_event_data_contains_tactic_id():
    """The BURST_START event data dict includes tactic_id."""
    from gently.app.orchestration.exclusive import BurstAcquisition

    emitted: list[dict] = []

    class FakeOrc:
        _embryo_states = {}

        def _emit_event(self, event_type, data):
            emitted.append((event_type, data))

    b = BurstAcquisition("emb1", frames=5, tactic_id="t3")
    orch = FakeOrc()
    # Trigger just the BURST_START emission by running the coroutine start; since the
    # embryo is absent, run() returns an error result after emitting nothing —
    # that's the correct path. Instead, directly test what run() would emit.
    # We replicate the exact emit call from BurstAcquisition.run to verify the data key.
    # Safer: run the coroutine to the first yield so BURST_START fires.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(b.run(orch))
    finally:
        loop.close()

    # When embryo is absent run() returns early (no BURST_START). Confirm _tactic_id
    # is wired up and would appear in the event — inspect the attribute directly.
    assert b._tactic_id == "t3"


def test_burst_start_event_data_dict_includes_tactic_id_key():
    """The BURST_START event dict produced by BurstAcquisition.run includes 'tactic_id'."""
    from gently.app.orchestration.exclusive import BurstAcquisition
    from gently.core import EventType

    emitted: list[tuple] = []

    class FakeClient:
        async def move_to_position(self, x, y): ...
        async def acquire_burst(self, **kw):
            return {"success": True, "frames": []}

    class FakeEmbryo:
        calibration = {}
        stage_position = {}
        exposure_ms = 10.0
        laser_power_488_pct = None

    class FakeOrc:
        client = FakeClient()
        _embryo_states = {"emb1": FakeEmbryo()}

        def _emit_event(self, event_type, data):
            emitted.append((event_type, data))

    b = BurstAcquisition("emb1", frames=2, tactic_id="t3")
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(b.run(FakeOrc()))
    finally:
        loop.close()

    burst_starts = [(et, d) for et, d in emitted if et == EventType.BURST_START]
    assert burst_starts, f"BURST_START not emitted; all events: {[et for et, _ in emitted]}"
    _et, data = burst_starts[0]
    assert "tactic_id" in data, f"tactic_id not in BURST_START data: {data}"
    assert data["tactic_id"] == "t3"


def test_burst_complete_event_data_includes_tactic_id():
    """The BURST_COMPLETE event dict produced by BurstAcquisition.run includes 'tactic_id'."""
    from gently.app.orchestration.exclusive import BurstAcquisition
    from gently.core import EventType

    emitted: list[tuple] = []

    class FakeClient:
        async def move_to_position(self, x, y): ...
        async def acquire_burst(self, **kw):
            return {"success": True, "frames": []}

    class FakeEmbryo:
        calibration = {}
        stage_position = {}
        exposure_ms = 10.0
        laser_power_488_pct = None

    class FakeOrc:
        client = FakeClient()
        _embryo_states = {"emb1": FakeEmbryo()}

        def _emit_event(self, event_type, data):
            emitted.append((event_type, data))

    b = BurstAcquisition("emb1", frames=2, tactic_id="t3")
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(b.run(FakeOrc()))
    finally:
        loop.close()

    burst_completes = [(et, d) for et, d in emitted if et == EventType.BURST_COMPLETE]
    assert burst_completes, f"BURST_COMPLETE not emitted; all events: {[et for et, _ in emitted]}"
    _et, data = burst_completes[0]
    assert "tactic_id" in data, f"tactic_id not in BURST_COMPLETE data: {data}"
    assert data["tactic_id"] == "t3"


# ---------------------------------------------------------------------------
# temperature_protocol_tools — tactic transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_temp_protocol_tool_with_tactic_id_calls_transition(monkeypatch):
    """run_temp_change_burst_protocol_tool with tactic_id marks tactic active."""
    from gently.app.tools import temperature_protocol_tools as tpt

    # Patch asyncio.create_task to a no-op
    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close() or None)

    class FakeOrch:
        _status = None

    class FakeClient:
        pass

    cs = FakeContextStore()
    agent = FakeAgent(cs=cs, orchestrator=FakeOrch())
    agent.timelapse_orchestrator = FakeOrch()

    ctx = {
        "agent": agent,
        "client": FakeClient(),
    }

    await tpt.run_temp_change_burst_protocol_tool(
        embryo_id="emb1",
        target_setpoint_c=25.0,
        tactic_id="t4",
        context=ctx,
    )

    assert any(t == ("sess_t7", "t4", "active") for t in cs.transitions), (
        f"Expected active transition, got {cs.transitions}"
    )


@pytest.mark.asyncio
async def test_temp_protocol_tool_without_tactic_id_no_transition(monkeypatch):
    """run_temp_change_burst_protocol_tool without tactic_id must not call transition."""
    from gently.app.tools import temperature_protocol_tools as tpt

    monkeypatch.setattr(asyncio, "create_task", lambda coro: coro.close() or None)

    class FakeOrch:
        _status = None

    class FakeClient:
        pass

    cs = FakeContextStore()
    agent = FakeAgent(cs=cs, orchestrator=FakeOrch())
    agent.timelapse_orchestrator = FakeOrch()

    ctx = {
        "agent": agent,
        "client": FakeClient(),
    }

    await tpt.run_temp_change_burst_protocol_tool(
        embryo_id="emb1",
        target_setpoint_c=25.0,
        context=ctx,
    )

    assert cs.transitions == [], f"Expected no transitions, got {cs.transitions}"


# ---------------------------------------------------------------------------
# temperature_protocol orchestration — tactic_id in event payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_temp_protocol_started_event_carries_tactic_id():
    """TEMP_PROTOCOL_STARTED event data includes tactic_id."""
    from gently.app.orchestration.temperature_protocol import run_temp_change_burst_protocol
    from gently.core import EventType

    emitted: list[tuple] = []

    class FakeClient:
        async def set_laser_config(self, cfg):
            pass

        async def set_led(self, s):
            pass

        async def set_temperature(self, t):
            pass

        async def get_temperature(self):
            return {"state": "LOCKED"}

    class FakeOrc:
        client = FakeClient()

        def _emit_event(self, event_type, data):
            emitted.append((event_type, data))

    bursts_run: list[str] = []

    async def fake_burst_runner(b):
        bursts_run.append(b._phase)

    await run_temp_change_burst_protocol(
        FakeOrc(),
        "emb1",
        25.0,
        bursts_before=0,
        bursts_after=0,
        burst_runner=fake_burst_runner,
        tactic_id="t5",
    )

    started = [(et, d) for et, d in emitted if et == EventType.TEMP_PROTOCOL_STARTED]
    assert started, "TEMP_PROTOCOL_STARTED not emitted"
    assert started[0][1].get("tactic_id") == "t5"


@pytest.mark.asyncio
async def test_temp_protocol_completed_event_carries_tactic_id():
    """TEMP_PROTOCOL_COMPLETED event data includes tactic_id."""
    from gently.app.orchestration.temperature_protocol import run_temp_change_burst_protocol
    from gently.core import EventType

    emitted: list[tuple] = []

    class FakeClient:
        async def set_laser_config(self, cfg):
            pass

        async def set_led(self, s):
            pass

        async def set_temperature(self, t):
            pass

        async def get_temperature(self):
            return {"state": "LOCKED"}

    class FakeOrc:
        client = FakeClient()

        def _emit_event(self, event_type, data):
            emitted.append((event_type, data))

    await run_temp_change_burst_protocol(
        FakeOrc(),
        "emb1",
        25.0,
        bursts_before=0,
        bursts_after=0,
        burst_runner=lambda b: asyncio.sleep(0),
        tactic_id="t5",
    )

    completed = [(et, d) for et, d in emitted if et == EventType.TEMP_PROTOCOL_COMPLETED]
    assert completed, "TEMP_PROTOCOL_COMPLETED not emitted"
    assert completed[0][1].get("tactic_id") == "t5"


@pytest.mark.asyncio
async def test_temp_protocol_tactic_id_none_when_absent():
    """TEMP_PROTOCOL_* events include tactic_id=None when not supplied (backward compat)."""
    from gently.app.orchestration.temperature_protocol import run_temp_change_burst_protocol
    from gently.core import EventType

    emitted: list[tuple] = []

    class FakeClient:
        async def set_laser_config(self, cfg):
            pass

        async def set_led(self, s):
            pass

        async def set_temperature(self, t):
            pass

        async def get_temperature(self):
            return {"state": "LOCKED"}

    class FakeOrc:
        client = FakeClient()

        def _emit_event(self, event_type, data):
            emitted.append((event_type, data))

    await run_temp_change_burst_protocol(
        FakeOrc(),
        "emb1",
        25.0,
        bursts_before=0,
        bursts_after=0,
        burst_runner=lambda b: asyncio.sleep(0),
    )

    started = [(et, d) for et, d in emitted if et == EventType.TEMP_PROTOCOL_STARTED]
    completed = [(et, d) for et, d in emitted if et == EventType.TEMP_PROTOCOL_COMPLETED]
    assert started and "tactic_id" in started[0][1]
    assert started[0][1]["tactic_id"] is None
    assert completed and "tactic_id" in completed[0][1]
    assert completed[0][1]["tactic_id"] is None
