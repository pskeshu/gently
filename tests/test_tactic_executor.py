"""
Tests for app.orchestration.tactic_executor — the deterministic mapping from a
declarative tactic to orchestrator actions (the first caller of
resolve_scope_embryos).

Covers: scope resolution (embryos/role/global/empty), kind dispatch
(standing_timelapse + monitoring, reactive_monitor, exclusive_burst, oneshot),
unknown-kind handling, and the transition_tactic 'active' side effect.
"""

import types

from gently.app.orchestration.tactic_executor import execute_tactic


class _Emb:
    def __init__(self, role):
        self.role = role


class _Orch:
    def __init__(self):
        self.start_calls = []
        self.monitor_calls = []
        self.burst_calls = []

    async def start(
        self,
        embryo_ids=None,
        stop_condition="manual",
        base_interval_seconds=120.0,
        condition_value=None,
    ):
        self.start_calls.append(
            {
                "embryo_ids": embryo_ids,
                "stop_condition": stop_condition,
                "interval": base_interval_seconds,
                "condition_value": condition_value,
            }
        )
        return "started"

    def enable_monitoring_mode(self, name, embryo_ids=None, **kw):
        self.monitor_calls.append({"name": name, "embryo_ids": embryo_ids})
        return f"monitor:{name}"

    def queue_burst(self, embryo_id, frames=60, mode="1hz", num_slices=1, tactic_id=None, **kw):
        self.burst_calls.append({"embryo_id": embryo_id, "frames": frames, "tactic_id": tactic_id})
        return f"Burst queued for {embryo_id}"


class _CS:
    def __init__(self):
        self.transitions = []

    def transition_tactic(self, session_id, tactic_id, state=None, **bind):
        self.transitions.append((tactic_id, state))
        return True


def _agent(embryos):
    exp = types.SimpleNamespace(embryos=embryos)
    return types.SimpleNamespace(
        experiment=exp,
        timelapse_orchestrator=_Orch(),
        context_store=_CS(),
        session_id="sess1",
    )


def _roster3():
    return {"embryo_1": _Emb("test"), "embryo_2": _Emb("test"), "embryo_3": _Emb("calibration")}


async def test_standing_timelapse_starts_scoped():
    agent = _agent(_roster3())
    t = {
        "id": "t1",
        "name": "TL",
        "kind": "standing_timelapse",
        "state": "planned",
        "scope": {"mode": "embryos", "embryo_ids": ["embryo_1", "embryo_2"]},
        "structure": {"cadence_s": 90, "stop_condition": "manual", "monitoring_mode": "idle"},
    }
    res = await execute_tactic(agent, t)
    assert res["ok"] is True
    assert res["embryo_ids"] == ["embryo_1", "embryo_2"]
    sc = agent.timelapse_orchestrator.start_calls
    assert len(sc) == 1
    assert sc[0]["embryo_ids"] == ["embryo_1", "embryo_2"] and sc[0]["interval"] == 90
    # idle → no monitoring installed
    assert agent.timelapse_orchestrator.monitor_calls == []
    # tactic marked active
    assert ("t1", "active") in agent.context_store.transitions


async def test_standing_timelapse_with_monitoring():
    agent = _agent(_roster3())
    t = {
        "id": "t2",
        "name": "TL",
        "kind": "standing_timelapse",
        "state": "planned",
        "scope": {"mode": "role", "role": "test"},
        "structure": {"interval": 120, "monitoring_mode": "expression_monitoring"},
    }
    res = await execute_tactic(agent, t)
    assert res["ok"] is True
    # role scope → the two 'test' embryos
    assert sorted(res["embryo_ids"]) == ["embryo_1", "embryo_2"]
    assert agent.timelapse_orchestrator.monitor_calls[0]["name"] == "expression_monitoring"


async def test_reactive_monitor():
    agent = _agent(_roster3())
    t = {
        "id": "t3",
        "name": "M",
        "kind": "reactive_monitor",
        "state": "planned",
        "scope": {"mode": "global"},
        "structure": {"monitoring_mode": "pre_terminal_monitoring"},
    }
    res = await execute_tactic(agent, t)
    assert res["ok"] is True
    assert sorted(res["embryo_ids"]) == ["embryo_1", "embryo_2", "embryo_3"]  # global = all
    assert agent.timelapse_orchestrator.monitor_calls[0]["name"] == "pre_terminal_monitoring"


async def test_exclusive_burst_per_embryo():
    agent = _agent(_roster3())
    t = {
        "id": "t4",
        "name": "B",
        "kind": "exclusive_burst",
        "state": "planned",
        "scope": {"mode": "embryos", "embryo_ids": ["embryo_1", "embryo_3"]},
        "structure": {"frames": 30},
    }
    res = await execute_tactic(agent, t)
    assert res["ok"] is True
    bc = agent.timelapse_orchestrator.burst_calls
    assert [b["embryo_id"] for b in bc] == ["embryo_1", "embryo_3"]
    assert all(b["frames"] == 30 and b["tactic_id"] == "t4" for b in bc)


async def test_oneshot_recorded_even_without_scope():
    agent = _agent(_roster3())
    t = {
        "id": "t5",
        "name": "one",
        "kind": "oneshot",
        "state": "planned",
        "scope": {"mode": "embryos", "embryo_ids": []},
    }
    res = await execute_tactic(agent, t)
    assert res["ok"] is True  # oneshot tolerated with empty scope
    assert agent.timelapse_orchestrator.start_calls == []


async def test_empty_scope_fails_for_timelapse():
    agent = _agent(_roster3())
    t = {
        "id": "t6",
        "name": "TL",
        "kind": "standing_timelapse",
        "state": "planned",
        "scope": {"mode": "embryos", "embryo_ids": []},
        "structure": {},
    }
    res = await execute_tactic(agent, t)
    assert res["ok"] is False
    assert "no embryos" in res["message"]
    assert agent.timelapse_orchestrator.start_calls == []


async def test_unknown_kind():
    agent = _agent(_roster3())
    t = {
        "id": "t7",
        "name": "x",
        "kind": "warp_drive",
        "state": "planned",
        "scope": {"mode": "global"},
    }
    res = await execute_tactic(agent, t)
    assert res["ok"] is False
    assert "unknown tactic kind" in res["message"]


async def test_no_orchestrator():
    agent = types.SimpleNamespace(
        experiment=types.SimpleNamespace(embryos=_roster3()),
        timelapse_orchestrator=None,
        context_store=_CS(),
        session_id="s",
    )
    res = await execute_tactic(
        agent,
        {"id": "t", "kind": "standing_timelapse", "scope": {"mode": "global"}, "structure": {}},
    )
    assert res["ok"] is False and res["message"] == "no orchestrator"
