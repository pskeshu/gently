"""Connection B: the running agent's awareness summary carries the plan narrative
(goal + what's next), not just the active item's spec sheet."""

from gently.harness.memory.interface import AgentMemory


def test_awareness_includes_goal_and_next(file_context_store):
    cs = file_context_store
    cid = cs.create_campaign(
        description="Pioneer guidance", target="how pioneers steer the nerve ring"
    )
    active = cs.create_plan_item(
        campaign_id=cid,
        type="imaging",
        title="WT baseline",
        spec={"strain": "N2", "num_slices": 50, "laser_wavelength_nm": 488, "laser_power_pct": 8},
    )
    cs.create_plan_item(campaign_id=cid, type="decision_point", title="Go/no-go gate")

    mem = AgentMemory(cs, session_id="s1")
    mem.active_plan_item_id = active
    summary = mem.get_awareness_summary()

    assert "Goal of the investigation: how pioneers steer the nerve ring" in summary
    assert "Next up:" in summary
    assert "Go/no-go gate (decision point)" in summary
    # spec block still present, incl. laser power when the field is set
    assert "WT baseline" in summary
    assert "Laser: 488nm at 8%" in summary
