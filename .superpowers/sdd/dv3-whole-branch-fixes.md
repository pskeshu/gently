# dv3 whole-branch review fixes — backend half

Agent: backend-fixes agent (this session).
Commit: see footer.
Frontend fixes (experiment-overview.js + css): separate agent, not touched here.

---

## Fix #2 — agent prompt guidance to declare the Operation Plan

**File:** `gently/harness/prompts/templates.py`

Added constant `OPERATION_PLAN_GUIDANCE` (~line 225) and injected `{OPERATION_PLAN_GUIDANCE}`
into the system template f-string right after `{REACTIVE_MONITORING_MODES}` (~line 510).
The block tells the agent to call `declare_operation_plan` at experiment planning time
with kind/name/target/scope/rationale + a `live` object (`readouts`/`phases` + flat keys),
and to re-call (patch) on each tactic transition. Also notes that execution tools
(`queue_burst`, `enable_monitoring_mode`, etc.) accept `tactic_id` and flip state automatically.

---

## Fix #3 — wire operation-plan seeding in exit_plan_mode

**File:** `gently/app/agent.py`, `exit_plan_mode()` (~line 475)

Hook site confirmed: inside `if active_id:` block, right after the `link_session_campaign`
try/except (line ~464-469). At that point `self.context_store` and `self.session_id` are
guaranteed valid (context_store is used unguarded at `get_plan_item` one line above).

Added:
```python
try:
    from gently.app.tools.operation_plan_seed import seed_operation_plan_from_plan_item
    seed_operation_plan_from_plan_item(self.context_store, self.session_id)
except Exception:
    logger.exception("operation-plan seeding failed")
```

---

## Fix #4 — queue_burst phantom-active on soft-reject

**File:** `gently/app/tools/timelapse_tools.py` (~line 1204)

`orchestrator.queue_burst` returns `"Burst queued for {embryo_id} (request_id=..., ...)"` on
success and a rejection sentence (embryo not in timelapse / already had burst / already queued)
on soft-reject. Success is reliably distinguishable by `result.startswith("Burst queued for ")`.

Guard applied:
```python
if tactic_id and isinstance(result, str) and result.startswith("Burst queued for "):
    cs.transition_tactic(session, tactic_id, "active")
```

New test added to `tests/test_tool_tactic_linkage.py`:
`test_queue_burst_soft_reject_does_not_transition` — fake orchestrator returns
`"Embryo 'emb1' already has a queued burst."` → verifies `cs.transitions == []`.

---

## Fix #5 — `paused` valid state

**File:** `gently/app/tools/operation_plan_tools.py`, line 29

`_VALID_STATES` extended: `frozenset({"planned", "active", "done", "paused"})`.
`pause_timelapse` transitions to `"paused"` — this was previously an invalid state
causing a ValueError on validation.

---

## Fix #7 — document `live` in the spec

**File:** `docs/superpowers/specs/2026-06-28-operations-tab-design.md`, §1

Added `live` field to the tactic schema block with `readouts`/`phases` sub-keys and
a documentation paragraph explaining that the agent authors `readouts`/`phases` at
declaration time and the updater merges flat bound keys (`request_id`, `sustained_hz`,
`setpoint`, `locked`, `last_fired`, `new_phase`, …) as telemetry arrives.

---

## Fix #8 — transition_tactic single-loop note

**File:** `gently/harness/memory/file_store.py`, `transition_tactic` docstring (~line 847)

Added one-line note: "read-modify-write with no lock; safe because all subscribed event
emissions run on the single asyncio loop thread — revisit if a worker-thread emitter
is ever added."

---

## Test results

```
pytest tests/test_operation_plan_tool.py tests/test_tool_tactic_linkage.py \
       tests/test_operation_plan_seeding.py tests/test_operation_plan_store.py -v
65 passed, 0 failed
```

Full suite (excluding 3 pre-existing collection errors in test_campaign_coordination.py,
test_gently_store.py, test_text_tool_call_extraction.py):
- Before changes: 32 failed, 673 passed, 19 skipped
- After changes:  32 failed, 674 passed, 19 skipped  (+1 new test, 0 new failures)

---

## Commit

`fix(operations): wire seeding + agent plan guidance + queue_burst guard + paused state + docs`
