# Context Hierarchy Refactoring Guide

## Background: What Exists Today

Gently is a microscopy agent — an LLM-powered assistant for researchers running diSPIM light-sheet microscopy experiments. The system has two separate data layers:

1. **GentlyStore** (`gently/store.py`) — Raw experiment data: images, embryo positions, acquisition parameters, sessions. This is the "what happened" layer.

2. **ContextStore** (`gently/context/store.py`) — The agent's "mind": learnings, campaigns, session intents, observations, expectations, watchpoints, questions. This is the "what the agent knows/believes/wants" layer. Backed by SQLite at `<storage_dir>/context/agent_mind.db`.

The ContextStore has a rich hierarchical data model defined in `gently/context/model.py`:

```
Context (full working memory for one thinking cycle)
├── Intentions (why are we doing this?)
│   ├── Campaign        — long-running research goals (weeks/months)
│   ├── Project         — discrete work within a campaign
│   ├── PlannedSession  — calendar of future imaging sessions
│   └── SessionIntent   — what THIS session is about
├── Understanding (what do we believe?)
│   ├── Learning        — facts: "Lab organism: C. elegans"
│   └── EmbryoUnderstanding — per-embryo knowledge (stage, health, flags)
├── Observations        — synthesized notes about what happened
├── Expectations        — predictions: "embryo_3 will reach comma by 14:30"
└── Attention (what to watch?)
    ├── Watchpoint      — things to monitor: "alert when hatching starts"
    └── Question        — open research questions
```

There's also a `ContextUpdates` dataclass for batch writes (new observations, resolved expectations, triggered watchpoints, etc.).

### The Startup Wizard

A startup wizard (`gently/context/startup_wizard.py`) runs at startup between the WebSocket connect and the REPL loop. It uses gap assessment (`gently/context/gap_assessment.py`) to determine what context is missing, then conversationally fills gaps through choice pickers:

- **First launch**: Asks organism → research campaign → session intent
- **Returning user**: Campaign selection → session intent
- **Fully oriented**: Wizard skipped

The wizard writes to the ContextStore: `Learning` entries for lab identity, `Campaign` entries for research programs, `SessionIntent` for today's plan. It uses LLM extraction (`gently/context/onboarding.py`) for free-text responses.

### The System Prompt

The agent's system prompt is built by `build_system_prompt()` in `gently/agent/prompts.py`:

```python
def build_system_prompt(
    experiment_state: ExperimentState,
    connection_status: dict = None,
    context_summary: str = None,    # AI-generated timelapse/event summary
) -> str:
```

It assembles:
1. Role definition (hardcodes "C. elegans embryos")
2. Hardware connection status (online/offline)
3. `CELEGANS_BIOLOGY` — ~2K tokens of C. elegans developmental staging (hardcoded)
4. `DISPIM_HARDWARE` — diSPIM hardware specs (hardcoded)
5. `CV_SUBAGENT` — vision capabilities
6. `ADAPTIVE_TIMELAPSE` — timelapse system description
7. `USER_INTERACTION_GUIDELINES` — behavioral rules
8. `SESSION_MANAGEMENT` — session handling
9. Current experiment state from `ExperimentState.get_summary()`
10. Optional `context_summary` — AI-generated summary of timelapse status and recent events
11. Tool use guidelines and behavior rules

The `context_summary` parameter is populated by `_get_cached_context_summary()` in `agent.py`, which calls `_gather_context_data()` to collect timelapse status, timeline events, and detection results, then asks Haiku to summarize. This is cached for 5 minutes.

The agent is instantiated in `agent.py`:
```python
class MicroscopyAgent:
    def __init__(self, api_key, storage_path, model, microscope_client, session_id, store: GentlyStore):
```

The `_update_system_prompt()` method (line 264) rebuilds the prompt and is called before each API call.

### The Wiring Gap

In `launch_agent.py` (line 203-206):
```python
context_store = CtxStore(context_db)
bridge.init_wizard(context_store=context_store, claude_client=agent.claude)
```

The ContextStore is created and passed to the wizard, but **NOT to the agent**. The agent has no reference to the ContextStore. It cannot read learnings, campaigns, or session intents. The wizard writes to the store, but the data is never surfaced in the system prompt.

**Result**: When the user asks "what's the session context?", the agent only knows about hardware status and experiment state. It doesn't know the researcher works with C. elegans, their research campaign, or their session plan — even though the wizard just collected all of that.

---

## The Problem: Two Disconnects

### Disconnect 1: Context Store → System Prompt (read path)

`load_active()` returns a `Context` object described as "what gets passed to the agent each thinking cycle." But nothing calls it. The agent's system prompt is built from `ExperimentState` (hardware/embryo data) and a timelapse summary — never from the ContextStore.

### Disconnect 2: System Prompt ← Agent (write path)

`ContextUpdates` is designed as "what the agent returns after thinking" — new observations, expectations, triggered watchpoints. But nothing processes it. The agent doesn't write observations, doesn't set expectations, doesn't track watchpoints. The context store only grows during onboarding.

### Disconnect 3: Static prompt vs dynamic identity

The system prompt hardcodes "C. elegans embryos" and includes a C. elegans staging guide regardless of organism. But the wizard asks "what organism do you work with?" — the answer should drive what knowledge is loaded. A zebrafish researcher shouldn't get a C. elegans developmental timing table.

---

## The Four-Phase Refactoring

### Phase 1: Read Path (inject context into prompt) — DO THIS FIRST

Wire `load_active()` into the system prompt so the agent knows what the wizard collected.

**Files to modify:**

1. **`gently/context/store.py`** — Add `build_context_preamble(session_id: str) -> str`
   - Calls `load_active()` to get the full `Context`
   - Formats learnings, campaigns, session intent, watchpoints, questions as a text block
   - Returns empty string if nothing stored (no noise in prompt)
   - Example output:
     ```
     # Researcher Context

     ## Lab Identity
     - Lab organism: C. elegans
     - Imaging system: diSPIM light-sheet

     ## Active Campaign
     Embryonic cell lineage tracking — monitoring division timing in early development
     Progress: 23/50 hatching events captured

     ## Session Intent
     Run a timelapse of embryo development

     ## Watchpoints
     - Alert when any embryo approaches hatching

     ## Open Questions
     - Why does batch 7 develop 20% faster than batch 6?
     ```

2. **`gently/agent/prompts.py`** — Add `researcher_context` parameter to `build_system_prompt()`
   - Insert between connection status and biology sections (the "who you're working with" before "what you know about science")
   - Only include if non-empty

3. **`gently/agent/agent.py`** — Give agent access to ContextStore
   - Add `self.context_store: Optional[ContextStore] = None` in `__init__` (after line 96)
   - Add `set_context_store(context_store: ContextStore)` method
   - In `_update_system_prompt()` (line 264): call `self.context_store.build_context_preamble(self.session_id)` and pass as `researcher_context`

4. **`launch_agent.py`** — Connect the dots (one line after line 206):
   ```python
   agent.set_context_store(context_store)
   ```

5. **`gently/visualization/routes/agent_ws.py`** — After wizard completes (before REPL loop), call `agent._update_system_prompt()` so the prompt picks up what the wizard just stored. Find the spot after `_run_wizard()` returns and before the main message loop.

**Verification**: `/reset-context` → restart → wizard collects organism + campaign + intent → ask "what's the session context?" → agent should now mention organism, campaign, and session plan.

### Phase 2: Knowledge Modules (context-driven prompt composition)

Replace hardcoded biology with organism-specific modules.

1. Create `gently/agent/knowledge/` directory with modules:
   - `celegans.py` — current `CELEGANS_BIOLOGY` content
   - `zebrafish.py` — zebrafish developmental staging
   - `drosophila.py` — Drosophila embryogenesis
   - `generic.py` — generic microscopy guidance (fallback)

2. In `prompts.py`, `build_system_prompt()` selects the biology module based on organism from researcher context:
   ```python
   organism = extract_organism_from_context(researcher_context)
   biology = load_knowledge_module(organism)  # Returns text or generic fallback
   ```

3. The role definition line changes from "C. elegans embryos" to reference the actual organism.

4. Future: LLM-generate a biology primer for novel organisms at onboarding time, cache as a learning in the ContextStore.

### Phase 3: Write Path (agent updates context)

Let the agent write back to the context store after each interaction.

1. After each agent response (or after significant tool calls), run a lightweight extraction:
   - "Based on this interaction, did you learn anything new? Make any observations? Form any expectations?"
   - Use the existing `ContextUpdates` dataclass

2. Add `apply_updates(updates: ContextUpdates)` to `store.py`

3. Trigger points:
   - After image analysis (new observations, stage updates)
   - After timelapse check-ins (expectations confirmed/surprised)
   - After researcher provides new information (learnings)

### Phase 4: Attention Loop (watchpoints and expectations)

The agent actively monitors its context:

1. After each observation, check active watchpoints — trigger if condition met
2. Compare observations against pending expectations — mark confirmed/surprised
3. Update campaign progress based on observations
4. Proactive alerts: if a watchpoint triggers, notify the researcher

---

## Key File Reference

| File | Role | Key Functions |
|------|------|---------------|
| `gently/context/model.py` | Data model — Context, ContextUpdates, all entity types | `Context`, `ContextUpdates` dataclasses |
| `gently/context/store.py` | SQLite CRUD for context entities | `load_active()`, `get_learnings()`, `get_active_campaigns()`, `get_current_session_intent()` |
| `gently/context/gap_assessment.py` | Inspects store, identifies missing context | `assess_gaps()` → `ContextGapReport` |
| `gently/context/startup_wizard.py` | Conversational onboarding at startup | `StartupWizard.run()` — writes Learnings, Campaigns, SessionIntent |
| `gently/context/onboarding.py` | LLM extraction from free-text responses | `process_onboarding_response()` |
| `gently/agent/prompts.py` | System prompt construction | `build_system_prompt()` (line 370) |
| `gently/agent/agent.py` | Main agent class | `_update_system_prompt()` (line 264), `_gather_context_data()` (line 287) |
| `gently/agent/agent_bridge.py` | WebSocket adapter, command handling | `init_wizard()`, `handle_command()` |
| `gently/visualization/routes/agent_ws.py` | WebSocket route, wizard loop, REPL loop | `_run_wizard()`, main message loop |
| `launch_agent.py` | Entry point — creates agent, store, bridge, TUI | Lines 203-206: ContextStore creation |

## Current State (as of this commit)

- Startup wizard: fully working, collects organism/campaign/intent via choice pickers
- ContextStore: has data after wizard runs, but agent never reads it
- System prompt: hardcodes C. elegans biology and diSPIM hardware
- Choice pickers: auto-append "Something else..." with inline text input (general TUI feature)
- Commands: `/reset-context` clears the store, `/wizard` re-runs the wizard
- Phase 1 (read path) is the immediate next step
