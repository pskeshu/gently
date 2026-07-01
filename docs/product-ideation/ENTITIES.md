# Gently — Entity Inventory + Matrices

30 domain entities the UI exposes, with where they surface and which operations exist today.

## Entity × Operation matrix

✓ = affordance exists in the UI · (blank) = candidate missing-affordance.

| Entity | view | create | edit | delete | link | export |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Session | ✓ |   |   |   | ✓ |   |
| Embryo | ✓ | ✓ | ✓ | ✓ | ✓ |   |
| Volume/Image | ✓ | ✓ |   |   |   |   |
| Projection | ✓ |   |   |   |   |   |
| Prediction/Stage | ✓ |   |   |   |   |   |
| Trace | ✓ |   |   |   |   |   |
| Ground truth |   |   |   |   |   |   |
| Campaign | ✓ |   |   |   | ✓ | ✓ |
| Plan item | ✓ |   | ✓ |   | ✓ |   |
| Plan item dependency | ✓ |   |   |   |   |   |
| Planned session | ✓ |   |   |   |   |   |
| Operation Plan | ✓ |   |   |   |   |   |
| Tactic | ✓ |   |   |   |   |   |
| Tactic library | ✓ |   |   |   |   |   |
| Notebook note | ✓ |   |   |   | ✓ |   |
| Learning | ✓ |   |   |   |   |   |
| Observation | ✓ |   |   |   |   |   |
| Question | ✓ |   |   |   | ✓ |   |
| Watchpoint | ✓ |   |   |   |   |   |
| Expectation | ✓ |   |   |   |   |   |
| Role | ✓ |   |   |   | ✓ |   |
| Setpoint (temperature) | ✓ |   | ✓ |   |   |   |
| Temperature sample/graph | ✓ |   |   |   |   |   |
| Device state | ✓ |   |   |   |   |   |
| Agent chat / turn | ✓ | ✓ |   |   |   |   |
| Ask (agent → human) | ✓ |   |   |   |   |   |
| Event / log | ✓ |   |   |   |   |   |
| Config / dashboard prefs | ✓ |   | ✓ |   |   | ✓ |
| Mesh / peer instance | ✓ |   |   |   |   |   |
| Auth / control | ✓ |   |   |   |   |   |

## Entity × Entity — related but NOT linked in the UI

Pairs worth connecting (value med/high) with no traversable UI path today — the cross-feature backlog.

| A | ↔ | B | value |
|---|:--:|---|:--:|
| Session | ↔ | Notebook note | high |
| Embryo | ↔ | Ground truth | high |
| Embryo | ↔ | Watchpoint | high |
| Embryo | ↔ | Expectation | high |
| Embryo | ↔ | Notebook note | high |
| Embryo | ↔ | Tactic | high |
| Prediction/Stage | ↔ | Ground truth | high |
| Prediction/Stage | ↔ | Expectation | high |
| Prediction/Stage | ↔ | Tactic | high |
| Campaign | ↔ | Learning | high |
| Campaign | ↔ | Notebook note | high |
| Plan item | ↔ | Notebook note | high |
| Operation Plan | ↔ | Learning | high |
| Tactic | ↔ | Learning | high |
| Session | ↔ | Campaign | med |
| Session | ↔ | Planned session | med |
| Embryo | ↔ | Question | med |
| Campaign | ↔ | Tactic | med |
| Campaign | ↔ | Mesh / peer instance | med |
| Plan item | ↔ | Tactic | med |
| Tactic | ↔ | Role | med |
| Learning | ↔ | Observation | med |

## Inventory

### Session
A single run of the microscope (folder under sessions/, session.yaml + lock + timeline). The unit that owns embryos, volumes, events, and temperature. Created by starting a run; carries an intent (planned vs actual).
- **Surfaced in:** Home > Recent Sessions, Sessions tab, header session-id badge/link (#session-id-link), landing > Resume, Logs tab (its events)
- **Ops today:** view, list, act(resume via POST /api/sessions/{id}/resume), act(copy id / touch), link(to campaign, only via a plan-item inspector)
- **Relates to:** Embryo, Volume/Image, Operation Plan, Campaign, Plan item, Session intent, Event/log, Temperature sample, Notebook note

### Embryo
A marked/tracked C. elegans embryo with position, calibration, uid; the subject of perception. Detected on the bottom cam (SAM) then confirmed into the single worklist ('THE PLAN').
- **Surfaced in:** Embryos tab (Default/Board/Film/Vitals), Operate view worklist (#op-board 'THE PLAN'), header embryo count, Notebook note chips, Gallery filter 'All embryos'
- **Ops today:** view, list, create(detect via POST /api/devices/detect_embryos + confirm /api/devices/embryos/confirm), edit(position PUT /api/embryos/{id}/position), delete(DELETE /api/embryos/{id}), act(mark, center /api/devices/stage/move), link(assign role POST /api/embryos/roles)
- **Relates to:** Session, Volume/Image, Prediction/Stage, Trace, Role, Embryo understanding, Ground truth, Watchpoint, Notebook note

### Volume/Image
An acquired 3D stack (volumes/t{NNNN}.tif + meta) plus its 2D projection (jpg) and standalone snapshots. The raw imaging payload.
- **Surfaced in:** Gallery tab (type filters Volume/Projection/Snapshot), Devices > 3D view, Home > Recent Images, timepoint player / lightbox, Calibration > Gallery
- **Ops today:** view, create(acquire POST /api/devices/acquire/volume, /acquire/burst), act(3D render /api/volumes3d, slice, raw /api/volume-raw), list(/api/volumes,/api/snapshots)
- **Relates to:** Embryo, Session, Projection, Prediction/Stage, Trace

### Projection
2D max/summary projection derived from a volume, the default thumbnail shown for a timepoint.
- **Surfaced in:** Gallery tab, Embryos > Default ('View All Projections'), projection-viewer, Sessions (session projection)
- **Ops today:** view, act(view-all / paginate via projection-viewer)
- **Relates to:** Volume/Image, Embryo, Prediction/Stage

### Prediction/Stage
Per-timepoint developmental-stage call for an embryo (predicted_stage, confidence, is_transitional) appended to predictions.jsonl. The core perception output.
- **Surfaced in:** Embryos > Board (stage sparkline column), Embryos > Vitals (stage strip chart over timepoints), Embryos > Default (detection cards)
- **Ops today:** view, act(agree/disagree — saved only to localStorage, not persisted)
- **Relates to:** Embryo, Trace, Ground truth, Expectation, Volume/Image

### Trace
The complete perception record for a timepoint (traces/t{NNNN}.json) — classifier/perceiver VLM reasoning and observed features behind a prediction.
- **Surfaced in:** Embryos > Default ('Show VLM reasoning', raw trace JSON)
- **Ops today:** view
- **Relates to:** Prediction/Stage, Embryo, Volume/Image

### Ground truth
Human-annotated correct stage for an embryo/timepoint (ground_truth.yaml; FileStore.set_ground_truth/get_ground_truth). Exists in storage but has NO authoring UI — the only feedback is binary Agree/Disagree to localStorage.
- **Surfaced in:** (no dedicated surface — backend store only; would live on Embryos detection cards)
- **Ops today:** —
- **Relates to:** Prediction/Stage, Embryo

### Campaign
A long-running research program with a description, target, status and hierarchy; the container for phases/plan items and linked sessions. Created only by the agent (create_campaign tool), never from the UI.
- **Surfaced in:** Plans tab (campaign navigator + canvas + inspector; Doc/Graph/Board/Decide/Matrix/Timeline), Home > Recent Plans, landing plan-wizard ('Continue …' / 'Start new campaign')
- **Ops today:** view, list(/api/campaigns), act(tree /tree, document /document, versions /versions), link(session via item inspector /items/{id}/sessions)
- **Relates to:** Plan item, Session, Planned session, Tactic, Learning, Plan snapshot/version

### Plan item
A single unit of work inside a campaign (imaging/bench/genetics/analysis/decision_point) with title, status, dependencies, references, and an ImagingSpec. Structural edits (add/remove/reorder/deps) are agent-only.
- **Surfaced in:** Plans tab > Doc (item rows, '→ blocks:'), Plans > Graph/Board/Matrix/Timeline/Decide, item inspector
- **Ops today:** view, edit(inline field edits in inspector), link(session link/delink), act(resolve imaging spec)
- **Relates to:** Campaign, Session, Planned session, ImagingSpec, Plan item dependency

### Plan item dependency
Blocks / blocked-by edges between plan items driving readiness (get_unblocked_plan_items).
- **Surfaced in:** Plans > Doc ('→ blocks: …'), Plans > Graph, Plans > Decide
- **Ops today:** view
- **Relates to:** Plan item, Campaign

### Planned session
A scheduled future session (title, date/time, estimated duration, acquisition params inherited from a source session) that becomes an actual Session when started.
- **Surfaced in:** Plans tab (campaign planned-sessions /api/campaigns/{id}/planned-sessions), plan-item inspector
- **Ops today:** view, list
- **Relates to:** Campaign, Plan item, Session, ImagingSpec

### Operation Plan
The live per-session tactic spine (operation_plans/{session}.yaml) — the ordered set of tactics with states (done/in-use/queued) that the agent is executing right now.
- **Surfaced in:** Operations tab > Overview (title + tactic spine)
- **Ops today:** view
- **Relates to:** Session, Tactic, Embryo, Setpoint (temperature)

### Tactic
An imaging behavior/protocol card (monitor, transmission burst, temp-change burst, recovery monitor …) with rationale/scope/cadence; the executable step of an Operation Plan and the reusable unit saved to the library.
- **Surfaced in:** Operations > Overview (expandable tactic cards, US-17), Operate > Run chooser ('From library', 'Continue a plan', 'Hand to agent'), tactic library
- **Ops today:** view, act(expand card), act(run POST /api/operate/run-tactic), act(apply saved tactic /api/tactic_library)
- **Relates to:** Operation Plan, Embryo (scope), Role, Setpoint (temperature), Campaign

### Tactic library
Saved reusable tactics (agent/ml store) that can be instantiated into a run.
- **Surfaced in:** Operate > Run chooser ('From library — a saved tactic')
- **Ops today:** view, list(/api/tactic_library), act(apply into run)
- **Relates to:** Tactic, Operation Plan

### Notebook note
The unified shared-lab-notebook entry: kind (observation/finding/question), author (human/agent), status (proposed/confirmed/open), with strain/embryo/session/thread/basis links. Read-only in the UI — there is NO 'add note' control.
- **Surfaced in:** Notebook tab (kind filters, thread rail, Ask box), Home / Agent chat 'AGENT'S VIEW > From the notebook', context-surface 'Agent's view'
- **Ops today:** view, list(/api/notebook/notes), filter(kind/author/status/strain/embryo/thread), act(ask /api/notebook/ask), link(threads /api/notebook/threads)
- **Relates to:** Learning, Observation, Question, Embryo, Session, Campaign, Plan item

### Learning
A durable agent insight (content, confidence, basis) accumulated in memory (learnings/*.yaml). Surfaces in the notebook as FINDING-kind notes.
- **Surfaced in:** Notebook tab > Findings filter
- **Ops today:** view
- **Relates to:** Observation, Notebook note, Embryo, Campaign

### Observation
A recorded observation (stage_transition/anomaly/session_summary/milestone) with significance and gently_refs. Surfaces as OBSERVATION-kind notes.
- **Surfaced in:** Notebook tab > Observations filter
- **Ops today:** view
- **Relates to:** Notebook note, Embryo, Session, Learning

### Question
An open question capturing agent uncertainty (or a human-posed one). Resolvable only by the control holder.
- **Surfaced in:** Notebook tab > Questions filter, context-surface 'Open questions'
- **Ops today:** view, act(resolve/answer POST /api/context/questions/{id}/resolve — control), link(thread)
- **Relates to:** Notebook note, Embryo, Watchpoint, Expectation

### Watchpoint
An active attention target (embryo + condition, e.g. 'approaching hatching') with priority. No dedicated tab — lives only in the always-on agent's-view surface.
- **Surfaced in:** context-surface 'Watching'
- **Ops today:** view, act(resolve POST /api/context/watchpoints/{id}/resolve — control)
- **Relates to:** Embryo, Question, Expectation

### Expectation
An agent belief about the future ('will reach comma stage' by expected_time, with uncertainty/basis) — the agent's forward prediction, distinct from a per-timepoint stage Prediction.
- **Surfaced in:** context-surface 'Expectations'
- **Ops today:** view, act(confirm POST /api/context/expectations/{id}/resolve — control)
- **Relates to:** Embryo, Prediction/Stage, Watchpoint

### Role
An embryo role from the static registry (unassigned/test/calibration/lineaging/subject/reference…) with role_class (subject vs reference), default cadence, and detector; governs how Operations foregrounds and images an embryo.
- **Surfaced in:** Operate > Run chooser ('ROLES (marked → subject)' chips), roles registry /api/roles
- **Ops today:** view, list, act(assign toggle subject/reference)
- **Relates to:** Embryo, Tactic, Cadence

### Setpoint (temperature)
A temperature target for the stage/thermalizer (ACUITYnano). Set from Devices, changed inside a temp-change tactic, and configured (serial/MQTT) in Settings.
- **Surfaced in:** Devices header temp control (input + Set; hidden until controller online), Operations tactic card ('→ 32.0 °C', 'setpoint change'), Settings > Hardware > Thermalizer
- **Ops today:** view, act(set POST /api/devices/temperature/set — control), edit(config /api/devices/temperature/config), act(test connection /config/test)
- **Relates to:** Temperature sample/graph, Session, Tactic, Device state

### Temperature sample/graph
Live water/setpoint temperature trace (temperature log per session; TEMPERATURE_UPDATE events + history backfill).
- **Surfaced in:** Devices tab temperature graph, Operations tactic 'STAGE TEMP' readout
- **Ops today:** view, act(history /api/temperature/{session}/history)
- **Relates to:** Setpoint (temperature), Session

### Device state
Live hardware state and controls — XY stage, bottom cam, SPIM/lightsheet, laser, LED/room-light, F-drive, piezo/galvo. The 'scope' surface reachable with no plan.
- **Surfaced in:** Devices tab (Operate/Map/Details/3D/Manual), landing 'Take a quick look'
- **Ops today:** view(/api/device-status), act(stage move, camera start, laser off, led/room-light set, live params, F-drive/bottom-Z nudge)
- **Relates to:** Embryo, Volume/Image, Setpoint (temperature), Session

### Agent chat / turn
The docked conversation with the agent — the delegation and steering channel; also the only place campaigns/plans get created and the notebook is briefed.
- **Surfaced in:** docked Agent chat panel (#agent-chat), 'Talk to Gently' rail button, landing 'or just tell me what you need'
- **Ops today:** view, act(send), act(stop/interrupt turn), act(queue message while busy)
- **Relates to:** Ask, Session, Campaign, Notebook note, Operation Plan

### Ask (agent → human)
A pending question/choice the agent raises mid-turn, rendered prominently on the main stage as well as in the transcript; answered by the control holder.
- **Surfaced in:** #ask-stage (main stage), Agent chat transcript
- **Ops today:** view, act(answer choice — control)
- **Relates to:** Agent chat / turn, Question

### Event / log
Session event stream and timeline (timeline.jsonl / interaction_log). The audit trail of what happened.
- **Surfaced in:** Logs tab (Log/Timeline/Summary views), footer counters ('N events')
- **Ops today:** view, list(/api/events), act(clear)
- **Relates to:** Session, Embryo, Volume/Image

### Config / dashboard prefs
Effective server config (read-only), per-browser dashboard view prefs, alert thresholds, and restart-required advanced tunables.
- **Surfaced in:** Settings page (Views/Alerts/Ambient/Board/Filmstrip/Vitals/Default/Effective config/Advanced)
- **Ops today:** view, edit(dashboard-defaults PUT, settings-overrides PUT, advanced save), export(prefs JSON), import(prefs JSON), act(save as rig defaults / reset)
- **Relates to:** Embryo (view rendering), Setpoint (temperature), Mesh / peer

### Mesh / peer instance
Other gently instances on the network (peer discovery, campaign sharing/participants server-side). Has NO interactive UI — only a read-only block in Settings and Advanced thresholds.
- **Surfaced in:** Settings > Effective config ('mesh' block, read-only), Settings > Advanced ('Mesh network' thresholds)
- **Ops today:** view
- **Relates to:** Campaign, Config / dashboard prefs

### Auth / control
The control-vs-view-only model: signing in grants control of the microscope; logged-out users watch read-only. No discoverable sign-in in the workspace — surfaced reactively via a 403 control-toast or the /login URL.
- **Surfaced in:** /login page, control-auth toast (control-auth.js)
- **Ops today:** act(login POST /api/auth/login), act(logout), act(continue view-only), view(/api/auth/me)
- **Relates to:** Session, Setpoint (temperature), Question, Watchpoint, Expectation, Agent chat / turn
