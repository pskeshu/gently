# Gently — Product-Ideation Framework

How we surface product ideas from the UX audit. Ideas come in KINDS; each lens is a
question swept across the entity inventory + the two matrices + the crawler graph +
the per-page screenshots. Built by a fan-out (7 generators + 4 method-improvers); the
improvers grew this from 7 core lenses to the set below.

## Lenses

### Core lenses

- **missing-affordance** — For a verb the user obviously wants here (create/edit/delete/save), is there a control on this surface — or is it reachable only via the agent or an incidental path?
- **cross-feature-link** — Two entities co-exist and reference each other in the data model, but is there a traversable UI path between their surfaces?
- **hidden-state / make-visible** — Is there durable state (lock, control-holder, role, health, liveness) that the backend already knows but no surface shows?
- **step-reduction** — Does a common action cost more clicks/tab-hops than the data on hand requires, and can it collapse to one?
- **flywheel (producer→consumer)** — Is an artifact produced (ground truth, ML assessment, embryo-understanding, resolved expectation) that no consumer ever reads back — a dead-end producer?
- **agentic (augmented-LLM in the loop)** — Is there a tedious judgement/triage/authoring step where the agent should narrow and the human should decide?
- **consistency / cross-surface parity** — Does the same entity get an affordance on one surface and an inert copy on another, so the two drift?

### Added by the method-improvers

- **error-recovery / failure-path** — When this action fails or its dependency is offline/empty, does the surface name what broke and offer a way forward (retry, fallback, where-to-look)?
- **navigation / findability** — Can you jump from any entity MENTION to that entity in one click, and find a thing by name without knowing its tab (deep-link + Cmd-K)?
- **collaboration / presence / control-ownership** — If two humans, or a human and the agent, share this session, do they see each other, know who holds control, and can they hand it off?
- **remove-don't-add (subtraction)** — What here is noise, redundant, stale, or actively misleading and should be deleted or merged rather than augmented?
- **trust / provenance / feedback-integrity** — Can the user see WHY the agent/model did what it did, reach what it was based on, judge it, and does that judgment actually go somewhere?
- **safety / reversibility** — For irreversible or specimen-affecting actions (laser, temperature, delete, stop), is there a proportionate guard, preview, or undo?
- **unattended / temporal** — For operations that outlive attention (a ~14h timelapse), is there progress/ETA and a way to be told out-of-app when a human is needed?
- **provenance / interop / export (app boundary)** — Can data, results, and their provenance leave the app for analysis, citation, or reproduction — and come back in?
- **onboarding & expert-mode** — Does a first-timer get oriented from a cold/empty state, and does a returning power-user get accelerants (memory, defaults, bulk, shortcuts)?
- **loop-closure (JTBD spine)** — Does this close a loop on Plan→Operate→Acquire→Perceive→Learn→Decide, or force an app-exit / re-keying / agent round-trip mid-loop? (ranking lens)
- **capability-orphan (store-verb diff)** — For every mutating store method, is there a route AND a UI control that invokes it? Orphaned verbs are missing affordances, derived not guessed. (generator)
- **dangling-edge (data-model FK diff)** — Which entity already stores a foreign key to another entity that the UI renders as dead text? Highest leverage/effort links live here. (generator)
- **agent-arbitrage (filter)** — This is already doable via the agent — is the manual affordance materially better (faster/safer/in-context/discoverable/works-when-agent-busy), or a redundant reimplementation?
- **noise-collapse (meta-filter)** — Is this the Nth instance of a surface-pattern template — and what single SYSTEMIC idea does the whole cluster collapse into?
- **frequency × friction** — For the handful of things this persona does 10+ times a day, how many hops does each cost and what collapses it to one tap? (ranking weight)

## Method notes

METHOD IMPROVEMENTS. (1) Shift from a 7-kind additive taxonomy to a LENS LIBRARY of 22 questions run against every (surface × entity × graph-edge) triple, split by role: GENERATORS derive candidates by construction rather than inspiration — capability-orphan (every mutating store method → route → UI control; orphaned verbs = missing affordances, e.g. set_ground_truth, create_campaign, note-create) and dangling-edge (every foreign key in the data model rendered as dead text = a cross-feature link, e.g. Note.embryos/strains/basis, item.session_ids/depends_on, claimed_by). These read the CODE/data model, not the rendered screen, which is exactly where the deepest, lowest-effort ideas hide and where a screenshot-only audit is structurally blind. (2) RANKING lenses: loop-closure on the Plan→Operate→Acquire→Perceive→Learn→Decide spine, and frequency×friction (weight the mark→run loop run ~20×/day over once-per-project config). (3) FILTERS applied before scoring: agent-arbitrage (a manual affordance must be materially better than the agent path — faster/safer/in-context/discoverable/works-when-agent-busy — or it's dropped; 'the agent can do it in chat' is a RED FLAG masking a gap, not coverage) and noise-collapse (an Nth template instance collapses to one systemic idea — ~15 'nicer empty state' items became one 'empty states deep-link to their seeding action').

QUALITY RUBRIC. Score = (impact × reach × depth × trust) / effort, gated by code evidence, with a structural bias (+ for missing-affordance and cross-feature-link) applied per the brief. DEPTH axis is decisive: DEEP if it creates/persists a new entity or edge (ground truth, dependency, note↔plan link); COSMETIC if it only moves pixels — the two must never rank equal. HARD-REJECT before scoring: template spam, cosmetic-only polish, audit-echo (restating a US-## gap adds nothing over an idea with a concrete mechanism), and agent-redundant LLM bolt-ons (the rejected 'AI summary on Logs' / 'refresh button everywhere'). Effort-blind ranking is banned — a one-line render of an existing foreign key must outrank a huge-payoff/huge-cost item, which is why IDEA-04/IDEA-11 rank above IDEA-20/IDEA-25/IDEA-39.

WHAT TO KEEP. The three graph artifacts as living inputs: G_nav (crawler graph.json — proves the app is a star of sibling tabs with zero entity-to-entity edges), G_data (entity/FK graph from the storage model), G_verb (store-method→route→handler capability graph). Emit ideas as mechanical diffs — cross-feature-link = a G_data edge whose endpoints both have surfaces but no G_nav path; missing-affordance = a G_verb verb that dead-ends before the UI; orphan-surface = a G_nav node only ever seen empty whose seeding action is unreachable. Dedup by systemic collapse (46 raw candidates → 45 ranked, with the largest merges being ground-truth ×11, add-note ×6, new-campaign/plan ×6, chip-deep-link ×5). Keep the added lenses that catch the core-lens blind spots the method-gap list named: failure branch, subtraction, second actor/control-ownership, temporal/unattended, app-boundary export, physical risk, and fine-grained navigability.

## The two matrices (the mechanical core)

- **Entity × Operation** (`ENTITIES.md`): view/create/edit/delete/link/export per entity — every empty cell is a candidate missing-affordance. Sharpened by *capability-orphan*: diff the store's mutating methods against routes+controls.
- **Entity × Entity linkage** (`ENTITIES.md`): related-but-unlinked pairs are cross-feature ideas. Sharpened by *dangling-edge*: a stored foreign key rendered as dead text is the highest-leverage link.

_Ideas land in `BACKLOG.md` (+ `backlog.json`, queryable/appendable). Re-run the engine to refresh; append by hand as ideas arrive._