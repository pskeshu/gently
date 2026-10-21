# The project layer

*What a project is in gently, where its facts live, and which door writes
them. Written 2026-10-20 with the code; the study that led here is
`project-layer-study.md` on PR #242.*

## One sentence

**A root campaign is a project.** Its phases are child campaigns, its plan is
the plan items under them, and everything else a project holds — sessions,
expectations, questions, findings, strains, references — is either a field on
something that already existed or a filter over the global pools, keyed by one
new field on the session.

## Membership: one home, many items

```
sessions/{…}/session.yaml      project_id: <root campaign id>     ← the one truth
agent/campaigns/{root}/plan/… PlanItem.session_ids[]              ← which items a session served
agent/session_intents/{sid}   campaign_ids[]                      ← kept for the resolution tools
```

A session has one home project and can serve plan items in any project. Every
writer goes through `gently/app/filing.py`:

| caller | call |
|---|---|
| the agent running a plan item (`plan_execution_tools`, `resolution_tools`, leaving plan mode) | `file_session_under(store, cs, sid, campaign_id, item_id)` |
| Sessions tab "File under…", the project's "+ add a session" | `POST /api/sessions/{sid}/project {campaign_id, plan_item_id?, move?}` |
| taking a session out | `POST /api/sessions/{sid}/project {campaign_id: null}` → `unfile_session` |
| startup, once | `backfill_session_projects` gives pre-existing sessions the home their plan links imply |

The home is set when the session has none. A later item from another project
links the item and keeps the home (`home_kept`), unless `move` is passed.

## Context is a filter

Observations, notes, expectations, questions and interaction logs hang off a
session id, and a session has a project, so the project's context is the
global pools filtered by membership. Nothing moved on disk. Three records
gained a field to make that filter exact:

- `Note.projects[]` — a finding that belongs to a project and to no session.
- `Expectation.session_id / project_id`, `Question.session_id / project_id` —
  made in a session, or for a project outside one.

`FileContextStore.list_expectations(project_id=, session_ids=)` and
`list_questions(...)` return everything that matches either; the ledger passes
the project id and its member session ids.

## The ledger

`GET /api/campaigns/{id}/ledger`

| column | entity | writer (agent) | writer (person) |
|---|---|---|---|
| next | plan items, unblocked (`get_unblocked_plan_items`) | `create_plan_item` | `POST …/items`, Accept |
| proposed | plan items with `status: proposed` | `create_plan_item` outside plan mode | — (accept or dismiss) |
| predicted | expectations | `record_expectation` | `POST /api/context/expectations` |
| measured · outcomes | completed or skipped items with `outcome` | `complete_plan_item` | `PATCH …/items/{id} {outcome}` |
| measured · findings | notes of kind `finding` | `propose_finding` (status proposed) | `POST /api/notebook/notes` (confirmed) |
| open · answered | questions | `raise_question` | `POST /api/context/questions`, `…/resolve` |

Two things need a person's acceptance because they run hardware or claim a
result: a plan item (`proposed` → `planned` via `POST …/items/{id}/status`)
and a finding (`proposed` → `confirmed` via `POST /api/notebook/notes/{id}/
status`). `get_unblocked_plan_items` never returns a proposed item, so the rig
cannot run a suggestion. `PlanItem.origin` says `agent` or `manual`.

## An intention becomes a spec

A plan item added by hand is a sentence. Two passes make it actionable
(`gently/harness/memory/intent_spec.py`):

1. `parse_intent` reads what the sentence states outright, offline: embryos,
   cadence, channels, stop stage or timepoints or duration, temperature,
   strain, slices, exposure, power, reporter, and the clause after "to check…"
   as the success criterion. Each field remembers the words it came from.
2. `draft_spec_with_model` asks Gently to reason the rest out: what kind of
   acquisition this is, the settings within the instrument's limits, a tactic
   outline the run can execute (a sweep is a `scripted_protocol` with steps),
   a short title, a one-line rationale and its assumptions. It rests on the
   project's context and a card of instrument facts.

`POST …/items` runs pass one at once and starts pass two in the background;
the card says "drafting…" and fills when the draft lands. `POST …/items/{id}/
draft` redrafts, with an optional `note` constraint. `POST …/spec/accept`
makes the drafted fields the person's. Provenance per field is `sentence`,
`model`, `typed` or `accepted`; a draft never replaces a person's field, and
the sentence outranks the draft. Nothing runs on a draft until it is accepted.

## The acquisition is the pane's shape

`ImagingSpec.acquisition` is the structure Devices › Operate runs, the same
one a saved tactic carries and the start route keeps with a session
(`gently/harness/memory/acquisition.py`): cadence, volumes on or off, slices
and exposure, the laser preset and per-line powers, the brightfield overview
block, the ending as the orchestrator's spec string. The older flat fields are
read into it (`acquisition_from_legacy`); an explicit acquisition wins
(`effective_acquisition`); every serialized item carries the result, with
`acquisition_derived` saying which. The card and the inspector say it in the
pane's own sentence (`AcquisitionPlan.describe`), the server in
`describe_acquisition`.

**Set up on Devices** hands an item to Devices › Operate: the Acquisition pane
fills from the item, a banner names it, and Start sends `plan_item_id`, on
which the start route files the session under the item's project and writes
the structure it ran back onto the item as the operator's own. The agent's
execute tool starts from the same structure.

## References, through one search

A root campaign carries `strains[]` and `references[]`, each
`{source, key, title, url, subtitle, added_by, added_at}` — the shape plan-item
references already had. `add_campaign_reference` keeps one entry per
`(source, key)`.

`GET /api/search/{source}?q=` answers one source at a time, in one shape,
`{source, status: ok|empty|error, took_ms, results[], error}`, so the add
popover queries five in parallel and draws each group as it lands. The agent's
`search_references` tool calls the same `research.search`.

| source | backed by | status 2026-10-20 |
|---|---|---|
| lab | names already in gently: project strains and links, plan-item specs, notebook strain tags | instant, offline |
| genes | Alliance of Genome Resources autocomplete (WormBase curies) | JSON |
| strains | CGC search page (HTML, parsed by `_parse_cgc_rows`) | works; links are absolute now |
| papers | Europe PMC (a DOI or PMID resolves directly) | JSON |
| terms | EBI OLS4, ontologies `wbls` and `wbbt` | JSON |
| WormBase REST | — | answers scripts with a Cloudflare challenge; not called |

Stage names map to WBls terms in `gently/organisms/celegans/stages.py`
(`WBLS`, `ontology_id`); the session detail hands the map to the Sessions tab.

## Observability is an aggregation

`GET /api/sessions?project_id=` adds, per member session, `errors` and
`warnings` from `events.jsonl`, `turns`, `corrections`, `tool_calls` and
`agent_errors` from `interaction_log.jsonl`, and the `plan_items` it served.
The helpers live in `gently/core/session_summary.py`, shared with the Sessions
tab and with the session-end note the agent writes when the live session
changes (`_note_session_end` in `gently/app/agent.py`: one deterministic
observation, nothing for an empty session, never twice).

The project list (`GET /api/campaigns`) carries a `project` summary per root:
`sessions`, `live`, `last_run_at`, `proposed_items`, `proposed_findings`.

## Deliberately not built

- **Run identity** (`sessions/{s}/runs/<id>/`, study slice 1). A session with
  two Starts reports its last run, as the Sessions tab does today.
- **Operate-start auto-filing** (study slice 5): a run started on the Operate
  pane in a session that has a project does not yet create a manual plan item.
  The one door exists; the pane does not call it.
- **Documents** (files copied into the project, study slice 4). Links by URL,
  DOI or id cover papers and pages; uploads do not exist.
