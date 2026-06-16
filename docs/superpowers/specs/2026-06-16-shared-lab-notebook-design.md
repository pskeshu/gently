# Design: The Active, Shared Lab Notebook (agent + human memory)

Status: design approved in brainstorm (2026-06-16). Concept trace: gently-project/gently#52.
Branch: `feature/memory-model` (off `feature/ux-v2`, which it depends on for UI).

## 1. Overview

Gently should have one **active, shared lab notebook** — a living memory the agent and
the scientist co-author. The agent writes from doing the work; the human adds notes,
literature, thoughts, and annotations; both read it and *think with* it. It accumulates
across timepoints, sessions, and (eventually) systems.

This replaces today's "agent mind" — a clean conceptual model (`Context`: intentions,
understanding, observations, expectations, attention) that was never engineered into a
working system. We keep its good bones and redesign the structure.

### What we keep from the old agent mind
- The **file-based, human-browsable YAML store** (`FileContextStore`) — auditable, no DB lock-in.
- **`Observation` as the entry template** (`model.py:379`) — content + refs + significance + embryo/session is already ~90% of a notebook entry.
- **`basis` + derived `confidence`** on knowledge (`Learning`, `model.py:329`).
- **`gently_refs`** for pointing at artifacts (`model.py:394`).
- The **sense → believe → predict → attend** cognitive framing, as the *meaning* of entries.
- The **event bus + `CONTEXT_UPDATED` → browser** path and the existing `/api/context` surface and "Agent's View" panel (`context-surface.js`).

### What was wrong (and we fix)
- The write loop is **dormant**: `apply_updates()` (`file_store.py:2178`) is never called anywhere. The mind was designed but never wired to fill.
- **Six parallel silos, no links** — `Observation.relates_to` exists but is never populated. A set of disconnected lists, not a connected mind.
- **No author/provenance dimension** — everything is implicitly agent-authored, so it can't be shared/curated.
- **Working and durable memory are mixed** — live per-embryo predictions sit beside things you'd keep forever.
- **No consolidation** — raw never becomes distilled on its own.
- **Type proliferation** — distinctions that should be *fields* (author, status) were encoded as separate *kinds*.

## 2. The data model

### Principle
A **kind** exists only if the system stores, reasons about, or *shows* it differently.
Otherwise it is a **field**. This prevents re-proliferation.

### Three kinds
- **Observation** — a record of something seen, done, read, or noted. *Immutable.* Timestamped evidence.
- **Finding** — a believed claim / conclusion / insight. *Revisable and supersedable*, with a `basis` (the observations it rests on) and a confidence *derived from evidence*. Can be contradicted — a signal, not an error.
- **Question** — an open inquiry being chased. Open→answered lifecycle. The large, long-lived ones **are the organizing spine** (the inquiry thread).

### Orthogonal fields (shared by all kinds; NOT new kinds)
- `author` — human | agent | perception (a human note, a literature ref, and an agent observation are all Observations with different authors).
- `status` — proposed | confirmed | superseded | answered | … (lifecycle).
- `confidence` — derived from supporting/contradicting evidence; never self-rated by the model.
- `scope` — strain / embryo / session / thread (the cross-cut indexes).
- `links` — typed edges to other entries (supports / contradicts / refines / answers / produced-by). The graph that makes it a wiki, not a log.
- `artifacts` — pointers into `FileStore` (images, volumes, projections, traces, sessions) by reference, never copied.

### The inquiry thread (the spine)
A `Question` that grew a body: holds the question, a rolling synthesis, status
(open/investigating/resolved), and denormalized scope. Membership lives on the entries
(`threads: [...]`), not in the thread — a flat note pool, derived reverse-indexes
(`by_thread`, `by_strain`, `by_embryo`) that are rebuildable caches. This lets "by question"
+ "by strain" + links coexist over the flat YAML store with no database.

### Working memory vs. notebook (the lifespan split)
- **Working memory** (transient, the live loop): the agent's predictions and things-to-watch
  during a run. A thin runtime layer, *not* notebook entries. When one matters (a violated
  prediction), it **graduates** into an Observation.
- **Notebook** (durable): Observations, Findings, Questions — what accumulates.

### The Question is the hinge between knowing and doing
"What experiments next?" is a Question, but it resolves into **plan items (action)** in the
existing campaign/plan layer, whereas a scientific question resolves into a **Finding
(knowledge)**. Same kind, different exit — the system tells them apart by *what they link to*
(plan items vs findings), no hardcoded subtype. The loop:

```
Question ("what next?") → brainstorm over Observations + Findings
   → plan items (a campaign) → run them
   → Observations → Findings → may answer the scientific Question → thread converges
```

Knowledge → action → knowledge. The plan/campaign layer (and its export) is the "action" arm.

### Mapping the old six types
| Old type | New representation |
|---|---|
| `Observation` | `Observation` (near-identity; `embryo_id/session_id` → `scope`, `gently_refs` → `artifacts`, `relates_to` → `links`) |
| `Learning` | `Finding` (`content`→body, `basis`/`confidence` kept) |
| `Expectation` | working memory; graduates to `Observation` when violated |
| `Watchpoint` | working memory (live monitoring) |
| `Question` | `Question` (passing) or inquiry thread (recurring) |
| `EmbryoUnderstanding` | `Finding`(s) scoped to an embryo; supersession gives stage history |

## 3. Presentation (two faces)

The notebook is **active**, so it can't live only in a tab (the write-only-junk-drawer failure).
It has two faces:

1. **The Notebook tab** (LIBRARY group, beside Plans and Sessions): the reading room.
   *Plans = what we'll do · Sessions = what we did · Notebook = what we know.* Landing view
   organized by the **inquiry-thread spine**, with strain/embryo **filters** and **search +
   link-graph** navigation. A thread page reads like a living entry — the question, its rolling
   synthesis, Findings and Observations beneath, and **links out to its campaign(s) in Plans and
   runs in Sessions**. The three library tabs interlink rather than duplicate.

2. **The ambient edge** — the notebook coming to you, in context (Home, mid-session, planning):
   the session-start brief, the one-thing surfacing, the brainstorm reply. **This is what
   "Agent's View" becomes** — repurposed from near-empty transient cognition into the notebook's
   live edge.

## 4. Retrieval & extraction

Principle: **structure → indexes (deterministic); meaning & judgment → models.**
(Exact structural retrieval is not a heuristic — it's an index lookup, and a model must not do it.)

Three-layer pipeline:
1. **Narrow — structural (instant).** Fields/indexes filter by strain, embryo, thread, time,
   status, link-traversal. Cuts thousands → dozens.
2. **Recall + rank — semantic.** Embeddings (computed once per entry, kept in a rebuildable
   sidecar vector cache; YAML stays source of truth) pull semantically-near candidates; an LLM
   **re-ranks/selects** with reasons. Recall cheap, precision is judgment.
3. **Synthesize — generation (model).** Thread summaries, session-start briefs, brainstorm
   answers, contradiction calls, consolidated Findings. Non-negotiable: **grounded + cited**
   (uncited claim = bug; "not in notebook" is valid), **forced structured output** (validated
   objects, never re-scraped prose), **ranking from structural facts not self-rated confidence**.

Two query faces:
- **Agent**: full pipeline feeding its reasoning.
- **Human search**: *Filter* mode (instant, structural, as-you-type) + *Ask* mode (full grounded
  LLM answer with citations). No model in the typeahead path.

"Unlimited API credits" removes **cost** (lean on models for judgment/synthesis, frequent
consolidation, regenerate thread summaries on change) but not **latency** (keep interactive
filtering model-free) or **hallucination** (grounding + citation + structured output remain the
guardrails).

## 5. Trust principles
- Append-only evidence; **supersede, never overwrite**, conclusions (the chain is the history).
- Agent and human are **separate, attributed voices**; the agent appends/proposes/links, never
  silently rewrites the human's words.
- Every claim **carries its basis** and cites the entries it rests on.
- Confidence is **derived from evidence**, never self-rated.
- **Revisiting is designed in** — stale findings/questions/unrevisited results resurface.

## 6. Build decomposition (increments)

The full vision is several independently-shippable increments. Ordered by value-vs-risk:

1. **Foundation / keystone — make memory accumulate and be browsable.**
   Unified `Note` (3 kinds + fields) + store + indexes; wire the dormant producer loop
   (`apply_updates`) so a session actually writes Observations/Findings; surface a read-only
   **Notebook tab** (thread-organized) + repurpose Agent's View as the live edge.
   *Produce (perception/agent) → store (notebook) → consume (tab + ambient read).*
2. **Retrieval & brainstorm** — structural indexes + embeddings sidecar + grounded LLM
   synthesis; the "Ask the notebook" + agent brainstorm pipeline.
3. **Human authorship & curation** — human-authored entries/annotations (on images/embryos),
   proposed→confirmed curation, supersession UI.
4. **Active surfacing & consolidation** — structural triggers (contradiction / answered-question
   / violated-expectation / echo), throttled one-at-a-moment; scheduled reflection passes.
5. **Cross-system** — sharing/sync of the notebook across instances ("shared brain").

**First slice = Increment 1** (the keystone): it proves the model end-to-end and turns the
"wired but starving" problem into the first visible win. Increment 1 gets its own
implementation plan.

## 7. Out of scope (for now)
Cross-system sync (incr. 5), proactive surfacing tuning (incr. 4) — designed here, built later.
