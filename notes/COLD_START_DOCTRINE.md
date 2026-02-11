# Cold Start Doctrine

The daemon architecture provides the vessel for continuous cognition. This document
addresses what fills that vessel: how the daemon bootstraps from an empty mind to a
useful thinking partner.

The technical cold start (launching the process) takes seconds. The cognitive cold
start (going from empty context to useful partnership) is the real problem.

## The Problem

On first launch, `context.db` is empty. Every table --- campaigns, learnings,
observations, expectations, watchpoints, questions, embryo_understanding --- is blank.
The daemon has its system prompt (baked-in C. elegans knowledge), but it knows nothing
about this lab, this researcher, this experiment, or this organism's history under
this microscope.

A daemon with no expectations cannot be surprised. A daemon with no goals cannot
prioritize. A daemon with no history cannot pattern-match. It can only log.

## Context Strata

The daemon's context has natural layers, each with different lifespans, different
cold-start strategies, and different sources of truth.

### Layer 1: Lab Identity (seeded once, persists forever)

What organism. What microscope. What imaging modality. Who works here. What the
research program is about. Lab conventions and preferences.

This is the most stable layer. Set once, updated rarely. A daemon that knows the
lab can orient itself in any session.

**Source:** Conversational onboarding on first-ever launch.

### Layer 2: Campaign (weeks to months)

A campaign is a research goal that spans multiple sessions. "Generate training data
for a stage classifier." "Characterize division timing variability under temperature
stress." "Replicate the imaging protocol from [paper]."

Campaigns carry: goals, expected timelines, sample requirements, imaging parameter
recommendations, success criteria.

**Sources:**
- Conversational: researcher describes what they're working on.
- Ingestion: researcher shares papers, protocols, or notes. The daemon reads them
  and proposes a campaign plan --- what experiments to run, what samples are needed,
  what imaging parameters to use, what to watch for.

### Layer 3: Session Intent (hours)

What are we doing today? Which campaign does this session belong to? What samples
are loaded? What do we expect to observe?

If Layers 1--2 are filled, this is lightweight: the daemon already knows the lab and
the campaign, so it asks targeted questions rather than open-ended ones.

**Source:** Brief check-in at session start + induction from first observations.

### Layer 4: Real-Time (minutes)

What's happening now. What just changed. This is pure observation --- no bootstrapping
needed. This is where the existing daemon loop already works.

**Source:** Event bus, perception system, hardware state.

### The Compounding Effect

Each filled layer makes the next layer's cold start cheaper and richer. A daemon that
knows the lab and the campaign can bootstrap a new session with one or two questions
instead of twenty. A daemon with no Layer 1 context is guessing.

## Phase 0: Gap Assessment

Every startup begins with self-assessment. The daemon inspects its own context store
and identifies what's missing. Each gap maps to an action.

| Check | Gap means | Action |
|-------|-----------|--------|
| No learnings with `basis` mentioning lab/setup | Don't know this lab | Layer 1 onboarding |
| No active campaigns | No research direction | Campaign onboarding or ingestion |
| No session_intents for this session | Don't know today's plan | Session check-in |
| No expectations | Can't be surprised | Need goals to reason against |
| No watchpoints | Not watching for anything | Need campaign/session context first |
| Active campaign exists, recent sessions exist | Continuing work | Light check-in only |

The gap assessment produces a `ContextGap` report. This report drives what happens
next --- the daemon generates interaction tasks (ASK, SURFACE) proportional to what
it lacks.

A daemon that starts with rich Layer 1--2 context and an active campaign skips
straight to: "Are we continuing [campaign]? What's on the stage today?"

## Phase 1: Lab Onboarding (first launch only)

Triggered when the gap assessment finds no lab-level context. This is a conversation,
not a form. The daemon (via the copilot) asks questions and internally maps the
answers to structured context.

The researcher says: "I've been struggling to catch the early divisions --- they
happen fast and I keep missing them." The daemon hears:
- Watchpoint: early divisions
- Concern: timing
- Implicit goal: catch early divisions
- Expectation: they happen fast

One sentence, four context entries.

The tone is apprenticeship, not configuration. The daemon is the new lab member
learning how things work here.

**Seeds:** Lab identity learnings, initial watchpoints, research program understanding.

**Duration:** A few minutes. Not exhaustive --- the daemon learns more over time.

## Phase 2: Campaign Planning

Triggered when no active campaign exists, or when the researcher starts new work.
This phase has two paths:

### Path A: Conversational

The researcher describes their goals. "I want to characterize how temperature
affects early division timing." The daemon asks clarifying questions, then proposes
a campaign: target sample count, imaging intervals, stages to capture, success
criteria.

### Path B: Ingestion-Driven

The researcher shares papers, protocols, or notes:

```
/ingest https://doi.org/10.xxxx/paper-about-division-timing
/ingest C:\lab\protocols\temperature_stress_imaging.pdf
/ingest "We want to replicate Figure 3 from the Smith et al paper"
```

The daemon reads the material and extracts:
- **Campaign plan:** What experiments to run, over what timeframe
- **Sample requirements:** What strains, stages, conditions are needed
- **Imaging parameters:** Recommended intervals, z-stack depth, exposure, channels
- **Expected timelines:** Developmental milestones and when to expect them
- **Things to watch for:** Known failure modes, subtle phenotypes, quality indicators

The daemon proposes, the researcher refines: "yes but we only have N2 strain" /
"our microscope can't do simultaneous dual-view." The daemon adapts the plan.

This compresses the cold start curve dramatically. Instead of slowly learning what
matters over ten sessions, the daemon arrives at session one already understanding
the scientific context.

**Seeds:** Campaign with goals and targets, projects, imaging parameter learnings,
expectations, watchpoints.

## Phase 3: Session Intent (each session start)

If a campaign is active, the daemon checks continuity: "Are we continuing
[campaign]? What's the plan for today?"

If the daemon has past session data (from `gently.db`), it can propose:
"Last session we imaged 6 of 8 planned embryos. Embryo 3 showed unexpected early
division. Recommend prioritizing embryo 3 today and completing the remaining 2."

**Seeds:** Session intent, session-specific expectations and watchpoints.

## Phase 4: Observation and Accumulation (during session)

The existing daemon loop: observe, predict, surprise, learn. Now much richer because
Layers 1--3 provide context to reason against. The daemon's observations are
meaningful because it has goals. Its surprises are real because it has expectations.

## Phase 5: Session Synthesis (session end)

When a session ends, the daemon reflects:
- What happened vs. what was planned
- Campaign progress update
- Revised expectations for the next session
- New learnings to carry forward
- Open questions discovered during the session

This synthesis becomes the input for the next session's Phase 3.

## The Ingestion Capability

Ingestion is a first-class capability alongside hardware, perception, and interaction.
It is the mechanism by which external knowledge enters the daemon's context.

### Inputs

- **Paper URLs:** Fetched via web, processed by Claude for experimental design extraction.
- **PDF file paths:** Read locally, processed similarly.
- **Plain text / notes:** Direct input from the researcher.
- **Past session data:** Read from `gently.db` to bootstrap historical understanding.
- **Web search:** Find related work, protocols, or reference parameters.

### Processing

The ingestion capability passes the content to Claude with a structured extraction
prompt. For a paper, this might extract: organism, methods, imaging parameters,
key findings, developmental timelines, recommended protocols. For a protocol
document, it might extract: step-by-step procedures, required reagents/strains,
expected outcomes.

The extracted information is then mapped to context model entries: campaigns,
learnings, expectations, watchpoints.

### Interface

- `/ingest <url-or-path>` --- CLI command to ingest a document
- `/ingest` with no argument --- prompts for URL, path, or paste
- Conversational: "I want to base our next campaign on these papers"
- Automatic: daemon notices past sessions in `gently.db` and offers to learn from them

### Web Capabilities

The ingestion capability requires web access for paper URLs. This is implemented
via Anthropic's API --- fetch the content, pass it as context to Claude for
extraction. For search, the daemon can find related protocols or reference imaging
parameters for specific organisms/stages.

## The Cold Start Curve

The doctrine defines transitions along a curve:

```
Blank ──> Oriented ──> Directed ──> Engaged ──> Attuned
  │           │            │            │           │
  │     Lab identity   Campaign     Session     Accumulated
  │      (Layer 1)    (Layer 2)   (Layer 3)    experience
  │           │            │            │           │
  │     First-launch   Ingestion   Check-in    Observation
  │     conversation   + planning  at start    over sessions
  │           │            │            │           │
  └───────────┴────────────┴────────────┴───────────┘
              Each stage makes the next cheaper
```

- **Blank:** Knows nothing, can only log.
- **Oriented:** Knows the lab, the organism, the setup. Can form basic expectations.
- **Directed:** Knows the campaign goals. Can prioritize, plan sessions.
- **Engaged:** Knows today's intent. Has expectations, watchpoints. Can surprise.
- **Attuned:** Has seen patterns, built predictions, learned the researcher's style.
  Can anticipate, suggest, connect.

The first three transitions are deliberate (onboarding, planning, check-in). The
last is emergent (accumulated through observation). The doctrine governs the
deliberate transitions; the existing daemon architecture handles the emergent one.

## Design Principles

1. **Gap-driven, not flow-driven.** The daemon assesses what it lacks and acts
   accordingly. No fixed onboarding sequence --- a daemon with rich Layer 1 context
   from a previous install skips straight to campaign planning.

2. **Apprenticeship, not configuration.** The tone is a new lab member learning,
   not a form to fill out. One conversational sentence can seed multiple context
   entries.

3. **Front-load active, taper to passive.** The first session is conversation-heavy.
   By the fifth session, the daemon mostly observes and only asks when something
   contradicts its model.

4. **Ingestion compresses the curve.** Papers and protocols can teleport the daemon
   from Blank to Directed in minutes, bypassing slow observational learning.

5. **Each layer serves the next.** Lab identity makes campaign planning easier.
   Campaign context makes session check-ins lighter. Session intent makes
   real-time observation meaningful.
