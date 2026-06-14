"""Eval / replay / shadow primitives.

Substrate for testing orchestrator architectures without running real
hardware. The three layers:

  EventCapture — records every EventBus event to a per-session jsonl
                 file so the agent's input stream is durable.
  EventReplay  — reads a captured jsonl and republishes events to a
                 target bus, preserving original timestamps.
  ShadowRunner — hosts candidate orchestrators that subscribe to the
                 live (or replayed) bus, log their decisions, and
                 never touch hardware. Diff their decision logs to
                 compare architectures.

See docs/EVAL.md (TODO) for usage.
"""

from .candidates import ReactiveCandidate
from .decision_log import Decision, DecisionLog, DecisionTrigger, prompt_hash
from .event_capture import EventCapture
from .event_replay import EventReplay
from .shadow import NoOpCandidate, OrchestratorCandidate, ShadowRunner

__all__ = [
    "EventCapture",
    "EventReplay",
    "Decision",
    "DecisionLog",
    "DecisionTrigger",
    "prompt_hash",
    "OrchestratorCandidate",
    "ShadowRunner",
    "NoOpCandidate",
    "ReactiveCandidate",
]
