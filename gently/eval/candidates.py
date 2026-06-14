"""Canned shadow orchestrator candidates.

NoOpCandidate lives in shadow.py as the trivial baseline. Anything more
interesting — even pure-rule architectures with state — lives here. As
LLM-driven candidates land they should slot into this module too.

Conventions every candidate should keep:
  - It maintains its own tiny world model. The production agent's
    `experiment` is intentionally not shared (a candidate that mutates
    production state would defeat the point of shadow mode).
  - Decisions go through `log_decision`. Never call hardware tools.
  - State updates from events are cheap (no LLM, no I/O).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from gently.core.event_bus import Event

from .decision_log import DecisionLog, DecisionTrigger
from .shadow import OrchestratorCandidate

logger = logging.getLogger(__name__)


@dataclass
class _ReactiveWorldModel:
    """The tiniest possible world model — everything ReactiveCandidate
    needs to make rule-based decisions without re-reading the agent."""

    # {embryo_id: {"coarse": {x, y} | None, "fine": {x, y} | None,
    #              "has_fine": bool, "confidence": float}}
    embryos: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Last live stage XY (µm) from a STAGE_MOVED event.
    last_stage_um: dict[str, float] | None = None

    # Last error message + timestamp, so the candidate can avoid
    # spam-proposing escalations for the same recurring failure.
    last_error: dict[str, Any] | None = None

    # Count of events seen, by type name — useful debug field that also
    # ends up in the decision context_summary.
    seen: dict[str, int] = field(default_factory=dict)


class ReactiveCandidate(OrchestratorCandidate):
    """Pure-rule reactive shadow orchestrator.

    The thesis being tested: *can a rule-based responder do the
    routine bookkeeping that today only happens when the operator
    chats with Claude?*

    Reactions
    ---------
    OPERATOR_EDITED_EMBRYO
        Operator moved an embryo on the Map. The PUT also clears fine
        position. Propose `recalibrate_embryo(embryo_id)` so the new
        coarse position gets a SPIM-fine alignment before the next
        acquisition. If `fine_position_invalidated` was False (no fine
        existed yet) skip the proposal — there's nothing to refresh.

    OPERATOR_MARKED_EMBRYOS
        Operator just confirmed a fresh set of embryos via the marking
        canvas. Propose `calibrate_all_embryos` to bring them all into
        focus. Cheap pattern: kick off calibration the moment sightings
        land, instead of waiting for the operator to type it.

    OPERATOR_REMOVED_EMBRYO
        Operator deleted an embryo. Propose a tidy-up step
        `forget_embryo(embryo_id)` for any candidate that wants to
        clean caches / learnings keyed on the gone embryo. No-op for
        production today (state mutation already happened); the
        proposal is reserved for downstream cleanup tools.

    ERROR_OCCURRED
        Propose `escalate_to_operator(error_message)` once per distinct
        error. Suppresses if the same error fires twice within 30s —
        avoids drowning the operator in repeat alarms.

    EMBRYOS_UPDATE / STAGE_MOVED
        Update the world model. No decision logged (silent ingest).

    """

    # If two ERROR_OCCURRED events with the same message arrive within
    # this window, only the first proposes an escalation.
    ERROR_SUPPRESS_WINDOW_SEC = 30.0

    def __init__(self, name: str, decisions: DecisionLog):
        super().__init__(name, decisions)
        self.world = _ReactiveWorldModel()

    # ---- event handlers ----------------------------------------------------

    def on_event(self, event: Event) -> None:
        name = event.event_type.name
        self.world.seen[name] = self.world.seen.get(name, 0) + 1

        # Always ingest state-shaped events first.
        if name == "EMBRYOS_UPDATE":
            self._ingest_embryos_update(event)
            return
        if name == "STAGE_MOVED":
            self._ingest_stage_moved(event)
            return

        # Operator + error events produce decisions.
        if name == "OPERATOR_EDITED_EMBRYO":
            self._react_operator_edited(event)
            return
        if name == "OPERATOR_MARKED_EMBRYOS":
            self._react_operator_marked(event)
            return
        if name == "OPERATOR_REMOVED_EMBRYO":
            self._react_operator_removed(event)
            return
        if name == "ERROR_OCCURRED":
            self._react_error(event)
            return

    # ---- ingests -----------------------------------------------------------

    def _ingest_embryos_update(self, event: Event) -> None:
        embryos = (event.data or {}).get("embryos") or []
        new_world: dict[str, dict[str, Any]] = {}
        for emb in embryos:
            new_world[emb.get("id", "")] = {
                "coarse": emb.get("position_coarse"),
                "fine": emb.get("position_fine"),
                "has_fine": bool(emb.get("has_fine_position")),
                "confidence": emb.get("detection_confidence", 0.0),
            }
        self.world.embryos = new_world

    def _ingest_stage_moved(self, event: Event) -> None:
        d = event.data or {}
        if "x" in d and "y" in d:
            self.world.last_stage_um = {"x": float(d["x"]), "y": float(d["y"])}

    # ---- reactions ---------------------------------------------------------

    def _react_operator_edited(self, event: Event) -> None:
        data = event.data or {}
        eid = data.get("embryo_id") or ""
        invalidated = bool(data.get("fine_position_invalidated"))
        tool_calls: list[dict[str, Any]] = []
        # Only propose a recalibration when there was a fine position
        # that the edit just invalidated. New coarse without any prior
        # fine has nothing to refresh yet.
        if invalidated:
            tool_calls.append(
                {
                    "name": "recalibrate_embryo",
                    "input": {"embryo_id": eid},
                    "id": None,
                }
            )
        self.log_decision(
            trigger=DecisionTrigger.EVENT,
            trigger_detail="OPERATOR_EDITED_EMBRYO",
            tool_calls=tool_calls,
            response_text=(
                f"Operator moved {eid}; proposing recalibration."
                if invalidated
                else f"Operator moved {eid}; no prior fine -- no action."
            ),
            recent_event_ids=[event.event_id],
            context_summary=self._summary(),
        )

    def _react_operator_marked(self, event: Event) -> None:
        data = event.data or {}
        ids = data.get("embryo_ids") or []
        count = data.get("count", len(ids))
        tool_calls: list[dict[str, Any]] = []
        if count:
            tool_calls.append(
                {
                    "name": "calibrate_all_embryos",
                    "input": {"embryo_ids": list(ids)},
                    "id": None,
                }
            )
        self.log_decision(
            trigger=DecisionTrigger.EVENT,
            trigger_detail="OPERATOR_MARKED_EMBRYOS",
            tool_calls=tool_calls,
            response_text=(
                f"Operator marked {count} embryos; proposing calibration."
                if count
                else "Operator marked zero embryos; no action."
            ),
            recent_event_ids=[event.event_id],
            context_summary=self._summary(),
        )

    def _react_operator_removed(self, event: Event) -> None:
        data = event.data or {}
        eid = data.get("embryo_id") or ""
        self.log_decision(
            trigger=DecisionTrigger.EVENT,
            trigger_detail="OPERATOR_REMOVED_EMBRYO",
            tool_calls=[
                {
                    "name": "forget_embryo",
                    "input": {"embryo_id": eid},
                    "id": None,
                }
            ],
            response_text=f"Operator removed {eid}; proposing cache tidy-up.",
            recent_event_ids=[event.event_id],
            context_summary=self._summary(),
        )

    def _react_error(self, event: Event) -> None:
        from datetime import datetime

        data = event.data or {}
        msg = str(data.get("msg") or data.get("error") or data.get("message") or "unknown")
        now = datetime.now()
        prior = self.world.last_error
        suppress = (
            prior is not None
            and prior.get("msg") == msg
            and (now - prior["ts"]).total_seconds() < self.ERROR_SUPPRESS_WINDOW_SEC
        )
        self.world.last_error = {"msg": msg, "ts": now}
        if suppress:
            self.log_decision(
                trigger=DecisionTrigger.EVENT,
                trigger_detail="ERROR_OCCURRED",
                tool_calls=[],
                response_text=(
                    f"Suppressed repeat error within"
                    f" {self.ERROR_SUPPRESS_WINDOW_SEC:.0f}s window: {msg[:120]}"
                ),
                recent_event_ids=[event.event_id],
                context_summary=self._summary(),
            )
            return
        self.log_decision(
            trigger=DecisionTrigger.EVENT,
            trigger_detail="ERROR_OCCURRED",
            tool_calls=[
                {
                    "name": "escalate_to_operator",
                    "input": {"error_message": msg, "source": event.source},
                    "id": None,
                }
            ],
            response_text=f"New error -- proposing escalation: {msg[:120]}",
            recent_event_ids=[event.event_id],
            context_summary=self._summary(),
        )

    # ---- helpers -----------------------------------------------------------

    def _summary(self) -> str:
        n_emb = len(self.world.embryos)
        n_fine = sum(1 for v in self.world.embryos.values() if v.get("has_fine"))
        stage = self.world.last_stage_um
        stage_str = f"({stage['x']:.1f}, {stage['y']:.1f})" if stage else "unknown"
        seen = sum(self.world.seen.values())
        return (
            f"{n_emb} embryos ({n_fine} fine-calibrated); stage {stage_str}; {seen} events ingested"
        )
