"""
UnderstandingMixin — Observations, expectations, watchpoints, questions,
learnings, embryo understanding, agent state, and batch updates.

Mixed into ContextStore; relies on self._conn, self._tx(), self._now(),
self._gen_id() provided by the host class.
"""

import json
import logging
import sqlite3
from datetime import datetime

from ._protocols import StoreProtocol
from .model import (
    Attention,
    Confidence,
    ContextUpdates,
    EmbryoUnderstanding,
    Expectation,
    ExpectationStatus,
    Learning,
    Observation,
    Question,
    QuestionStatus,
    Significance,
    Understanding,
    Watchpoint,
    WatchpointStatus,
)

logger = logging.getLogger(__name__)


class UnderstandingMixin(StoreProtocol):
    """Observations, expectations, watchpoints, questions, learnings,
    embryo understanding, agent state, and batch updates."""

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_understanding(self) -> Understanding:
        return Understanding(
            embryo_states=self._load_embryo_states(),
            learnings=self.get_learnings(),
        )

    def _load_embryo_states(self) -> dict[str, EmbryoUnderstanding]:
        rows = self._conn.execute(
            "SELECT * FROM embryo_understanding WHERE is_tracked = 1"
        ).fetchall()
        result = {}
        for row in rows:
            d = dict(row)
            result[d["embryo_id"]] = EmbryoUnderstanding(
                embryo_id=d["embryo_id"],
                current_stage=d.get("current_stage"),
                stage_confidence=Confidence(d["stage_confidence"])
                if d.get("stage_confidence")
                else None,
                health_assessment=d.get("health_assessment"),
                notes=json.loads(d["notes"]) if d.get("notes") else [],
                last_observed=datetime.fromisoformat(d["last_observed"])
                if d.get("last_observed")
                else None,
                is_tracked=bool(d.get("is_tracked", True)),
                is_hatched=bool(d.get("is_hatched", False)),
                needs_attention=bool(d.get("needs_attention", False)),
                attention_reason=d.get("attention_reason"),
            )
        return result

    def _load_attention(self) -> Attention:
        return Attention(
            watchpoints=self.get_active_watchpoints(),
            open_questions=self.get_open_questions(),
        )

    # ==================================================================
    # Observations
    # ==================================================================

    def add_observation(self, obs: Observation):
        """Add an observation."""
        with self._tx():
            self._conn.execute(
                "INSERT INTO observations "
                "(id, timestamp, type, content, embryo_id, significance, session_id, "
                " gently_refs, relates_to) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    obs.id,
                    obs.timestamp.isoformat(),
                    obs.type,
                    obs.content,
                    obs.embryo_id,
                    obs.significance.value,
                    obs.session_id,
                    json.dumps(obs.gently_refs) if obs.gently_refs else None,
                    json.dumps(obs.relates_to) if obs.relates_to else None,
                ),
            )

    def get_recent_observations(self, limit: int = 50) -> list[Observation]:
        """Get recent observations."""
        rows = self._conn.execute(
            "SELECT * FROM observations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_observation(row) for row in reversed(rows)]

    def get_observations_for_embryo(self, embryo_id: str, limit: int = 20) -> list[Observation]:
        """Get observations for a specific embryo."""
        rows = self._conn.execute(
            "SELECT * FROM observations WHERE embryo_id = ? ORDER BY timestamp DESC LIMIT ?",
            (embryo_id, limit),
        ).fetchall()
        return [self._row_to_observation(row) for row in reversed(rows)]

    def _row_to_observation(self, row: sqlite3.Row) -> Observation:
        d = dict(row)
        return Observation(
            id=d["id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            type=d["type"],
            content=d["content"],
            embryo_id=d.get("embryo_id"),
            significance=Significance(d.get("significance", "medium")),
            session_id=d.get("session_id"),
            gently_refs=json.loads(d["gently_refs"]) if d.get("gently_refs") else None,
            relates_to=json.loads(d["relates_to"]) if d.get("relates_to") else None,
        )

    # ==================================================================
    # Expectations
    # ==================================================================

    def add_expectation(self, exp: Expectation):
        """Add an expectation."""
        with self._tx():
            self._conn.execute(
                "INSERT INTO expectations "
                "(id, target, prediction, expected_time, uncertainty, basis, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    exp.id,
                    exp.target,
                    exp.prediction,
                    exp.expected_time.isoformat(),
                    exp.uncertainty,
                    exp.basis,
                    exp.status.value,
                    exp.created_at.isoformat(),
                ),
            )

    def get_pending_expectations(self) -> list[Expectation]:
        """Get all pending expectations."""
        rows = self._conn.execute(
            "SELECT * FROM expectations WHERE status = 'pending' ORDER BY expected_time"
        ).fetchall()
        return [self._row_to_expectation(row) for row in rows]

    def get_expectation_for(self, target: str) -> Expectation | None:
        """Get the pending expectation for a specific target."""
        row = self._conn.execute(
            "SELECT * FROM expectations WHERE target = ? AND status = 'pending' "
            "ORDER BY expected_time LIMIT 1",
            (target,),
        ).fetchone()
        return self._row_to_expectation(row) if row else None

    def resolve_expectation(self, exp_id: str, status: ExpectationStatus):
        """Resolve an expectation."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "UPDATE expectations SET status = ?, resolved_at = ? WHERE id = ?",
                (status.value, now, exp_id),
            )

    def _row_to_expectation(self, row: sqlite3.Row) -> Expectation:
        d = dict(row)
        return Expectation(
            id=d["id"],
            target=d["target"],
            prediction=d["prediction"],
            expected_time=datetime.fromisoformat(d["expected_time"]),
            uncertainty=d.get("uncertainty"),
            basis=d.get("basis"),
            status=ExpectationStatus(d.get("status", "pending")),
            created_at=datetime.fromisoformat(d["created_at"]),
            resolved_at=datetime.fromisoformat(d["resolved_at"]) if d.get("resolved_at") else None,
        )

    # ==================================================================
    # Watchpoints
    # ==================================================================

    def add_watchpoint(self, wp: Watchpoint):
        """Add a watchpoint."""
        with self._tx():
            self._conn.execute(
                "INSERT INTO watchpoints (id, target, condition, priority, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    wp.id,
                    wp.target,
                    wp.condition,
                    wp.priority.value,
                    wp.status.value,
                    wp.created_at.isoformat(),
                ),
            )

    def get_active_watchpoints(self) -> list[Watchpoint]:
        """Get all active watchpoints."""
        rows = self._conn.execute(
            "SELECT * FROM watchpoints WHERE status = 'active' ORDER BY priority DESC, created_at"
        ).fetchall()
        return [self._row_to_watchpoint(row) for row in rows]

    def trigger_watchpoint(self, wp_id: str):
        """Mark a watchpoint as triggered."""
        with self._tx():
            self._conn.execute(
                "UPDATE watchpoints SET status = 'triggered' WHERE id = ?",
                (wp_id,),
            )

    def resolve_watchpoint(self, wp_id: str):
        """Mark a watchpoint as resolved."""
        with self._tx():
            self._conn.execute(
                "UPDATE watchpoints SET status = 'resolved' WHERE id = ?",
                (wp_id,),
            )

    def _row_to_watchpoint(self, row: sqlite3.Row) -> Watchpoint:
        d = dict(row)
        return Watchpoint(
            id=d["id"],
            target=d["target"],
            condition=d["condition"],
            priority=Significance(d.get("priority", "medium")),
            status=WatchpointStatus(d.get("status", "active")),
            created_at=datetime.fromisoformat(d["created_at"]),
        )

    # ==================================================================
    # Questions
    # ==================================================================

    def add_question(self, q: Question):
        """Add a question."""
        with self._tx():
            self._conn.execute(
                "INSERT INTO questions (id, content, status, created_at) VALUES (?, ?, ?, ?)",
                (q.id, q.content, q.status.value, q.created_at.isoformat()),
            )

    def get_open_questions(self) -> list[Question]:
        """Get all open questions."""
        rows = self._conn.execute(
            "SELECT * FROM questions WHERE status IN ('open', 'investigating') ORDER BY created_at"
        ).fetchall()
        return [self._row_to_question(row) for row in rows]

    def resolve_question(self, q_id: str, resolution: str):
        """Resolve a question."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "UPDATE questions SET status = 'resolved', resolution = ?, resolved_at = ? "
                "WHERE id = ?",
                (resolution, now, q_id),
            )

    def _row_to_question(self, row: sqlite3.Row) -> Question:
        d = dict(row)
        return Question(
            id=d["id"],
            content=d["content"],
            status=QuestionStatus(d.get("status", "open")),
            resolution=d.get("resolution"),
            created_at=datetime.fromisoformat(d["created_at"]),
            resolved_at=datetime.fromisoformat(d["resolved_at"]) if d.get("resolved_at") else None,
        )

    # ==================================================================
    # Learnings
    # ==================================================================

    def add_learning(self, learning: Learning):
        """Add a learning."""
        with self._tx():
            self._conn.execute(
                "INSERT INTO learnings (id, content, confidence, basis, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    learning.id,
                    learning.content,
                    learning.confidence.value,
                    learning.basis,
                    learning.created_at.isoformat(),
                ),
            )

    def get_learnings(self, limit: int = 50) -> list[Learning]:
        """Get learnings."""
        rows = self._conn.execute(
            "SELECT * FROM learnings ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_learning(row) for row in rows]

    def _row_to_learning(self, row: sqlite3.Row) -> Learning:
        d = dict(row)
        return Learning(
            id=d["id"],
            content=d["content"],
            confidence=Confidence(d.get("confidence", "medium")),
            basis=d.get("basis"),
            created_at=datetime.fromisoformat(d["created_at"]),
        )

    # ==================================================================
    # Embryo Understanding
    # ==================================================================

    def update_embryo_understanding(
        self,
        embryo_id: str,
        current_stage: str | None = None,
        stage_confidence: Confidence | None = None,
        health_assessment: str | None = None,
        note: str | None = None,
        is_hatched: bool | None = None,
        needs_attention: bool | None = None,
        attention_reason: str | None = None,
    ):
        """Update understanding of an embryo."""
        now = self._now()

        # Get existing or create new
        existing = self._conn.execute(
            "SELECT * FROM embryo_understanding WHERE embryo_id = ?",
            (embryo_id,),
        ).fetchone()

        if existing:
            existing = dict(existing)
            notes = json.loads(existing.get("notes") or "[]")
            if note:
                notes.append(note)

            with self._tx():
                self._conn.execute(
                    "UPDATE embryo_understanding SET "
                    "current_stage = COALESCE(?, current_stage), "
                    "stage_confidence = COALESCE(?, stage_confidence), "
                    "health_assessment = COALESCE(?, health_assessment), "
                    "notes = ?, "
                    "last_observed = ?, "
                    "is_hatched = COALESCE(?, is_hatched), "
                    "needs_attention = COALESCE(?, needs_attention), "
                    "attention_reason = COALESCE(?, attention_reason) "
                    "WHERE embryo_id = ?",
                    (
                        current_stage,
                        stage_confidence.value if stage_confidence else None,
                        health_assessment,
                        json.dumps(notes),
                        now,
                        1 if is_hatched else (0 if is_hatched is False else None),
                        1 if needs_attention else (0 if needs_attention is False else None),
                        attention_reason,
                        embryo_id,
                    ),
                )
        else:
            notes = [note] if note else []
            with self._tx():
                self._conn.execute(
                    "INSERT INTO embryo_understanding "
                    "(embryo_id, current_stage, stage_confidence, health_assessment, notes, "
                    " last_observed, is_hatched, needs_attention, attention_reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        embryo_id,
                        current_stage,
                        stage_confidence.value if stage_confidence else None,
                        health_assessment,
                        json.dumps(notes),
                        now,
                        1 if is_hatched else 0,
                        1 if needs_attention else 0,
                        attention_reason,
                    ),
                )

    # ==================================================================
    # Agent State
    # ==================================================================

    def get_state(self, key: str) -> str | None:
        """Get a state value."""
        row = self._conn.execute("SELECT value FROM agent_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str):
        """Set a state value."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO agent_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )

    # ==================================================================
    # Apply Batch Updates
    # ==================================================================

    def apply_updates(self, updates: ContextUpdates):
        """
        Apply a batch of updates from the agent.

        This is called after each thinking cycle.
        """
        # Add new items
        for obs in updates.new_observations:
            self.add_observation(obs)

        for exp in updates.new_expectations:
            self.add_expectation(exp)

        for wp in updates.new_watchpoints:
            self.add_watchpoint(wp)

        for learning in updates.new_learnings:
            self.add_learning(learning)

        for q in updates.new_questions:
            self.add_question(q)

        # Resolve expectations
        for exp_id, status in updates.resolved_expectations.items():
            self.resolve_expectation(exp_id, status)

        # Trigger watchpoints
        for wp_id in updates.triggered_watchpoints:
            self.trigger_watchpoint(wp_id)

        # Resolve questions
        for q_id, resolution in updates.resolved_questions.items():
            self.resolve_question(q_id, resolution)

        # Update embryo understanding
        for embryo_id, update_dict in updates.embryo_updates.items():
            self.update_embryo_understanding(embryo_id, **update_dict)

        # Update campaign progress
        for campaign_id, progress in updates.campaign_progress.items():
            self.update_campaign_progress(campaign_id, progress)

        # Update focus
        if updates.new_focus is not None:
            self.set_state("current_focus", updates.new_focus)
