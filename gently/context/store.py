"""
ContextStore — SQLite-backed storage for the agent's mind.

Separate from GentlyStore. This holds understanding, not raw data.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from .model import (
    Campaign,
    Project,
    SessionIntent,
    Learning,
    Observation,
    Expectation,
    Watchpoint,
    Question,
    EmbryoUnderstanding,
    Intentions,
    Understanding,
    Attention,
    Context,
    ContextUpdates,
    Status,
    Confidence,
    Significance,
    ExpectationStatus,
    WatchpointStatus,
    QuestionStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """\
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

-- Intentions: why are we doing this?
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    shorthand TEXT,
    summary TEXT,
    target TEXT,
    progress TEXT,
    parent_id TEXT,
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES campaigns(id)
);

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    campaign_id TEXT,
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
);

CREATE TABLE IF NOT EXISTS session_intents (
    session_id TEXT PRIMARY KEY,
    planned_intent TEXT,
    actual_summary TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

-- Many-to-many: a session can contribute to multiple campaigns
CREATE TABLE IF NOT EXISTS session_campaigns (
    session_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    PRIMARY KEY (session_id, campaign_id),
    FOREIGN KEY (session_id) REFERENCES session_intents(session_id),
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
);

-- Understanding: what do we believe?
CREATE TABLE IF NOT EXISTS learnings (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    confidence TEXT DEFAULT 'medium',
    basis TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embryo_understanding (
    embryo_id TEXT PRIMARY KEY,
    current_stage TEXT,
    stage_confidence TEXT,
    health_assessment TEXT,
    notes TEXT,
    last_observed TEXT,
    is_tracked INTEGER DEFAULT 1,
    is_hatched INTEGER DEFAULT 0,
    needs_attention INTEGER DEFAULT 0,
    attention_reason TEXT
);

-- Observations: what have we seen? (synthesized, not raw)
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    embryo_id TEXT,
    significance TEXT DEFAULT 'medium',
    session_id TEXT,
    gently_refs TEXT,
    relates_to TEXT
);

-- Expectations: what do we predict?
CREATE TABLE IF NOT EXISTS expectations (
    id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    prediction TEXT NOT NULL,
    expected_time TEXT NOT NULL,
    uncertainty TEXT,
    basis TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

-- Attention: what should we watch?
CREATE TABLE IF NOT EXISTS watchpoints (
    id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    condition TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    status TEXT DEFAULT 'open',
    resolution TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

-- Agent state
CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_observations_timestamp ON observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_embryo ON observations(embryo_id);
CREATE INDEX IF NOT EXISTS idx_expectations_status ON expectations(status);
CREATE INDEX IF NOT EXISTS idx_watchpoints_status ON watchpoints(status);
CREATE INDEX IF NOT EXISTS idx_campaigns_status ON campaigns(status);
CREATE INDEX IF NOT EXISTS idx_campaigns_parent ON campaigns(parent_id);
CREATE INDEX IF NOT EXISTS idx_session_campaigns_session ON session_campaigns(session_id);
CREATE INDEX IF NOT EXISTS idx_session_campaigns_campaign ON session_campaigns(campaign_id);
"""


# ---------------------------------------------------------------------------
# ContextStore
# ---------------------------------------------------------------------------

class ContextStore:
    """
    SQLite-backed storage for the agent's context.

    This is the agent's mind — understanding, not raw data.
    """

    def __init__(self, db_path: Path):
        """
        Parameters
        ----------
        db_path : Path
            Path to SQLite database file. Created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_db()

    def _open_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        return conn

    @contextmanager
    def _tx(self):
        """Context manager that commits on success, rolls back on error."""
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _gen_id(self) -> str:
        return str(uuid.uuid4())[:8]

    # ==================================================================
    # Load Active Context
    # ==================================================================

    def load_active(self) -> Context:
        """
        Load the active context for the agent.

        This is what gets passed to the agent each thinking cycle.
        """
        return Context(
            intentions=self._load_intentions(),
            understanding=self._load_understanding(),
            observations=self.get_recent_observations(limit=50),
            expectations=self.get_pending_expectations(),
            attention=self._load_attention(),
        )

    def _load_intentions(self) -> Intentions:
        return Intentions(
            campaigns=self.get_active_campaigns(),
            projects=self.get_active_projects(),
            current_focus=self.get_state("current_focus"),
            session_intent=self.get_current_session_intent(),
        )

    def _load_understanding(self) -> Understanding:
        return Understanding(
            embryo_states=self._load_embryo_states(),
            learnings=self.get_learnings(),
        )

    def _load_embryo_states(self) -> Dict[str, EmbryoUnderstanding]:
        rows = self._conn.execute(
            "SELECT * FROM embryo_understanding WHERE is_tracked = 1"
        ).fetchall()
        result = {}
        for row in rows:
            d = dict(row)
            result[d["embryo_id"]] = EmbryoUnderstanding(
                embryo_id=d["embryo_id"],
                current_stage=d.get("current_stage"),
                stage_confidence=Confidence(d["stage_confidence"]) if d.get("stage_confidence") else None,
                health_assessment=d.get("health_assessment"),
                notes=json.loads(d["notes"]) if d.get("notes") else [],
                last_observed=datetime.fromisoformat(d["last_observed"]) if d.get("last_observed") else None,
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
    # Campaigns
    # ==================================================================

    def create_campaign(
        self,
        description: str,
        shorthand: Optional[str] = None,
        summary: Optional[str] = None,
        target: Optional[str] = None,
        parent_id: Optional[str] = None,
        campaign_id: Optional[str] = None,
    ) -> str:
        """Create a new campaign. Returns campaign ID."""
        cid = campaign_id or self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO campaigns "
                "(id, description, shorthand, summary, target, parent_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (cid, description, shorthand, summary, target, parent_id, now, now),
            )
        label = shorthand or description[:50]
        logger.info(f"Created campaign {cid} [{label}]")
        return cid

    def get_active_campaigns(self) -> List[Campaign]:
        """Get all active campaigns."""
        rows = self._conn.execute(
            "SELECT * FROM campaigns WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get a specific campaign."""
        row = self._conn.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        return self._row_to_campaign(row) if row else None

    def update_campaign_progress(self, campaign_id: str, progress: str):
        """Update campaign progress."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "UPDATE campaigns SET progress = ?, updated_at = ? WHERE id = ?",
                (progress, now, campaign_id),
            )

    def update_campaign_status(self, campaign_id: str, status: Status):
        """Update campaign status."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "UPDATE campaigns SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, campaign_id),
            )

    def get_subcampaigns(self, campaign_id: str) -> List[Campaign]:
        """Get direct children of a campaign."""
        rows = self._conn.execute(
            "SELECT * FROM campaigns WHERE parent_id = ? ORDER BY created_at",
            (campaign_id,),
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def get_campaign_tree(self, campaign_id: str) -> Dict[str, Any]:
        """Get a campaign and all its descendants as a tree."""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {}
        children = self.get_subcampaigns(campaign_id)
        return {
            "campaign": campaign,
            "children": [self.get_campaign_tree(c.id) for c in children],
        }

    def get_root_campaigns(self) -> List[Campaign]:
        """Get top-level campaigns (no parent)."""
        rows = self._conn.execute(
            "SELECT * FROM campaigns WHERE parent_id IS NULL AND status = 'active' "
            "ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def update_campaign(
        self,
        campaign_id: str,
        description: Optional[str] = None,
        shorthand: Optional[str] = None,
        summary: Optional[str] = None,
        target: Optional[str] = None,
        parent_id: Optional[str] = None,
    ):
        """Update campaign fields. Only non-None values are applied."""
        now = self._now()
        updates = []
        values = []
        for col, val in [
            ("description", description),
            ("shorthand", shorthand),
            ("summary", summary),
            ("target", target),
            ("parent_id", parent_id),
        ]:
            if val is not None:
                updates.append(f"{col} = ?")
                values.append(val)
        if not updates:
            return
        updates.append("updated_at = ?")
        values.append(now)
        values.append(campaign_id)
        with self._tx():
            self._conn.execute(
                f"UPDATE campaigns SET {', '.join(updates)} WHERE id = ?",
                values,
            )

    def _row_to_campaign(self, row: sqlite3.Row) -> Campaign:
        d = dict(row)
        return Campaign(
            id=d["id"],
            description=d["description"],
            shorthand=d.get("shorthand"),
            summary=d.get("summary"),
            target=d.get("target"),
            progress=d.get("progress"),
            parent_id=d.get("parent_id"),
            status=Status(d.get("status", "active")),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )

    # ==================================================================
    # Projects
    # ==================================================================

    def create_project(
        self,
        description: str,
        campaign_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """Create a new project. Returns project ID."""
        pid = project_id or self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO projects (id, description, campaign_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', ?, ?)",
                (pid, description, campaign_id, now, now),
            )
        logger.info(f"Created project {pid}: {description}")
        return pid

    def get_active_projects(self) -> List[Project]:
        """Get all active projects."""
        rows = self._conn.execute(
            "SELECT * FROM projects WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_project(row) for row in rows]

    def _row_to_project(self, row: sqlite3.Row) -> Project:
        d = dict(row)
        return Project(
            id=d["id"],
            description=d["description"],
            campaign_id=d.get("campaign_id"),
            status=Status(d.get("status", "active")),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )

    # ==================================================================
    # Session Intents
    # ==================================================================

    def create_session_intent(
        self,
        session_id: str,
        planned_intent: Optional[str] = None,
        campaign_ids: Optional[List[str]] = None,
    ):
        """Create or update session intent, optionally linking to campaigns."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO session_intents "
                "(session_id, planned_intent, created_at) "
                "VALUES (?, ?, ?)",
                (session_id, planned_intent, now),
            )
        if campaign_ids:
            for cid in campaign_ids:
                self.link_session_campaign(session_id, cid)

    def get_current_session_intent(self) -> Optional[SessionIntent]:
        """Get the most recent incomplete session intent."""
        row = self._conn.execute(
            "SELECT * FROM session_intents WHERE completed_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        session_id = d["session_id"]
        campaign_ids = self.get_campaign_ids_for_session(session_id)
        return SessionIntent(
            session_id=session_id,
            planned_intent=d.get("planned_intent"),
            actual_summary=d.get("actual_summary"),
            campaign_ids=campaign_ids,
            created_at=datetime.fromisoformat(d["created_at"]),
            completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
        )

    def complete_session_intent(self, session_id: str, actual_summary: str):
        """Mark session intent as completed with summary."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "UPDATE session_intents SET actual_summary = ?, completed_at = ? "
                "WHERE session_id = ?",
                (actual_summary, now, session_id),
            )

    # ==================================================================
    # Session ↔ Campaign (many-to-many)
    # ==================================================================

    def link_session_campaign(self, session_id: str, campaign_id: str):
        """Link a session to a campaign."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO session_campaigns "
                "(session_id, campaign_id, linked_at) VALUES (?, ?, ?)",
                (session_id, campaign_id, now),
            )

    def unlink_session_campaign(self, session_id: str, campaign_id: str):
        """Unlink a session from a campaign."""
        with self._tx():
            self._conn.execute(
                "DELETE FROM session_campaigns WHERE session_id = ? AND campaign_id = ?",
                (session_id, campaign_id),
            )

    def get_campaign_ids_for_session(self, session_id: str) -> List[str]:
        """Get campaign IDs linked to a session."""
        rows = self._conn.execute(
            "SELECT campaign_id FROM session_campaigns WHERE session_id = ? "
            "ORDER BY linked_at",
            (session_id,),
        ).fetchall()
        return [row["campaign_id"] for row in rows]

    def get_campaigns_for_session(self, session_id: str) -> List[Campaign]:
        """Get campaigns linked to a session."""
        rows = self._conn.execute(
            "SELECT c.* FROM campaigns c "
            "JOIN session_campaigns sc ON c.id = sc.campaign_id "
            "WHERE sc.session_id = ? ORDER BY sc.linked_at",
            (session_id,),
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def get_sessions_for_campaign(self, campaign_id: str) -> List[SessionIntent]:
        """Get session intents linked to a campaign."""
        rows = self._conn.execute(
            "SELECT si.* FROM session_intents si "
            "JOIN session_campaigns sc ON si.session_id = sc.session_id "
            "WHERE sc.campaign_id = ? ORDER BY si.created_at",
            (campaign_id,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            sid = d["session_id"]
            cids = self.get_campaign_ids_for_session(sid)
            results.append(SessionIntent(
                session_id=sid,
                planned_intent=d.get("planned_intent"),
                actual_summary=d.get("actual_summary"),
                campaign_ids=cids,
                created_at=datetime.fromisoformat(d["created_at"]),
                completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
            ))
        return results

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

    def get_recent_observations(self, limit: int = 50) -> List[Observation]:
        """Get recent observations."""
        rows = self._conn.execute(
            "SELECT * FROM observations ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_observation(row) for row in reversed(rows)]

    def get_observations_for_embryo(self, embryo_id: str, limit: int = 20) -> List[Observation]:
        """Get observations for a specific embryo."""
        rows = self._conn.execute(
            "SELECT * FROM observations WHERE embryo_id = ? "
            "ORDER BY timestamp DESC LIMIT ?",
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

    def get_pending_expectations(self) -> List[Expectation]:
        """Get all pending expectations."""
        rows = self._conn.execute(
            "SELECT * FROM expectations WHERE status = 'pending' ORDER BY expected_time"
        ).fetchall()
        return [self._row_to_expectation(row) for row in rows]

    def get_expectation_for(self, target: str) -> Optional[Expectation]:
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

    def get_active_watchpoints(self) -> List[Watchpoint]:
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
                "INSERT INTO questions (id, content, status, created_at) "
                "VALUES (?, ?, ?, ?)",
                (q.id, q.content, q.status.value, q.created_at.isoformat()),
            )

    def get_open_questions(self) -> List[Question]:
        """Get all open questions."""
        rows = self._conn.execute(
            "SELECT * FROM questions WHERE status IN ('open', 'investigating') "
            "ORDER BY created_at"
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

    def get_learnings(self, limit: int = 50) -> List[Learning]:
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
        current_stage: Optional[str] = None,
        stage_confidence: Optional[Confidence] = None,
        health_assessment: Optional[str] = None,
        note: Optional[str] = None,
        is_hatched: Optional[bool] = None,
        needs_attention: Optional[bool] = None,
        attention_reason: Optional[str] = None,
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

    def get_state(self, key: str) -> Optional[str]:
        """Get a state value."""
        row = self._conn.execute(
            "SELECT value FROM agent_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str):
        """Set a state value."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO agent_state (key, value, updated_at) "
                "VALUES (?, ?, ?)",
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

    # ==================================================================
    # Utility
    # ==================================================================

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("ContextStore closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"ContextStore(db_path={self.db_path})"
