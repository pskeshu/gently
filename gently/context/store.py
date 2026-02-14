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
    PlannedSession,
    PlannedSessionStatus,
    PlanItem,
    PlanItemStatus,
    PlanItemType,
    ImagingSpec,
    BenchSpec,
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

-- Project calendar: planned imaging sessions
CREATE TABLE IF NOT EXISTS planned_sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    notes TEXT,
    scheduled_date TEXT,
    scheduled_time TEXT,
    estimated_duration_minutes INTEGER,
    acquisition_params TEXT,
    source_session_id TEXT,
    status TEXT DEFAULT 'planned',
    session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Many-to-many: a planned session can serve multiple campaigns
CREATE TABLE IF NOT EXISTS planned_session_campaigns (
    planned_session_id TEXT NOT NULL,
    campaign_id TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    PRIMARY KEY (planned_session_id, campaign_id),
    FOREIGN KEY (planned_session_id) REFERENCES planned_sessions(id),
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

-- Plan items: tasks in an experimental plan
CREATE TABLE IF NOT EXISTS plan_items (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'planned',
    outcome TEXT,
    spec TEXT,
    inherit_from TEXT,
    planned_session_id TEXT,
    session_id TEXT,
    phase_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
    FOREIGN KEY (planned_session_id) REFERENCES planned_sessions(id),
    FOREIGN KEY (inherit_from) REFERENCES plan_items(id)
);

-- Plan item dependencies
CREATE TABLE IF NOT EXISTS plan_item_dependencies (
    item_id TEXT NOT NULL,
    depends_on_id TEXT NOT NULL,
    PRIMARY KEY (item_id, depends_on_id),
    FOREIGN KEY (item_id) REFERENCES plan_items(id),
    FOREIGN KEY (depends_on_id) REFERENCES plan_items(id)
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
CREATE INDEX IF NOT EXISTS idx_planned_sessions_date ON planned_sessions(scheduled_date);
CREATE INDEX IF NOT EXISTS idx_planned_sessions_status ON planned_sessions(status);
CREATE INDEX IF NOT EXISTS idx_planned_session_campaigns_ps ON planned_session_campaigns(planned_session_id);
CREATE INDEX IF NOT EXISTS idx_planned_session_campaigns_c ON planned_session_campaigns(campaign_id);
CREATE INDEX IF NOT EXISTS idx_plan_items_campaign ON plan_items(campaign_id);
CREATE INDEX IF NOT EXISTS idx_plan_items_status ON plan_items(status);
CREATE INDEX IF NOT EXISTS idx_plan_items_type ON plan_items(type);
CREATE INDEX IF NOT EXISTS idx_plan_items_inherit ON plan_items(inherit_from);
CREATE INDEX IF NOT EXISTS idx_plan_item_deps_item ON plan_item_dependencies(item_id);
CREATE INDEX IF NOT EXISTS idx_plan_item_deps_dep ON plan_item_dependencies(depends_on_id);
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

    def reset(self) -> dict:
        """Drop all data from every table. The schema is preserved.

        Returns a dict of table_name → rows_deleted.
        """
        tables = [
            "plan_item_dependencies",
            "plan_items",
            "planned_session_campaigns",
            "session_campaigns",
            "planned_sessions",
            "session_intents",
            "projects",
            "campaigns",
            "learnings",
            "embryo_understanding",
            "observations",
            "expectations",
            "watchpoints",
            "questions",
            "agent_state",
        ]
        counts = {}
        with self._tx() as conn:
            for t in tables:
                before = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                conn.execute(f"DELETE FROM {t}")
                if before > 0:
                    counts[t] = before
        total = sum(counts.values())
        logger.info(f"Context store reset — {total} rows cleared from {len(counts)} tables")
        return counts

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
            planned_sessions=self.get_upcoming_sessions(limit=5),
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

    def delete_campaign(self, campaign_id: str, cascade: bool = True) -> Dict[str, int]:
        """
        Delete a campaign and optionally its children and plan items.

        Parameters
        ----------
        campaign_id : str
            Campaign to delete.
        cascade : bool
            If True, also delete subcampaigns, plan items, and dependencies.

        Returns
        -------
        dict
            Counts of deleted records by type.
        """
        counts = {"campaigns": 0, "plan_items": 0, "dependencies": 0}

        def _delete_recursive(cid: str):
            if cascade:
                # Delete children first
                children = self.get_subcampaigns(cid)
                for child in children:
                    _delete_recursive(child.id)

            # Delete plan item dependencies for items in this campaign
            items = self.get_plan_items(campaign_id=cid)
            for item in items:
                r = self._conn.execute(
                    "DELETE FROM plan_item_dependencies WHERE item_id = ? OR depends_on_id = ?",
                    (item.id, item.id),
                )
                counts["dependencies"] += r.rowcount

            # Delete plan items
            r = self._conn.execute(
                "DELETE FROM plan_items WHERE campaign_id = ?", (cid,),
            )
            counts["plan_items"] += r.rowcount

            # Delete campaign
            r = self._conn.execute(
                "DELETE FROM campaigns WHERE id = ?", (cid,),
            )
            counts["campaigns"] += r.rowcount

        with self._tx():
            _delete_recursive(campaign_id)

        return counts

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
            # Ensure session_intents row exists (FK target)
            self._conn.execute(
                "INSERT OR IGNORE INTO session_intents "
                "(session_id, created_at) VALUES (?, ?)",
                (session_id, now),
            )
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
    # Planned Sessions (project calendar)
    # ==================================================================

    def create_planned_session(
        self,
        scheduled_date: str,
        title: Optional[str] = None,
        notes: Optional[str] = None,
        scheduled_time: Optional[str] = None,
        estimated_duration_minutes: Optional[int] = None,
        acquisition_params: Optional[Dict] = None,
        source_session_id: Optional[str] = None,
        campaign_ids: Optional[List[str]] = None,
        planned_session_id: Optional[str] = None,
    ) -> str:
        """Create a planned imaging session. Returns its ID."""
        psid = planned_session_id or self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO planned_sessions "
                "(id, title, notes, scheduled_date, scheduled_time, "
                " estimated_duration_minutes, acquisition_params, "
                " source_session_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?)",
                (
                    psid, title, notes, scheduled_date, scheduled_time,
                    estimated_duration_minutes,
                    json.dumps(acquisition_params) if acquisition_params else None,
                    source_session_id, now, now,
                ),
            )
        if campaign_ids:
            for cid in campaign_ids:
                self.link_planned_session_campaign(psid, cid)
        logger.info(f"Created planned session {psid} for {scheduled_date}: {title or notes or '(untitled)'}")
        return psid

    def get_planned_session(self, planned_session_id: str) -> Optional[PlannedSession]:
        """Get a specific planned session."""
        row = self._conn.execute(
            "SELECT * FROM planned_sessions WHERE id = ?",
            (planned_session_id,),
        ).fetchone()
        return self._row_to_planned_session(row) if row else None

    def get_planned_sessions(
        self,
        status: Optional[str] = None,
        campaign_id: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[PlannedSession]:
        """
        Query planned sessions with optional filters.

        Parameters
        ----------
        status : str, optional
            Filter by status ("planned", "active", "completed", etc.)
        campaign_id : str, optional
            Filter to sessions linked to this campaign.
        from_date, to_date : str, optional
            ISO date range filter (inclusive).
        """
        if campaign_id:
            query = (
                "SELECT ps.* FROM planned_sessions ps "
                "JOIN planned_session_campaigns psc ON ps.id = psc.planned_session_id "
                "WHERE psc.campaign_id = ?"
            )
            params: list = [campaign_id]
        else:
            query = "SELECT * FROM planned_sessions WHERE 1=1"
            params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if from_date:
            query += " AND scheduled_date >= ?"
            params.append(from_date)
        if to_date:
            query += " AND scheduled_date <= ?"
            params.append(to_date)

        query += " ORDER BY scheduled_date, scheduled_time"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_planned_session(row) for row in rows]

    def get_upcoming_sessions(self, limit: int = 10) -> List[PlannedSession]:
        """Get upcoming planned sessions (today and future, status=planned)."""
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self._conn.execute(
            "SELECT * FROM planned_sessions "
            "WHERE status = 'planned' AND scheduled_date >= ? "
            "ORDER BY scheduled_date, scheduled_time LIMIT ?",
            (today, limit),
        ).fetchall()
        return [self._row_to_planned_session(row) for row in rows]

    def get_todays_sessions(self) -> List[PlannedSession]:
        """Get planned sessions for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self._conn.execute(
            "SELECT * FROM planned_sessions "
            "WHERE scheduled_date = ? AND status IN ('planned', 'active') "
            "ORDER BY scheduled_time",
            (today,),
        ).fetchall()
        return [self._row_to_planned_session(row) for row in rows]

    def update_planned_session(
        self,
        planned_session_id: str,
        title: Optional[str] = None,
        notes: Optional[str] = None,
        scheduled_date: Optional[str] = None,
        scheduled_time: Optional[str] = None,
        estimated_duration_minutes: Optional[int] = None,
        acquisition_params: Optional[Dict] = None,
        source_session_id: Optional[str] = None,
        status: Optional[PlannedSessionStatus] = None,
        session_id: Optional[str] = None,
    ):
        """Update a planned session. Only non-None values are applied."""
        now = self._now()
        updates = []
        values = []
        for col, val in [
            ("title", title),
            ("notes", notes),
            ("scheduled_date", scheduled_date),
            ("scheduled_time", scheduled_time),
            ("estimated_duration_minutes", estimated_duration_minutes),
            ("source_session_id", source_session_id),
            ("session_id", session_id),
        ]:
            if val is not None:
                updates.append(f"{col} = ?")
                values.append(val)
        if acquisition_params is not None:
            updates.append("acquisition_params = ?")
            values.append(json.dumps(acquisition_params))
        if status is not None:
            updates.append("status = ?")
            values.append(status.value)
        if not updates:
            return
        updates.append("updated_at = ?")
        values.append(now)
        values.append(planned_session_id)
        with self._tx():
            self._conn.execute(
                f"UPDATE planned_sessions SET {', '.join(updates)} WHERE id = ?",
                values,
            )

    def start_planned_session(self, planned_session_id: str, session_id: str):
        """Mark a planned session as active and link it to the real session."""
        self.update_planned_session(
            planned_session_id,
            status=PlannedSessionStatus.ACTIVE,
            session_id=session_id,
        )

    def complete_planned_session(self, planned_session_id: str):
        """Mark a planned session as completed."""
        self.update_planned_session(
            planned_session_id,
            status=PlannedSessionStatus.COMPLETED,
        )

    def link_planned_session_campaign(self, planned_session_id: str, campaign_id: str):
        """Link a planned session to a campaign."""
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO planned_session_campaigns "
                "(planned_session_id, campaign_id, linked_at) VALUES (?, ?, ?)",
                (planned_session_id, campaign_id, now),
            )

    def unlink_planned_session_campaign(self, planned_session_id: str, campaign_id: str):
        """Unlink a planned session from a campaign."""
        with self._tx():
            self._conn.execute(
                "DELETE FROM planned_session_campaigns "
                "WHERE planned_session_id = ? AND campaign_id = ?",
                (planned_session_id, campaign_id),
            )

    def get_campaign_ids_for_planned_session(self, planned_session_id: str) -> List[str]:
        """Get campaign IDs linked to a planned session."""
        rows = self._conn.execute(
            "SELECT campaign_id FROM planned_session_campaigns "
            "WHERE planned_session_id = ? ORDER BY linked_at",
            (planned_session_id,),
        ).fetchall()
        return [row["campaign_id"] for row in rows]

    def _row_to_planned_session(self, row: sqlite3.Row) -> PlannedSession:
        d = dict(row)
        psid = d["id"]
        campaign_ids = self.get_campaign_ids_for_planned_session(psid)
        return PlannedSession(
            id=psid,
            title=d.get("title"),
            notes=d.get("notes"),
            scheduled_date=d.get("scheduled_date"),
            scheduled_time=d.get("scheduled_time"),
            estimated_duration_minutes=d.get("estimated_duration_minutes"),
            acquisition_params=json.loads(d["acquisition_params"]) if d.get("acquisition_params") else None,
            source_session_id=d.get("source_session_id"),
            status=PlannedSessionStatus(d.get("status", "planned")),
            session_id=d.get("session_id"),
            campaign_ids=campaign_ids,
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )

    # ==================================================================
    # Plan Items (experimental plan)
    # ==================================================================

    def create_plan_item(
        self,
        campaign_id: str,
        type: str,
        title: str,
        description: Optional[str] = None,
        spec: Optional[Dict] = None,
        inherit_from: Optional[str] = None,
        planned_session_id: Optional[str] = None,
        phase_order: int = 0,
        depends_on: Optional[List[str]] = None,
        item_id: Optional[str] = None,
    ) -> str:
        """Create a plan item. Returns its ID."""
        pid = item_id or self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO plan_items "
                "(id, campaign_id, type, title, description, spec, inherit_from, "
                " planned_session_id, phase_order, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?)",
                (
                    pid, campaign_id, type, title, description,
                    json.dumps(spec) if spec else None,
                    inherit_from, planned_session_id, phase_order, now, now,
                ),
            )
            if depends_on:
                for dep_id in depends_on:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO plan_item_dependencies "
                        "(item_id, depends_on_id) VALUES (?, ?)",
                        (pid, dep_id),
                    )
        logger.info(f"Created plan item {pid} [{type}]: {title}")
        return pid

    def get_plan_item(self, item_id: str) -> Optional[PlanItem]:
        """Get a specific plan item."""
        row = self._conn.execute(
            "SELECT * FROM plan_items WHERE id = ?", (item_id,)
        ).fetchone()
        return self._row_to_plan_item(row) if row else None

    def get_plan_items(
        self,
        campaign_id: Optional[str] = None,
        status: Optional[str] = None,
        type: Optional[str] = None,
        include_children: bool = False,
    ) -> List[PlanItem]:
        """
        Query plan items with optional filters.

        Parameters
        ----------
        campaign_id : str, optional
            Filter to items in this campaign. If include_children is True,
            also includes items in child campaigns.
        status : str, optional
            Filter by status.
        type : str, optional
            Filter by type (imaging, bench, etc.).
        include_children : bool
            If True, include items from child campaigns of campaign_id.
        """
        if campaign_id and include_children:
            # Get all campaign IDs in the tree
            campaign_ids = self._get_campaign_tree_ids(campaign_id)
            placeholders = ",".join("?" * len(campaign_ids))
            query = f"SELECT * FROM plan_items WHERE campaign_id IN ({placeholders})"
            params: list = list(campaign_ids)
        elif campaign_id:
            query = "SELECT * FROM plan_items WHERE campaign_id = ?"
            params = [campaign_id]
        else:
            query = "SELECT * FROM plan_items WHERE 1=1"
            params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)

        query += " ORDER BY phase_order, created_at"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_plan_item(row) for row in rows]

    def update_plan_item(
        self,
        item_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[PlanItemStatus] = None,
        outcome: Optional[str] = None,
        spec: Optional[Dict] = None,
        planned_session_id: Optional[str] = None,
        session_id: Optional[str] = None,
        phase_order: Optional[int] = None,
    ):
        """Update a plan item. Only non-None values are applied."""
        now = self._now()
        updates = []
        values = []
        for col, val in [
            ("title", title),
            ("description", description),
            ("outcome", outcome),
            ("planned_session_id", planned_session_id),
            ("session_id", session_id),
        ]:
            if val is not None:
                updates.append(f"{col} = ?")
                values.append(val)
        if status is not None:
            updates.append("status = ?")
            values.append(status.value)
        if spec is not None:
            updates.append("spec = ?")
            values.append(json.dumps(spec))
        if phase_order is not None:
            updates.append("phase_order = ?")
            values.append(phase_order)
        if not updates:
            return
        updates.append("updated_at = ?")
        values.append(now)
        values.append(item_id)
        with self._tx():
            self._conn.execute(
                f"UPDATE plan_items SET {', '.join(updates)} WHERE id = ?",
                values,
            )

    def complete_plan_item(self, item_id: str, outcome: str):
        """Mark a plan item as completed with an outcome description."""
        self.update_plan_item(
            item_id, status=PlanItemStatus.COMPLETED, outcome=outcome,
        )

    def skip_plan_item(self, item_id: str, reason: Optional[str] = None):
        """Mark a plan item as skipped."""
        self.update_plan_item(
            item_id,
            status=PlanItemStatus.SKIPPED,
            outcome=reason or "Skipped",
        )

    def add_plan_item_dependency(self, item_id: str, depends_on_id: str):
        """Add a dependency between plan items."""
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO plan_item_dependencies "
                "(item_id, depends_on_id) VALUES (?, ?)",
                (item_id, depends_on_id),
            )

    def remove_plan_item_dependency(self, item_id: str, depends_on_id: str):
        """Remove a dependency between plan items."""
        with self._tx():
            self._conn.execute(
                "DELETE FROM plan_item_dependencies "
                "WHERE item_id = ? AND depends_on_id = ?",
                (item_id, depends_on_id),
            )

    def get_plan_item_dependencies(self, item_id: str) -> List[str]:
        """Get IDs of items this item depends on."""
        rows = self._conn.execute(
            "SELECT depends_on_id FROM plan_item_dependencies WHERE item_id = ?",
            (item_id,),
        ).fetchall()
        return [row["depends_on_id"] for row in rows]

    def get_plan_item_dependents(self, item_id: str) -> List[str]:
        """Get IDs of items that depend on this item."""
        rows = self._conn.execute(
            "SELECT item_id FROM plan_item_dependencies WHERE depends_on_id = ?",
            (item_id,),
        ).fetchall()
        return [row["item_id"] for row in rows]

    def get_unblocked_plan_items(self, campaign_id: str) -> List[PlanItem]:
        """
        Get plan items that are planned and have all dependencies completed.
        These are the items that can be started next.
        """
        items = self.get_plan_items(
            campaign_id=campaign_id, status="planned", include_children=True,
        )
        unblocked = []
        for item in items:
            if not item.depends_on:
                unblocked.append(item)
                continue
            # Check if all dependencies are completed or skipped
            all_resolved = True
            for dep_id in item.depends_on:
                dep = self.get_plan_item(dep_id)
                if dep and dep.status not in (
                    PlanItemStatus.COMPLETED, PlanItemStatus.SKIPPED,
                ):
                    all_resolved = False
                    break
            if all_resolved:
                unblocked.append(item)
        return unblocked

    def get_plan_status(self, campaign_id: str) -> Dict[str, Any]:
        """
        Get a summary of plan progress for a campaign and its children.

        Returns
        -------
        dict
            {
                "total": int,
                "completed": int,
                "in_progress": int,
                "planned": int,
                "skipped": int,
                "blocked": int,
                "by_type": {"imaging": {"total": N, "completed": N, ...}, ...},
                "next_actions": [PlanItem, ...],
                "pending_decisions": [PlanItem, ...],
            }
        """
        items = self.get_plan_items(
            campaign_id=campaign_id, include_children=True,
        )
        result = {
            "total": len(items),
            "completed": 0,
            "in_progress": 0,
            "planned": 0,
            "skipped": 0,
            "blocked": 0,
            "by_type": {},
            "next_actions": [],
            "pending_decisions": [],
        }
        for item in items:
            status_key = item.status.value
            if status_key in result:
                result[status_key] += 1

            # By type
            type_key = item.type.value
            if type_key not in result["by_type"]:
                result["by_type"][type_key] = {"total": 0, "completed": 0}
            result["by_type"][type_key]["total"] += 1
            if item.status == PlanItemStatus.COMPLETED:
                result["by_type"][type_key]["completed"] += 1

            # Pending decisions
            if (
                item.type == PlanItemType.DECISION_POINT
                and item.status == PlanItemStatus.PLANNED
            ):
                result["pending_decisions"].append(item)

        # Next actions = unblocked items
        result["next_actions"] = self.get_unblocked_plan_items(campaign_id)

        return result

    def resolve_imaging_spec(self, item: PlanItem) -> Optional[ImagingSpec]:
        """
        Resolve the full ImagingSpec for an item, following inheritance.

        If the item inherits from another, the parent's spec is loaded
        first, then local fields override.
        """
        import dataclasses

        if item.type != PlanItemType.IMAGING:
            return None

        # Base case: no inheritance
        if not item.inherit_from:
            return item.imaging_spec

        # Load parent spec (recursive)
        parent = self.get_plan_item(item.inherit_from)
        if not parent:
            return item.imaging_spec

        parent_spec = self.resolve_imaging_spec(parent)
        if not parent_spec:
            return item.imaging_spec

        # Merge: local fields override parent fields
        if not item.imaging_spec:
            return parent_spec

        merged = dataclasses.replace(parent_spec)
        for f in dataclasses.fields(ImagingSpec):
            local_val = getattr(item.imaging_spec, f.name)
            if local_val is not None:
                setattr(merged, f.name, local_val)
        return merged

    def _get_campaign_tree_ids(self, campaign_id: str) -> List[str]:
        """Get all campaign IDs in a tree (recursive)."""
        ids = [campaign_id]
        children = self._conn.execute(
            "SELECT id FROM campaigns WHERE parent_id = ?",
            (campaign_id,),
        ).fetchall()
        for child in children:
            ids.extend(self._get_campaign_tree_ids(child["id"]))
        return ids

    def _row_to_plan_item(self, row: sqlite3.Row) -> PlanItem:
        d = dict(row)
        item_id = d["id"]

        # Load dependencies
        deps = self.get_plan_item_dependencies(item_id)

        # Parse spec into ImagingSpec or BenchSpec based on type
        spec_data = json.loads(d["spec"]) if d.get("spec") else None
        item_type = PlanItemType(d["type"])
        imaging_spec = None
        bench_spec = None

        if spec_data:
            if item_type == PlanItemType.IMAGING:
                import dataclasses as _dc
                valid = {f.name for f in _dc.fields(ImagingSpec)}
                imaging_spec = ImagingSpec(**{
                    k: v for k, v in spec_data.items() if k in valid
                })
            else:
                import dataclasses as _dc
                valid = {f.name for f in _dc.fields(BenchSpec)}
                bench_spec = BenchSpec(**{
                    k: v for k, v in spec_data.items() if k in valid
                })

        return PlanItem(
            id=item_id,
            campaign_id=d["campaign_id"],
            type=item_type,
            title=d["title"],
            description=d.get("description"),
            status=PlanItemStatus(d.get("status", "planned")),
            depends_on=deps,
            outcome=d.get("outcome"),
            imaging_spec=imaging_spec,
            bench_spec=bench_spec,
            planned_session_id=d.get("planned_session_id"),
            session_id=d.get("session_id"),
            inherit_from=d.get("inherit_from"),
            phase_order=d.get("phase_order", 0),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
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
