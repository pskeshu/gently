"""
ContextStore — SQLite-backed storage for the agent's mind.

Separate from GentlyStore. This holds understanding, not raw data.

The store is composed from three domain mixins:
  - IntentionsMixin: campaigns, projects, sessions, planned sessions
  - PlansMixin: plan items, templates, snapshots, dependencies
  - UnderstandingMixin: observations, expectations, watchpoints,
    questions, learnings, embryo understanding, agent state, batch updates
"""

import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ._intentions import IntentionsMixin
from ._ml_pipelines import MlPipelinesMixin
from ._plans import PlansMixin
from ._understanding import UnderstandingMixin
from .model import (
    Context,
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
    estimated_days INTEGER,
    phase_order INTEGER DEFAULT 0,
    "references" TEXT,
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

-- Plan templates
CREATE TABLE IF NOT EXISTS plan_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    template_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Plan snapshots (version history)
CREATE TABLE IF NOT EXISTS plan_snapshots (
    version_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    summary TEXT,
    label TEXT,
    parent_version_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
    FOREIGN KEY (parent_version_id) REFERENCES plan_snapshots(version_id)
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
CREATE INDEX IF NOT EXISTS idx_planned_session_campaigns_ps
    ON planned_session_campaigns(planned_session_id);
CREATE INDEX IF NOT EXISTS idx_planned_session_campaigns_c
    ON planned_session_campaigns(campaign_id);
CREATE INDEX IF NOT EXISTS idx_plan_items_campaign ON plan_items(campaign_id);
CREATE INDEX IF NOT EXISTS idx_plan_items_status ON plan_items(status);
CREATE INDEX IF NOT EXISTS idx_plan_items_type ON plan_items(type);
CREATE INDEX IF NOT EXISTS idx_plan_items_inherit ON plan_items(inherit_from);
CREATE INDEX IF NOT EXISTS idx_plan_item_deps_item ON plan_item_dependencies(item_id);
CREATE INDEX IF NOT EXISTS idx_plan_item_deps_dep ON plan_item_dependencies(depends_on_id);
CREATE INDEX IF NOT EXISTS idx_plan_snapshots_campaign ON plan_snapshots(campaign_id);
CREATE INDEX IF NOT EXISTS idx_plan_snapshots_version
    ON plan_snapshots(campaign_id, version_number);
"""


# ---------------------------------------------------------------------------
# ContextStore
# ---------------------------------------------------------------------------


class ContextStore(IntentionsMixin, PlansMixin, UnderstandingMixin, MlPipelinesMixin):
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
        self._migrate(conn)
        conn.commit()
        return conn

    def _migrate(self, conn: sqlite3.Connection):
        """Run lightweight migrations for columns/tables added after initial schema."""
        # Add "references" column to plan_items if missing (added after initial release)
        pi_cols = {row[1] for row in conn.execute("PRAGMA table_info(plan_items)").fetchall()}
        if "references" not in pi_cols:
            conn.execute('ALTER TABLE plan_items ADD COLUMN "references" TEXT')
            logger.info("Migration: added 'references' column to plan_items")

        # Mesh campaign coordination columns
        camp_cols = {row[1] for row in conn.execute("PRAGMA table_info(campaigns)").fetchall()}
        if "is_shared" not in camp_cols:
            conn.execute("ALTER TABLE campaigns ADD COLUMN is_shared INTEGER DEFAULT 0")
            logger.info("Migration: added 'is_shared' column to campaigns")

        if "claimed_by" not in pi_cols:
            conn.execute("ALTER TABLE plan_items ADD COLUMN claimed_by TEXT")
            logger.info("Migration: added 'claimed_by' column to plan_items")
        if "claimed_by_hostname" not in pi_cols:
            conn.execute("ALTER TABLE plan_items ADD COLUMN claimed_by_hostname TEXT")
            logger.info("Migration: added 'claimed_by_hostname' column to plan_items")

        if "estimated_days" not in pi_cols:
            conn.execute("ALTER TABLE plan_items ADD COLUMN estimated_days INTEGER")
            logger.info("Migration: added 'estimated_days' column to plan_items")

        # Campaign participants table for mesh coordination
        conn.execute("""
            CREATE TABLE IF NOT EXISTS campaign_participants (
                campaign_id TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                hostname TEXT,
                joined_at TEXT NOT NULL,
                PRIMARY KEY (campaign_id, instance_id),
                FOREIGN KEY (campaign_id) REFERENCES campaigns(id)
            )
        """)
        logger.debug("Migration: ensured campaign_participants table exists")

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

        Returns a dict of table_name -> rows_deleted.
        """
        tables = [
            "plan_item_dependencies",
            "plan_items",
            "plan_snapshots",
            "plan_templates",
            "planned_session_campaigns",
            "session_campaigns",
            "planned_sessions",
            "session_intents",
            "projects",
            "campaign_participants",
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
