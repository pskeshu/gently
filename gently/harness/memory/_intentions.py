"""
IntentionsMixin — Campaign, project, session, and planned-session management.

Mixed into ContextStore; relies on self._conn, self._tx(), self._now(),
self._gen_id() provided by the host class.
"""

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

from ._protocols import StoreProtocol
from .model import (
    Campaign,
    Intentions,
    PlannedSession,
    PlannedSessionStatus,
    Project,
    SessionIntent,
    Status,
)

logger = logging.getLogger(__name__)


class IntentionsMixin(StoreProtocol):
    """Campaign management, projects, session intents, and planned sessions."""

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_intentions(self) -> Intentions:
        return Intentions(
            campaigns=self.get_active_campaigns(),
            projects=self.get_active_projects(),
            planned_sessions=self.get_upcoming_sessions(limit=5),
            current_focus=self.get_state("current_focus"),
            session_intent=self.get_current_session_intent(),
        )

    # ==================================================================
    # Campaigns
    # ==================================================================

    def create_campaign(
        self,
        description: str,
        shorthand: str | None = None,
        summary: str | None = None,
        target: str | None = None,
        parent_id: str | None = None,
        campaign_id: str | None = None,
    ) -> str:
        """Create a new campaign. Returns campaign ID."""
        cid = campaign_id or self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO campaigns "
                "(id, description, shorthand, summary, target, parent_id,"
                " status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                (cid, description, shorthand, summary, target, parent_id, now, now),
            )
        label = shorthand or description[:50]
        logger.info(f"Created campaign {cid} [{label}]")
        return cid

    def get_active_campaigns(self) -> list[Campaign]:
        """Get all active campaigns."""
        rows = self._conn.execute(
            "SELECT * FROM campaigns WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def count_non_active_campaigns(self) -> int:
        """Count campaigns whose status is not 'active'."""
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM campaigns WHERE status != 'active'"
        ).fetchone()
        return row["cnt"] if row else 0

    def count_session_intents(self) -> int:
        """Count total session intent records."""
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM session_intents").fetchone()
        return row["cnt"] if row else 0

    def get_all_campaigns(self, limit: int = 50) -> list[Campaign]:
        """Get all campaigns regardless of status, ordered by created_at descending."""
        rows = self._conn.execute(
            "SELECT * FROM campaigns ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def get_recent_session_intents(self, limit: int = 50) -> list["SessionIntent"]:
        """Get recent session intents, ordered by created_at descending."""
        rows = self._conn.execute(
            "SELECT * FROM session_intents ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            sid = d["session_id"]
            campaign_ids = self.get_campaign_ids_for_session(sid)
            results.append(
                SessionIntent(
                    session_id=sid,
                    planned_intent=d.get("planned_intent"),
                    actual_summary=d.get("actual_summary"),
                    campaign_ids=campaign_ids,
                    created_at=datetime.fromisoformat(d["created_at"]),
                    completed_at=datetime.fromisoformat(d["completed_at"])
                    if d.get("completed_at")
                    else None,
                )
            )
        return results

    def get_campaign(self, campaign_id: str) -> Campaign | None:
        """Get a specific campaign by exact ID."""
        row = self._conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        return self._row_to_campaign(row) if row else None

    def resolve_campaign(self, ref: str) -> Campaign | None:
        """Resolve a campaign by UUID, shorthand, UUID prefix, or description.

        Tries exact ID first, then falls back to _resolve_campaign_label
        for fuzzy matching. This lets tools accept "nrf-2026" or
        "nerve ring" instead of requiring the raw UUID.
        """
        # Exact ID
        campaign = self.get_campaign(ref)
        if campaign:
            return campaign
        # Fuzzy: shorthand, UUID prefix, description substring
        resolved_id = self._resolve_campaign_label(ref)
        if resolved_id:
            return self.get_campaign(resolved_id)
        return None

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

    def delete_campaign(self, campaign_id: str, cascade: bool = True) -> dict[str, int]:
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
                "DELETE FROM plan_items WHERE campaign_id = ?",
                (cid,),
            )
            counts["plan_items"] += r.rowcount

            # Delete campaign participants
            self._conn.execute(
                "DELETE FROM campaign_participants WHERE campaign_id = ?",
                (cid,),
            )

            # Delete campaign
            r = self._conn.execute(
                "DELETE FROM campaigns WHERE id = ?",
                (cid,),
            )
            counts["campaigns"] += r.rowcount

        with self._tx():
            _delete_recursive(campaign_id)

        return counts

    def get_subcampaigns(self, campaign_id: str) -> list[Campaign]:
        """Get direct children of a campaign."""
        rows = self._conn.execute(
            "SELECT * FROM campaigns WHERE parent_id = ? ORDER BY created_at",
            (campaign_id,),
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def get_nth_subcampaign(self, parent_id: str, n: int) -> Campaign | None:
        """Get the nth child campaign (1-indexed) of a parent, ordered by creation."""
        phases = self.get_subcampaigns(parent_id)
        if 1 <= n <= len(phases):
            return phases[n - 1]
        return None

    def get_campaign_tree(self, campaign_id: str) -> dict[str, Any]:
        """Get a campaign and all its descendants as a tree."""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {}
        children = self.get_subcampaigns(campaign_id)
        return {
            "campaign": campaign,
            "children": [self.get_campaign_tree(c.id) for c in children],
        }

    def get_root_campaigns(self, status: str | None = "active") -> list[Campaign]:
        """Get top-level campaigns (no parent). If status is None, returns all."""
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM campaigns WHERE parent_id IS NULL ORDER BY updated_at DESC LIMIT 50"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM campaigns WHERE parent_id IS NULL AND status = ? "
                "ORDER BY updated_at DESC LIMIT 50",
                (status,),
            ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def update_campaign(
        self,
        campaign_id: str,
        description: str | None = None,
        shorthand: str | None = None,
        summary: str | None = None,
        target: str | None = None,
        parent_id: str | None = None,
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

    # ------------------------------------------------------------------
    # Campaign sharing (mesh coordination)
    # ------------------------------------------------------------------

    def share_campaign(self, campaign_id: str):
        """Mark a campaign as shared on the mesh."""
        with self._tx():
            self._conn.execute(
                "UPDATE campaigns SET is_shared = 1, updated_at = ? WHERE id = ?",
                (self._now(), campaign_id),
            )

    def unshare_campaign(self, campaign_id: str):
        """Remove shared flag from a campaign."""
        with self._tx():
            self._conn.execute(
                "UPDATE campaigns SET is_shared = 0, updated_at = ? WHERE id = ?",
                (self._now(), campaign_id),
            )

    def get_shared_campaigns(self) -> list[Campaign]:
        """Get all campaigns marked as shared."""
        rows = self._conn.execute(
            "SELECT * FROM campaigns WHERE is_shared = 1 ORDER BY created_at",
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def add_campaign_participant(self, campaign_id: str, instance_id: str, hostname: str):
        """Register a mesh peer as a participant in a campaign."""
        with self._tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO campaign_participants "
                "(campaign_id, instance_id, hostname, joined_at) VALUES (?, ?, ?, ?)",
                (campaign_id, instance_id, hostname, self._now()),
            )

    def get_campaign_participants(self, campaign_id: str) -> list[dict]:
        """Get all participants for a campaign."""
        rows = self._conn.execute(
            "SELECT * FROM campaign_participants WHERE campaign_id = ? ORDER BY joined_at",
            (campaign_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def claim_plan_item(self, item_id: str, instance_id: str, hostname: str) -> bool:
        """Atomically claim a plan item for a mesh node.

        Returns True if the claim succeeded, False if already claimed by another node.
        Uses atomic UPDATE with WHERE to prevent races.
        """
        with self._tx():
            r = self._conn.execute(
                "UPDATE plan_items SET claimed_by = ?, claimed_by_hostname = ?, updated_at = ? "
                "WHERE id = ? AND (claimed_by IS NULL OR claimed_by = ?)",
                (instance_id, hostname, self._now(), item_id, instance_id),
            )
            return r.rowcount > 0

    def unclaim_plan_item(self, item_id: str):
        """Release a claim on a plan item."""
        with self._tx():
            self._conn.execute(
                "UPDATE plan_items SET claimed_by = NULL, claimed_by_hostname = NULL, "
                "updated_at = ? WHERE id = ?",
                (self._now(), item_id),
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
            is_shared=bool(d.get("is_shared", 0)),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )

    # ==================================================================
    # Projects
    # ==================================================================

    def create_project(
        self,
        description: str,
        campaign_id: str | None = None,
        project_id: str | None = None,
    ) -> str:
        """Create a new project. Returns project ID."""
        pid = project_id or self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO projects"
                " (id, description, campaign_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', ?, ?)",
                (pid, description, campaign_id, now, now),
            )
        logger.info(f"Created project {pid}: {description}")
        return pid

    def get_active_projects(self) -> list[Project]:
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
        planned_intent: str | None = None,
        campaign_ids: list[str] | None = None,
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

    def get_current_session_intent(self) -> SessionIntent | None:
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
            completed_at=datetime.fromisoformat(d["completed_at"])
            if d.get("completed_at")
            else None,
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
    # Session <-> Campaign (many-to-many)
    # ==================================================================

    def link_session_campaign(self, session_id: str, campaign_id: str):
        """Link a session to a campaign."""
        now = self._now()
        with self._tx():
            # Ensure session_intents row exists (FK target)
            self._conn.execute(
                "INSERT OR IGNORE INTO session_intents (session_id, created_at) VALUES (?, ?)",
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

    def get_campaign_ids_for_session(self, session_id: str) -> list[str]:
        """Get campaign IDs linked to a session."""
        rows = self._conn.execute(
            "SELECT campaign_id FROM session_campaigns WHERE session_id = ? ORDER BY linked_at",
            (session_id,),
        ).fetchall()
        return [row["campaign_id"] for row in rows]

    def get_campaigns_for_session(self, session_id: str) -> list[Campaign]:
        """Get campaigns linked to a session."""
        rows = self._conn.execute(
            "SELECT c.* FROM campaigns c "
            "JOIN session_campaigns sc ON c.id = sc.campaign_id "
            "WHERE sc.session_id = ? ORDER BY sc.linked_at",
            (session_id,),
        ).fetchall()
        return [self._row_to_campaign(row) for row in rows]

    def get_sessions_for_campaign(self, campaign_id: str) -> list[SessionIntent]:
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
            results.append(
                SessionIntent(
                    session_id=sid,
                    planned_intent=d.get("planned_intent"),
                    actual_summary=d.get("actual_summary"),
                    campaign_ids=cids,
                    created_at=datetime.fromisoformat(d["created_at"]),
                    completed_at=datetime.fromisoformat(d["completed_at"])
                    if d.get("completed_at")
                    else None,
                )
            )
        return results

    # ==================================================================
    # Planned Sessions (project calendar)
    # ==================================================================

    def create_planned_session(
        self,
        scheduled_date: str,
        title: str | None = None,
        notes: str | None = None,
        scheduled_time: str | None = None,
        estimated_duration_minutes: int | None = None,
        acquisition_params: dict | None = None,
        source_session_id: str | None = None,
        campaign_ids: list[str] | None = None,
        planned_session_id: str | None = None,
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
                    psid,
                    title,
                    notes,
                    scheduled_date,
                    scheduled_time,
                    estimated_duration_minutes,
                    json.dumps(acquisition_params) if acquisition_params else None,
                    source_session_id,
                    now,
                    now,
                ),
            )
        if campaign_ids:
            for cid in campaign_ids:
                self.link_planned_session_campaign(psid, cid)
        logger.info(
            f"Created planned session {psid} for {scheduled_date}: {title or notes or '(untitled)'}"
        )
        return psid

    def get_planned_session(self, planned_session_id: str) -> PlannedSession | None:
        """Get a specific planned session."""
        row = self._conn.execute(
            "SELECT * FROM planned_sessions WHERE id = ?",
            (planned_session_id,),
        ).fetchone()
        return self._row_to_planned_session(row) if row else None

    def get_planned_sessions(
        self,
        status: str | None = None,
        campaign_id: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[PlannedSession]:
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

    def get_upcoming_sessions(self, limit: int = 10) -> list[PlannedSession]:
        """Get upcoming planned sessions (today and future, status=planned)."""
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self._conn.execute(
            "SELECT * FROM planned_sessions "
            "WHERE status = 'planned' AND scheduled_date >= ? "
            "ORDER BY scheduled_date, scheduled_time LIMIT ?",
            (today, limit),
        ).fetchall()
        return [self._row_to_planned_session(row) for row in rows]

    def get_todays_sessions(self) -> list[PlannedSession]:
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
        title: str | None = None,
        notes: str | None = None,
        scheduled_date: str | None = None,
        scheduled_time: str | None = None,
        estimated_duration_minutes: int | None = None,
        acquisition_params: dict | None = None,
        source_session_id: str | None = None,
        status: PlannedSessionStatus | None = None,
        session_id: str | None = None,
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

    def get_campaign_ids_for_planned_session(self, planned_session_id: str) -> list[str]:
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
            acquisition_params=json.loads(d["acquisition_params"])
            if d.get("acquisition_params")
            else None,
            source_session_id=d.get("source_session_id"),
            status=PlannedSessionStatus(d.get("status", "planned")),
            session_id=d.get("session_id"),
            campaign_ids=campaign_ids,
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )
