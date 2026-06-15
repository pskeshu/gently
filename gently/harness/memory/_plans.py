"""
PlansMixin — Plan items, templates, snapshots, and dependency management.

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
    BenchSpec,
    ImagingSpec,
    PlanItem,
    PlanItemStatus,
    PlanItemType,
)

logger = logging.getLogger(__name__)


class PlansMixin(StoreProtocol):
    """Plan items, templates, snapshots, and dependency management."""

    # ==================================================================
    # Plan Items (experimental plan)
    # ==================================================================

    def create_plan_item(
        self,
        campaign_id: str,
        type: str,
        title: str,
        description: str | None = None,
        spec: dict | None = None,
        inherit_from: str | None = None,
        planned_session_id: str | None = None,
        phase_order: int = -1,
        depends_on: list[str] | None = None,
        item_id: str | None = None,
        references: list[dict] | None = None,
        estimated_days: int | None = None,
    ) -> str:
        """Create a plan item. Returns its ID.

        If phase_order is -1 (default), auto-assigns the next sequential
        number within the campaign (1-based).
        """
        pid = item_id or self._gen_id()
        now = self._now()

        if phase_order < 0:
            # Auto-assign: next number in this campaign
            row = self._conn.execute(
                "SELECT COALESCE(MAX(phase_order), 0) FROM plan_items WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            phase_order = row[0] + 1

        with self._tx():
            self._conn.execute(
                "INSERT INTO plan_items "
                "(id, campaign_id, type, title, description, spec, inherit_from, "
                " planned_session_id, estimated_days, phase_order,"
                ' "references", status, created_at, updated_at) '
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?)",
                (
                    pid,
                    campaign_id,
                    type,
                    title,
                    description,
                    json.dumps(spec) if spec else None,
                    inherit_from,
                    planned_session_id,
                    estimated_days,
                    phase_order,
                    json.dumps(references) if references else None,
                    now,
                    now,
                ),
            )
            if depends_on:
                for dep_id in depends_on:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO plan_item_dependencies "
                        "(item_id, depends_on_id) VALUES (?, ?)",
                        (pid, dep_id),
                    )
        logger.info(f"Created plan item {pid} [{type}] #{phase_order}: {title}")
        return pid

    def get_plan_item(self, item_id: str) -> PlanItem | None:
        """Get a specific plan item."""
        row = self._conn.execute("SELECT * FROM plan_items WHERE id = ?", (item_id,)).fetchone()
        return self._row_to_plan_item(row) if row else None

    def resolve_plan_item(
        self,
        ref: str,
        campaign_id: str | None = None,
    ) -> PlanItem | None:
        """Resolve a human-friendly plan item reference.

        Supported formats:
          - UUID or UUID prefix: "a1b2c3d4"
          - Task number within campaign: "3", "task 3", "#3"
          - Phase.task: "1.3", "2.1"
          - Campaign.phase.task: "nerve-ring.1.3", "ec11.2.1"
          - Natural language: "task 3 of phase 1", "phase 2 task 1"

        The campaign segment (in campaign.phase.task) can be a shorthand
        name or a UUID prefix.  When a bare task number or phase.task is
        given, campaign_id scopes the lookup (or falls back to the most
        recent root campaign).
        """
        import re

        ref = ref.strip().lower()

        # --- Direct ID match (UUID prefix) ---
        row = self._conn.execute("SELECT * FROM plan_items WHERE id = ?", (ref,)).fetchone()
        if row:
            return self._row_to_plan_item(row)

        # Also try UUID prefix match (e.g. first few chars)
        if len(ref) >= 4 and re.match(r"^[0-9a-f]+$", ref):
            row = self._conn.execute(
                "SELECT * FROM plan_items WHERE id LIKE ?", (ref + "%",)
            ).fetchone()
            if row:
                return self._row_to_plan_item(row)

        # --- Parse campaign.phase.task / phase.task / task ---
        phase_num = None
        task_num = None

        # "campaign.phase.task" — e.g. "nerve-ring.1.3" or "ec11.2.1"
        m = re.match(r"^([^.\s]+)\.(\d+)\.(\d+)$", ref)
        if m:
            campaign_label = m.group(1)
            phase_num, task_num = int(m.group(2)), int(m.group(3))
            # Resolve campaign label -> root_id
            resolved = self._resolve_campaign_label(campaign_label)
            if resolved:
                campaign_id = resolved

        # "1.3" or "2.1"
        if not task_num:
            m = re.match(r"^(\d+)\.(\d+)$", ref)
            if m:
                phase_num, task_num = int(m.group(1)), int(m.group(2))

        # "task 3 of phase 1" / "task 3 phase 1"
        if not task_num:
            m = re.search(r"task\s+(\d+)\s+(?:of\s+)?phase\s+(\d+)", ref)
            if m:
                task_num, phase_num = int(m.group(1)), int(m.group(2))

        # "phase 1 task 3"
        if not task_num:
            m = re.search(r"phase\s+(\d+)\s+task\s+(\d+)", ref)
            if m:
                phase_num, task_num = int(m.group(1)), int(m.group(2))

        # "task 3" / "#3" / just "3"
        if not task_num:
            m = re.match(r"^(?:task\s+|#)?(\d+)$", ref)
            if m:
                task_num = int(m.group(1))

        if not task_num:
            return None

        # --- Determine root campaign ---
        root_id = campaign_id
        if not root_id:
            # Fall back to first root campaign
            campaigns = self.get_root_campaigns()
            if campaigns:
                root_id = campaigns[0].id

        if not root_id:
            return None

        # --- Resolve phase -> campaign_id ---
        if phase_num is not None:
            phases = self.get_subcampaigns(root_id)
            if 1 <= phase_num <= len(phases):
                target_campaign = phases[phase_num - 1].id
            else:
                return None
        else:
            # No phase specified — check subcampaigns first, then root
            phases = self.get_subcampaigns(root_id)
            if phases:
                # Search across all phases for a global task number
                # Assign sequential numbers across phases: phase1 items, then phase2, etc.
                all_items = []
                for phase in phases:
                    items = self.get_plan_items(campaign_id=phase.id)
                    items.sort(key=lambda x: x.phase_order)
                    all_items.extend(items)
                if 1 <= task_num <= len(all_items):
                    return all_items[task_num - 1]
                return None
            else:
                target_campaign = root_id

        # --- Find task by phase_order within the target campaign ---
        items = self.get_plan_items(campaign_id=target_campaign)
        items.sort(key=lambda x: x.phase_order)
        if 1 <= task_num <= len(items):
            return items[task_num - 1]

        return None

    def _resolve_campaign_label(self, label: str) -> str | None:
        """Resolve a campaign shorthand or UUID prefix to an ID.

        Checks shorthand (case-insensitive), then UUID prefix, then
        description substring match.
        """
        label_lower = label.lower()

        # Shorthand match (case-insensitive)
        row = self._conn.execute(
            "SELECT id FROM campaigns WHERE LOWER(shorthand) = ? AND parent_id IS NULL",
            (label_lower,),
        ).fetchone()
        if row:
            return row["id"]

        # UUID prefix match
        if len(label) >= 4:
            row = self._conn.execute(
                "SELECT id FROM campaigns WHERE id LIKE ? AND parent_id IS NULL",
                (label_lower + "%",),
            ).fetchone()
            if row:
                return row["id"]

        # Description substring match (first word or hyphenated slug)
        row = self._conn.execute(
            "SELECT id FROM campaigns WHERE LOWER(description) LIKE ? AND parent_id IS NULL",
            ("%" + label_lower + "%",),
        ).fetchone()
        if row:
            return row["id"]

        return None

    def get_plan_items(
        self,
        campaign_id: str | None = None,
        status: str | None = None,
        type: str | None = None,
        include_children: bool = False,
    ) -> list[PlanItem]:
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
        title: str | None = None,
        description: str | None = None,
        status: PlanItemStatus | None = None,
        outcome: str | None = None,
        spec: dict | None = None,
        planned_session_id: str | None = None,
        session_id: str | None = None,
        phase_order: int | None = None,
        campaign_id: str | None = None,
        references: list[dict] | None = None,
        estimated_days: int | None = None,
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
            ("campaign_id", campaign_id),
            ("estimated_days", estimated_days),
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
        if references is not None:
            updates.append('"references" = ?')
            values.append(json.dumps(references))
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
            item_id,
            status=PlanItemStatus.COMPLETED,
            outcome=outcome,
        )

    def skip_plan_item(self, item_id: str, reason: str | None = None):
        """Mark a plan item as skipped."""
        self.update_plan_item(
            item_id,
            status=PlanItemStatus.SKIPPED,
            outcome=reason or "Skipped",
        )

    def delete_plan_item(self, item_id: str) -> bool:
        """Delete a plan item and all its dependency links.

        Returns True if the item existed and was deleted.
        """
        with self._tx():
            # Remove dependency links (both directions)
            self._conn.execute(
                "DELETE FROM plan_item_dependencies WHERE item_id = ? OR depends_on_id = ?",
                (item_id, item_id),
            )
            r = self._conn.execute(
                "DELETE FROM plan_items WHERE id = ?",
                (item_id,),
            )
            deleted = r.rowcount > 0
        if deleted:
            logger.info(f"Deleted plan item {item_id}")
        return deleted

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
                "DELETE FROM plan_item_dependencies WHERE item_id = ? AND depends_on_id = ?",
                (item_id, depends_on_id),
            )

    def get_plan_item_dependencies(self, item_id: str) -> list[str]:
        """Get IDs of items this item depends on."""
        rows = self._conn.execute(
            "SELECT depends_on_id FROM plan_item_dependencies WHERE item_id = ?",
            (item_id,),
        ).fetchall()
        return [row["depends_on_id"] for row in rows]

    def get_plan_item_dependents(self, item_id: str) -> list[str]:
        """Get IDs of items that depend on this item."""
        rows = self._conn.execute(
            "SELECT item_id FROM plan_item_dependencies WHERE depends_on_id = ?",
            (item_id,),
        ).fetchall()
        return [row["item_id"] for row in rows]

    def get_unblocked_plan_items(self, campaign_id: str) -> list[PlanItem]:
        """
        Get plan items that are planned and have all dependencies completed.
        These are the items that can be started next.
        """
        items = self.get_plan_items(
            campaign_id=campaign_id,
            status="planned",
            include_children=True,
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
                    PlanItemStatus.COMPLETED,
                    PlanItemStatus.SKIPPED,
                ):
                    all_resolved = False
                    break
            if all_resolved:
                unblocked.append(item)
        return unblocked

    def get_plan_status(self, campaign_id: str) -> dict[str, Any]:
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
            campaign_id=campaign_id,
            include_children=True,
        )
        result: dict[str, Any] = {
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
            if item.type == PlanItemType.DECISION_POINT and item.status == PlanItemStatus.PLANNED:
                result["pending_decisions"].append(item)

        # Next actions = unblocked items
        result["next_actions"] = self.get_unblocked_plan_items(campaign_id)

        return result

    def resolve_imaging_spec(self, item: PlanItem) -> ImagingSpec | None:
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

    # ==================================================================
    # Plan Templates
    # ==================================================================

    def save_plan_template(
        self,
        name: str,
        description: str | None,
        campaign_id: str,
    ) -> str:
        """
        Serialize a campaign tree (campaigns + items + specs + dependencies)
        into a reusable template. Returns the template ID.
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Build a portable representation
        template_data = self._serialize_campaign_tree(campaign_id)

        tid = self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO plan_templates "
                "(id, name, description, template_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tid, name, description, json.dumps(template_data), now, now),
            )
        logger.info(f"Saved plan template '{name}' ({tid})")
        return tid

    def _serialize_campaign_tree(self, campaign_id: str) -> dict:
        """Recursively serialize a campaign and its children/items."""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {}

        items = self.get_plan_items(campaign_id=campaign_id)
        items.sort(key=lambda x: x.phase_order)

        # Build item list with relative dependency indices
        all_item_ids = [it.id for it in items]
        serialized_items = []
        for item in items:
            item_data: dict[str, Any] = {
                "type": item.type.value,
                "title": item.title,
                "description": item.description,
                "phase_order": item.phase_order,
            }
            # Serialize spec
            if item.imaging_spec:
                import dataclasses as _dc

                spec_dict = {}
                for f in _dc.fields(item.imaging_spec):
                    val = getattr(item.imaging_spec, f.name)
                    if val is not None:
                        spec_dict[f.name] = val
                item_data["spec"] = spec_dict
            elif item.bench_spec:
                import dataclasses as _dc

                spec_dict = {}
                for f in _dc.fields(item.bench_spec):
                    val = getattr(item.bench_spec, f.name)
                    if val is not None:
                        spec_dict[f.name] = val
                item_data["spec"] = spec_dict

            # Dependencies as relative indices within this campaign's items
            if item.depends_on:
                dep_indices = []
                for dep_id in item.depends_on:
                    if dep_id in all_item_ids:
                        dep_indices.append(all_item_ids.index(dep_id))
                if dep_indices:
                    item_data["depends_on_indices"] = dep_indices

            if item.references:
                item_data["references"] = item.references

            serialized_items.append(item_data)

        # Recurse into sub-campaigns
        children = self.get_subcampaigns(campaign_id)
        serialized_children = []
        for child in children:
            serialized_children.append(self._serialize_campaign_tree(child.id))

        return {
            "description": campaign.description,
            "shorthand": campaign.shorthand,
            "target": campaign.target,
            "items": serialized_items,
            "children": serialized_children,
        }

    def list_plan_templates(self) -> list[dict]:
        """List all plan templates (id, name, description, dates)."""
        rows = self._conn.execute(
            "SELECT id, name, description, created_at, updated_at "
            "FROM plan_templates ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_plan_template(self, id_or_name: str) -> dict | None:
        """Get a plan template by ID or name."""
        row = self._conn.execute(
            "SELECT * FROM plan_templates WHERE id = ? OR name = ?",
            (id_or_name, id_or_name),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["template_json"] = json.loads(d["template_json"])
        return d

    def apply_plan_template(
        self,
        template_id: str,
        overrides: dict | None = None,
    ) -> str:
        """
        Instantiate a template into a new campaign with plan items.
        Overrides (e.g. strain, temperature_c) are applied to all imaging specs.
        Returns the new root campaign ID.
        """
        tmpl = self.get_plan_template(template_id)
        if not tmpl:
            raise ValueError(f"Template '{template_id}' not found")

        data = tmpl["template_json"]
        overrides = overrides or {}
        return self._instantiate_template_tree(data, parent_id=None, overrides=overrides)

    def _instantiate_template_tree(
        self,
        data: dict,
        parent_id: str | None,
        overrides: dict,
    ) -> str:
        """Recursively create campaigns and items from template data."""
        cid = self.create_campaign(
            description=data.get("description", "Untitled"),
            shorthand=data.get("shorthand"),
            target=data.get("target"),
            parent_id=parent_id,
        )

        # Create items, track new IDs for dependency wiring
        items_data = data.get("items", [])
        new_item_ids: list[str] = []

        for item_data in items_data:
            spec = item_data.get("spec")

            # Apply overrides to imaging specs
            if spec and item_data.get("type") == "imaging" and overrides:
                spec = dict(spec)  # copy
                for k, v in overrides.items():
                    if k in spec or k in (
                        "strain",
                        "genotype",
                        "reporter",
                        "temperature_c",
                        "num_slices",
                        "exposure_ms",
                        "interval_s",
                        "num_embryos",
                        "stop_condition",
                    ):
                        spec[k] = v

            item_id = self.create_plan_item(
                campaign_id=cid,
                type=item_data.get("type", "imaging"),
                title=item_data.get("title", "Untitled"),
                description=item_data.get("description"),
                spec=spec,
                phase_order=item_data.get("phase_order", -1),
                references=item_data.get("references"),
            )
            new_item_ids.append(item_id)

        # Wire up dependencies using relative indices
        for idx, item_data in enumerate(items_data):
            dep_indices = item_data.get("depends_on_indices", [])
            for dep_idx in dep_indices:
                if 0 <= dep_idx < len(new_item_ids):
                    self.add_plan_item_dependency(
                        new_item_ids[idx],
                        new_item_ids[dep_idx],
                    )

        # Recurse into children
        for child_data in data.get("children", []):
            self._instantiate_template_tree(child_data, parent_id=cid, overrides=overrides)

        return cid

    def delete_plan_template(self, template_id: str) -> bool:
        """Delete a plan template. Returns True if found and deleted."""
        with self._tx():
            r = self._conn.execute(
                "DELETE FROM plan_templates WHERE id = ? OR name = ?",
                (template_id, template_id),
            )
            return r.rowcount > 0

    # ==================================================================
    # Plan Snapshots (version history)
    # ==================================================================

    def create_plan_snapshot(
        self,
        campaign_id: str,
        label: str | None = None,
        summary: str | None = None,
    ) -> str:
        """Create a snapshot of the current plan state.

        Parameters
        ----------
        campaign_id : str
            Root campaign to snapshot.
        label : str, optional
            Human-readable label (e.g. "before PI feedback").
        summary : str, optional
            Text summary. Auto-generated if not provided.

        Returns
        -------
        str
            The version_id of the new snapshot.
        """
        snapshot_data = self._serialize_campaign_tree(campaign_id)
        if not summary:
            summary = self._generate_snapshot_summary(campaign_id)

        # Auto-increment version number for this campaign
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version_number), 0) FROM plan_snapshots WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        version_number = row[0] + 1

        # Find parent version (previous latest snapshot)
        parent_row = self._conn.execute(
            "SELECT version_id FROM plan_snapshots "
            "WHERE campaign_id = ? ORDER BY version_number DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        parent_version_id = parent_row["version_id"] if parent_row else None

        version_id = self._gen_id()
        now = self._now()
        with self._tx():
            self._conn.execute(
                "INSERT INTO plan_snapshots "
                "(version_id, campaign_id, version_number, snapshot_json, "
                " summary, label, parent_version_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    version_id,
                    campaign_id,
                    version_number,
                    json.dumps(snapshot_data),
                    summary,
                    label,
                    parent_version_id,
                    now,
                ),
            )
        logger.info(
            f"Created plan snapshot v{version_number} ({version_id}) for campaign {campaign_id}"
        )
        return version_id

    def _generate_snapshot_summary(self, campaign_id: str) -> str:
        """Generate a brief text summary of the current plan state."""
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return "Unknown campaign"

        phases = self.get_subcampaigns(campaign_id)
        items = self.get_plan_items(campaign_id=campaign_id, include_children=True)

        # Count items by status
        status_counts: dict[str, int] = {}
        for item in items:
            key = item.status.value
            status_counts[key] = status_counts.get(key, 0) + 1

        parts = [campaign.description]
        if phases:
            phase_names = [p.description for p in phases]
            parts.append(f"{len(phases)} phases: {', '.join(phase_names)}")
        parts.append(f"{len(items)} items total")
        for status_name, count in sorted(status_counts.items()):
            parts.append(f"  {status_name}: {count}")

        return "\n".join(parts)

    def list_plan_snapshots(
        self,
        campaign_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """List snapshots for a campaign (metadata only, no blob).

        Returns list of dicts with version_id, version_number, label,
        summary, parent_version_id, created_at.
        """
        rows = self._conn.execute(
            "SELECT version_id, campaign_id, version_number, summary, "
            "       label, parent_version_id, created_at "
            "FROM plan_snapshots "
            "WHERE campaign_id = ? ORDER BY version_number DESC LIMIT ?",
            (campaign_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_plan_snapshot(self, version_id: str) -> dict | None:
        """Get a full snapshot including the parsed JSON blob."""
        row = self._conn.execute(
            "SELECT * FROM plan_snapshots WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["snapshot_json"] = json.loads(d["snapshot_json"])
        return d

    def restore_plan_snapshot(self, version_id: str) -> str:
        """Restore a plan from a snapshot.

        1. Auto-snapshots the current state before restoring.
        2. Deletes the current campaign tree.
        3. Re-creates it from the snapshot JSON.

        Returns the new campaign_id (fresh IDs are generated).
        """
        snapshot = self.get_plan_snapshot(version_id)
        if not snapshot:
            raise ValueError(f"Snapshot {version_id} not found")

        campaign_id = snapshot["campaign_id"]
        version_number = snapshot["version_number"]

        # Auto-snapshot current state before restoring
        try:
            self.create_plan_snapshot(
                campaign_id,
                label=f"auto: before restore to v{version_number}",
            )
        except Exception:
            pass  # Don't block restore if snapshot fails

        # Get the original campaign's parent_id so the restored tree
        # is placed in the same position in the hierarchy
        campaign = self.get_campaign(campaign_id)
        parent_id = campaign.parent_id if campaign else None

        # Delete the current campaign tree
        self.delete_campaign(campaign_id, cascade=True)

        # Re-create from snapshot
        new_campaign_id = self._instantiate_template_tree(
            snapshot["snapshot_json"],
            parent_id=parent_id,
            overrides={},
        )

        logger.info(
            f"Restored plan snapshot v{version_number} ({version_id}) "
            f"-> new campaign {new_campaign_id}"
        )
        return new_campaign_id

    def _get_campaign_tree_ids(self, campaign_id: str) -> list[str]:
        """Get all campaign IDs in a tree (recursive)."""
        ids = [campaign_id]
        children = self._conn.execute(
            "SELECT id FROM campaigns WHERE parent_id = ?",
            (campaign_id,),
        ).fetchall()
        for child in children:
            ids.extend(self._get_campaign_tree_ids(child["id"]))
        return ids

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

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
                imaging_spec = ImagingSpec(**{k: v for k, v in spec_data.items() if k in valid})
            else:
                import dataclasses as _dc

                valid = {f.name for f in _dc.fields(BenchSpec)}
                bench_spec = BenchSpec(**{k: v for k, v in spec_data.items() if k in valid})

        references = json.loads(d["references"]) if d.get("references") else []

        return PlanItem(
            id=item_id,
            campaign_id=d["campaign_id"],
            type=item_type,
            title=d["title"],
            description=d.get("description"),
            status=PlanItemStatus(d.get("status", "planned")),
            depends_on=deps,
            outcome=d.get("outcome"),
            claimed_by=d.get("claimed_by"),
            claimed_by_hostname=d.get("claimed_by_hostname"),
            references=references,
            imaging_spec=imaging_spec,
            bench_spec=bench_spec,
            planned_session_id=d.get("planned_session_id"),
            session_id=d.get("session_id"),
            inherit_from=d.get("inherit_from"),
            estimated_days=d.get("estimated_days"),
            phase_order=d.get("phase_order", 0),
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
        )
