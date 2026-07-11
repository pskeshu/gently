"""
FileContextStore -- file-based storage for the agent's mind.

Drop-in replacement for ContextStore (SQLite-backed).  All data lives
under a single *agent_dir* directory as YAML files, organised by domain:

    agent/
      state.yaml                      # key-value agent state
      campaigns/{id}_{slug}/
        campaign.yaml                 # definition, status, session refs
        plan/
          current.yaml                # plan items list
          history/{YYYYMMDD_HHMM}.yaml
        templates/{name}.yaml
      projects/{id}_{slug}.yaml
      session_intents/{session_id}.yaml
      planned_sessions/{id}.yaml
      learnings/{id}_{slug}.yaml
      observations/{id}_{slug}.yaml
      active/
        expectations.yaml
        watchpoints.yaml
        questions.yaml
      embryo_understanding/{uid}.yaml
      ml/
        pipelines/{id}.yaml
        runs/{id}.yaml
        assessments/{id}.yaml
"""

import copy
import dataclasses
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .model import (
    Attention,
    BenchSpec,
    Campaign,
    Confidence,
    Context,
    ContextUpdates,
    EmbryoUnderstanding,
    Expectation,
    ExpectationStatus,
    ImagingSpec,
    Intentions,
    Learning,
    Observation,
    PlanItem,
    PlanItemStatus,
    PlanItemType,
    PlannedSession,
    PlannedSessionStatus,
    Project,
    Question,
    QuestionStatus,
    SessionIntent,
    Significance,
    Status,
    Understanding,
    Watchpoint,
    WatchpointStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML helpers -- keep datetimes as ISO strings in files
# ---------------------------------------------------------------------------


class _ISODumper(yaml.SafeDumper):
    """Custom dumper that serialises datetime objects as ISO strings."""

    pass


def _datetime_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.isoformat())


_ISODumper.add_representer(datetime, _datetime_representer)


# ---------------------------------------------------------------------------
# FileContextStore
# ---------------------------------------------------------------------------


class FileContextStore:
    """
    File-based storage for the agent's context.

    API-compatible with ContextStore (SQLite-backed) -- every public
    method has the same signature and semantics.
    """

    def __init__(self, agent_dir: Path):
        self.agent_dir = Path(agent_dir)
        self._ensure_dirs()
        # YAML parse cache: str(path) -> ((mtime, size), parsed). Collapses the
        # O(N^2) re-parsing in campaign-tree builds; auto-invalidated by file
        # mtime/size changes and explicitly on _write_yaml. Set BEFORE the index
        # rebuild below, which reads YAML through the cache.
        self._yaml_cache: dict[str, tuple] = {}
        # In-memory index: campaign_id -> folder Path
        self._campaign_index: dict[str, Path] = {}
        self._rebuild_campaign_index()

    # ------------------------------------------------------------------
    # Directory bootstrap
    # ------------------------------------------------------------------

    def _ensure_dirs(self):
        """Create the directory skeleton under agent_dir."""
        for subdir in (
            "campaigns",
            "projects",
            "session_intents",
            "planned_sessions",
            "learnings",
            "observations",
            "active",
            "embryo_understanding",
            "ml/pipelines",
            "ml/runs",
            "ml/assessments",
        ):
            (self.agent_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Campaign index
    # ------------------------------------------------------------------

    def _rebuild_campaign_index(self):
        """Scan campaigns/ and build {id: folder_path} mapping."""
        self._campaign_index.clear()
        campaigns_dir = self.agent_dir / "campaigns"
        if not campaigns_dir.exists():
            return
        for entry in campaigns_dir.iterdir():
            if entry.is_dir():
                campaign_file = entry / "campaign.yaml"
                if campaign_file.exists():
                    data = self._read_yaml(campaign_file)
                    if data and "id" in data:
                        self._campaign_index[data["id"]] = entry

    def _campaign_folder(self, campaign_id: str) -> Path | None:
        """Return the folder for a campaign, or None."""
        return self._campaign_index.get(campaign_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _gen_id(self) -> str:
        return str(uuid.uuid4())[:8]

    @staticmethod
    def _slugify(text: str) -> str:
        """Lowercase, hyphens, max 30 chars."""
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return slug[:30]

    def _write_yaml(self, path: Path, data):
        """Atomic write: write to .tmp then rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.dump(
                data,
                fh,
                Dumper=_ISODumper,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        # Atomic rename (on Windows this replaces the target).
        os.replace(str(tmp), str(path))
        # Invalidate the parse cache so the next read reloads (new mtime anyway).
        self._yaml_cache.pop(str(path), None)

    def _read_yaml(self, path: Path):
        """Read a YAML file, parse-cached by (mtime, size). Returns None if
        missing or empty. The cached object is never handed out directly — every
        return is a deepcopy — so callers may freely mutate the result without
        corrupting the cache."""
        try:
            st = path.stat()
        except OSError:
            return None
        key = str(path)
        sig = (st.st_mtime, st.st_size)
        cached = self._yaml_cache.get(key)
        if cached is not None and cached[0] == sig:
            return copy.deepcopy(cached[1])
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except Exception:
            logger.warning(f"Failed to read {path}", exc_info=True)
            return None
        self._yaml_cache[key] = (sig, data)
        return copy.deepcopy(data)

    def _append_jsonl(self, path: Path, record: dict):
        """Append one JSON line to a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """No-op for file store."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return f"FileContextStore(agent_dir={self.agent_dir})"

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self) -> dict:
        """Delete all data files; return counts of deleted items by category."""
        counts: dict[str, int] = {}

        def _count_and_remove(subdir: str, label: str):
            d = self.agent_dir / subdir
            if not d.exists():
                return
            n = 0
            for entry in list(d.iterdir()):
                if entry.is_file() and entry.suffix in (".yaml", ".yml"):
                    entry.unlink()
                    n += 1
                elif entry.is_dir():
                    shutil.rmtree(entry)
                    n += 1
            if n:
                counts[label] = n

        _count_and_remove("campaigns", "campaigns")
        _count_and_remove("projects", "projects")
        _count_and_remove("session_intents", "session_intents")
        _count_and_remove("planned_sessions", "planned_sessions")
        _count_and_remove("learnings", "learnings")
        _count_and_remove("observations", "observations")
        _count_and_remove("embryo_understanding", "embryo_understanding")
        _count_and_remove("ml/pipelines", "ml_pipelines")
        _count_and_remove("ml/runs", "ml_training_runs")
        _count_and_remove("ml/assessments", "ml_data_assessments")

        # Active lists
        for name in ("expectations", "watchpoints", "questions"):
            p = self.agent_dir / "active" / f"{name}.yaml"
            if p.exists():
                data = self._read_yaml(p) or []
                if data:
                    counts[name] = len(data)
                p.unlink()

        # Agent state
        state_path = self.agent_dir / "state.yaml"
        if state_path.exists():
            data = self._read_yaml(state_path) or {}
            if data:
                counts["agent_state"] = len(data)
            state_path.unlink()

        self._campaign_index.clear()
        total = sum(counts.values())
        logger.info(
            f"File context store reset -- {total} items cleared from {len(counts)} categories"
        )
        return counts

    # ==================================================================
    # Load Active Context
    # ==================================================================

    def load_active(self) -> Context:
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

    def _load_embryo_states(self) -> dict[str, EmbryoUnderstanding]:
        result: dict[str, EmbryoUnderstanding] = {}
        eu_dir = self.agent_dir / "embryo_understanding"
        if not eu_dir.exists():
            return result
        for f in eu_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if not data:
                    continue
                if not data.get("is_tracked", True):
                    continue
                eid = data["embryo_id"]
                result[eid] = self._dict_to_embryo_understanding(data)
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
        shorthand: str | None = None,
        summary: str | None = None,
        target: str | None = None,
        parent_id: str | None = None,
        campaign_id: str | None = None,
    ) -> str:
        cid = campaign_id or self._gen_id()
        now = self._now()
        slug = self._slugify(shorthand or description)
        folder = self.agent_dir / "campaigns" / f"{cid}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plan").mkdir(exist_ok=True)
        (folder / "plan" / "history").mkdir(exist_ok=True)
        (folder / "templates").mkdir(exist_ok=True)

        data: dict[str, Any] = {
            "id": cid,
            "description": description,
            "shorthand": shorthand,
            "summary": summary,
            "target": target,
            "progress": None,
            "parent_id": parent_id,
            "status": "active",
            "is_shared": False,
            "participants": [],
            "session_ids": [],
            "created_at": now,
            "updated_at": now,
        }
        self._write_yaml(folder / "campaign.yaml", data)
        self._write_yaml(folder / "plan" / "current.yaml", [])
        self._campaign_index[cid] = folder

        label = shorthand or description[:50]
        logger.info(f"Created campaign {cid} [{label}]")
        return cid

    def get_campaign(self, campaign_id: str) -> Campaign | None:
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return None
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return None
        return self._dict_to_campaign(data)

    def get_active_campaigns(self) -> list[Campaign]:
        campaigns = []
        for _cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if data and data.get("status") == "active":
                campaigns.append(self._dict_to_campaign(data))
        campaigns.sort(key=lambda c: c.created_at, reverse=True)
        return campaigns

    def count_non_active_campaigns(self) -> int:
        """Count campaigns whose status is not 'active'."""
        count = 0
        for _cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if data and data.get("status") != "active":
                count += 1
        return count

    def count_session_intents(self) -> int:
        """Count total session intent files."""
        si_dir = self.agent_dir / "session_intents"
        if not si_dir.exists():
            return 0
        return sum(1 for f in si_dir.iterdir() if f.suffix in (".yaml", ".yml"))

    def get_all_campaigns(self, limit: int = 50) -> list[Campaign]:
        """Get all campaigns regardless of status, ordered by created_at descending."""
        campaigns = []
        for _cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if data:
                campaigns.append(self._dict_to_campaign(data))
        campaigns.sort(key=lambda c: c.created_at, reverse=True)
        return campaigns[:limit]

    def get_recent_session_intents(self, limit: int = 50) -> list[SessionIntent]:
        """Get recent session intents, ordered by created_at descending."""
        si_dir = self.agent_dir / "session_intents"
        if not si_dir.exists():
            return []
        intents = []
        for f in si_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data:
                    intents.append(self._dict_to_session_intent(data))
        intents.sort(key=lambda i: i.created_at, reverse=True)
        return intents[:limit]

    def resolve_campaign(self, ref: str) -> Campaign | None:
        campaign = self.get_campaign(ref)
        if campaign:
            return campaign
        resolved_id = self._resolve_campaign_label(ref)
        if resolved_id:
            return self.get_campaign(resolved_id)
        return None

    def _resolve_campaign_label(self, label: str) -> str | None:
        label_lower = label.lower()

        # Shorthand match (case-insensitive), root campaigns only
        for cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if not data:
                continue
            if data.get("parent_id") is not None:
                continue
            if data.get("shorthand") and data["shorthand"].lower() == label_lower:
                return cid

        # UUID prefix match, root campaigns only
        if len(label) >= 4:
            for cid, folder in self._campaign_index.items():
                data = self._read_yaml(folder / "campaign.yaml")
                if not data or data.get("parent_id") is not None:
                    continue
                if cid.lower().startswith(label_lower):
                    return cid

        # Description substring match, root campaigns only
        for cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if not data or data.get("parent_id") is not None:
                continue
            if label_lower in data.get("description", "").lower():
                return cid

        return None

    def update_campaign_progress(self, campaign_id: str, progress: str):
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return
        data["progress"] = progress
        data["updated_at"] = self._now()
        self._write_yaml(folder / "campaign.yaml", data)
        self._notify_plan_change(campaign_id)

    def update_campaign_status(self, campaign_id: str, status: Status):
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return
        data["status"] = status.value
        data["updated_at"] = self._now()
        self._write_yaml(folder / "campaign.yaml", data)

    def update_campaign(
        self,
        campaign_id: str,
        description: str | None = None,
        shorthand: str | None = None,
        summary: str | None = None,
        target: str | None = None,
        parent_id: str | None = None,
    ):
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return
        changed = False
        for key, val in [
            ("description", description),
            ("shorthand", shorthand),
            ("summary", summary),
            ("target", target),
            ("parent_id", parent_id),
        ]:
            if val is not None:
                data[key] = val
                changed = True
        if not changed:
            return
        data["updated_at"] = self._now()
        self._write_yaml(folder / "campaign.yaml", data)

    def delete_campaign(self, campaign_id: str, cascade: bool = True) -> dict[str, int]:
        counts: dict[str, int] = {"campaigns": 0, "plan_items": 0, "dependencies": 0}

        def _delete_recursive(cid: str):
            if cascade:
                children = self.get_subcampaigns(cid)
                for child in children:
                    _delete_recursive(child.id)

            # Count plan items and their dependencies
            folder = self._campaign_folder(cid)
            if folder:
                items = self._read_plan_items_raw(cid)
                for item in items:
                    dep_count = len(item.get("depends_on", []))
                    counts["dependencies"] += dep_count
                counts["plan_items"] += len(items)

                # Remove campaign folder
                shutil.rmtree(folder)
                counts["campaigns"] += 1
                self._campaign_index.pop(cid, None)

        _delete_recursive(campaign_id)
        return counts

    def get_subcampaigns(self, campaign_id: str) -> list[Campaign]:
        children = []
        for _cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if data and data.get("parent_id") == campaign_id:
                children.append(self._dict_to_campaign(data))
        children.sort(key=lambda c: c.created_at)
        return children

    def get_nth_subcampaign(self, parent_id: str, n: int) -> Campaign | None:
        # Tolerate n arriving as a numeric string (tool args are often stringified).
        try:
            n = int(n)
        except (ValueError, TypeError):
            return None
        phases = self.get_subcampaigns(parent_id)
        if 1 <= n <= len(phases):
            return phases[n - 1]
        return None

    def get_campaign_tree(self, campaign_id: str) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {}
        children = self.get_subcampaigns(campaign_id)
        return {
            "campaign": campaign,
            "children": [self.get_campaign_tree(c.id) for c in children],
        }

    def get_root_campaigns(self, status: str | None = "active") -> list[Campaign]:
        """Get root campaigns (no parent). If status is None, returns all."""
        roots = []
        for _cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if data and data.get("parent_id") is None:
                if status is None or data.get("status") == status:
                    roots.append(self._dict_to_campaign(data))
        roots.sort(key=lambda c: c.updated_at or c.created_at, reverse=True)
        return roots

    # ------------------------------------------------------------------
    # Campaign sharing (mesh coordination)
    # ------------------------------------------------------------------

    def share_campaign(self, campaign_id: str):
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return
        data["is_shared"] = True
        data["updated_at"] = self._now()
        self._write_yaml(folder / "campaign.yaml", data)

    def unshare_campaign(self, campaign_id: str):
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return
        data["is_shared"] = False
        data["updated_at"] = self._now()
        self._write_yaml(folder / "campaign.yaml", data)

    def get_shared_campaigns(self) -> list[Campaign]:
        shared = []
        for _cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if data and data.get("is_shared"):
                shared.append(self._dict_to_campaign(data))
        shared.sort(key=lambda c: c.created_at)
        return shared

    def add_campaign_participant(self, campaign_id: str, instance_id: str, hostname: str):
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return
        participants = data.get("participants", [])
        # Replace existing entry for this instance_id
        participants = [p for p in participants if p.get("instance_id") != instance_id]
        participants.append(
            {
                "campaign_id": campaign_id,
                "instance_id": instance_id,
                "hostname": hostname,
                "joined_at": self._now(),
            }
        )
        data["participants"] = participants
        self._write_yaml(folder / "campaign.yaml", data)

    def get_campaign_participants(self, campaign_id: str) -> list[dict]:
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return []
        data = self._read_yaml(folder / "campaign.yaml")
        if not data:
            return []
        participants = data.get("participants", [])
        participants.sort(key=lambda p: p.get("joined_at", ""))
        return participants

    # ------------------------------------------------------------------
    # Plan item claim (mesh)
    # ------------------------------------------------------------------

    def claim_plan_item(self, item_id: str, instance_id: str, hostname: str) -> bool:
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return False
        campaign_id, items, idx = loc
        item = items[idx]
        existing_claim = item.get("claimed_by")
        if existing_claim is not None and existing_claim != instance_id:
            return False
        item["claimed_by"] = instance_id
        item["claimed_by_hostname"] = hostname
        item["updated_at"] = self._now()
        self._write_plan_items(campaign_id, items)
        return True

    def unclaim_plan_item(self, item_id: str):
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return
        campaign_id, items, idx = loc
        items[idx]["claimed_by"] = None
        items[idx]["claimed_by_hostname"] = None
        items[idx]["updated_at"] = self._now()
        self._write_plan_items(campaign_id, items)

    # ==================================================================
    # Projects
    # ==================================================================

    def create_project(
        self,
        description: str,
        campaign_id: str | None = None,
        project_id: str | None = None,
    ) -> str:
        pid = project_id or self._gen_id()
        now = self._now()
        slug = self._slugify(description)
        data = {
            "id": pid,
            "description": description,
            "campaign_id": campaign_id,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }
        self._write_yaml(
            self.agent_dir / "projects" / f"{pid}_{slug}.yaml",
            data,
        )
        logger.info(f"Created project {pid}: {description}")
        return pid

    def get_active_projects(self) -> list[Project]:
        projects: list[Project] = []
        proj_dir = self.agent_dir / "projects"
        if not proj_dir.exists():
            return projects
        for f in proj_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data and data.get("status") == "active":
                    projects.append(self._dict_to_project(data))
        projects.sort(key=lambda p: p.created_at, reverse=True)
        return projects

    # ==================================================================
    # Session Intents
    # ==================================================================

    def create_session_intent(
        self,
        session_id: str,
        planned_intent: str | None = None,
        campaign_ids: list[str] | None = None,
    ):
        now = self._now()
        path = self.agent_dir / "session_intents" / f"{session_id}.yaml"
        existing = self._read_yaml(path)
        data = existing or {}
        data.update(
            {
                "session_id": session_id,
                "planned_intent": planned_intent
                if planned_intent is not None
                else data.get("planned_intent"),
                "created_at": data.get("created_at", now),
                "campaign_ids": data.get("campaign_ids", []),
            }
        )
        if "actual_summary" not in data:
            data["actual_summary"] = None
        if "completed_at" not in data:
            data["completed_at"] = None
        self._write_yaml(path, data)

        if campaign_ids:
            for cid in campaign_ids:
                self.link_session_campaign(session_id, cid)

    def get_current_session_intent(self) -> SessionIntent | None:
        si_dir = self.agent_dir / "session_intents"
        if not si_dir.exists():
            return None
        # Find the most recent incomplete session intent
        candidates = []
        for f in si_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data and not data.get("completed_at"):
                    candidates.append(data)
        if not candidates:
            return None
        candidates.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        d = candidates[0]
        return self._dict_to_session_intent(d)

    def complete_session_intent(self, session_id: str, actual_summary: str):
        path = self.agent_dir / "session_intents" / f"{session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return
        data["actual_summary"] = actual_summary
        data["completed_at"] = self._now()
        self._write_yaml(path, data)

    # ==================================================================
    # Operation Plans
    # ==================================================================

    def set_operation_plan(self, session_id: str, plan: dict) -> None:
        """Persist the agent-authored Operation Plan for a session.

        The plan dict is stored verbatim (agent is the source of truth).
        Fires CONTEXT_UPDATED so the Operations UI refreshes live.
        """
        path = self.agent_dir / "operation_plans" / f"{session_id}.yaml"
        self._write_yaml(path, plan)
        self._notify_context_change("operation_plan")

    def get_operation_plan(self, session_id: str) -> dict | None:
        """Return the stored Operation Plan for a session, or None if absent."""
        path = self.agent_dir / "operation_plans" / f"{session_id}.yaml"
        return self._read_yaml(path)

    def transition_tactic(
        self, session_id: str, tactic_id: str, state: str | None = None, **bind
    ) -> bool:
        """Atomically update a tactic's state and/or bind live values onto it.

        Reads the plan via get_operation_plan, locates the tactic by id,
        sets its state (if provided), merges bind kwargs into its live dict
        (creating it if absent), stamps updated_at/updated_reason, and writes
        back via set_operation_plan so CONTEXT_UPDATED fires exactly once.

        Returns True on success, False if the plan is absent or the tactic id
        is not found (no-op, no crash).

        Note: read-modify-write with no lock; safe because all subscribed event
        emissions run on the single asyncio loop thread — revisit if a
        worker-thread emitter is ever added.
        """
        plan = self.get_operation_plan(session_id)
        if plan is None:
            return False

        tactics = plan.get("tactics", [])
        tactic = next((t for t in tactics if t.get("id") == tactic_id), None)
        if tactic is None:
            return False

        if state is not None:
            tactic["state"] = state

        if bind:
            live = tactic.setdefault("live", {})
            live.update(bind)

        plan["updated_at"] = self._now()
        plan["updated_reason"] = f"tactic {tactic_id} transitioned"

        self.set_operation_plan(session_id, plan)
        return True

    # ==================================================================
    # Session <-> Campaign (many-to-many)
    # ==================================================================

    def link_session_campaign(self, session_id: str, campaign_id: str):
        # Ensure session_intents entry exists
        path = self.agent_dir / "session_intents" / f"{session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            data = {
                "session_id": session_id,
                "planned_intent": None,
                "actual_summary": None,
                "campaign_ids": [],
                "created_at": self._now(),
                "completed_at": None,
            }
        cids = data.get("campaign_ids", [])
        if campaign_id not in cids:
            cids.append(campaign_id)
        data["campaign_ids"] = cids
        self._write_yaml(path, data)
        self._notify_plan_change(campaign_id)

    def unlink_session_campaign(self, session_id: str, campaign_id: str):
        path = self.agent_dir / "session_intents" / f"{session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return
        cids = data.get("campaign_ids", [])
        if campaign_id in cids:
            cids.remove(campaign_id)
        data["campaign_ids"] = cids
        self._write_yaml(path, data)

    def get_campaign_ids_for_session(self, session_id: str) -> list[str]:
        path = self.agent_dir / "session_intents" / f"{session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return []
        return data.get("campaign_ids", [])

    def get_campaigns_for_session(self, session_id: str) -> list[Campaign]:
        cids = self.get_campaign_ids_for_session(session_id)
        result = []
        for cid in cids:
            c = self.get_campaign(cid)
            if c:
                result.append(c)
        return result

    def get_sessions_for_campaign(self, campaign_id: str) -> list[SessionIntent]:
        results: list[SessionIntent] = []
        si_dir = self.agent_dir / "session_intents"
        if not si_dir.exists():
            return results
        for f in si_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data and campaign_id in data.get("campaign_ids", []):
                    results.append(self._dict_to_session_intent(data))
        results.sort(key=lambda s: s.created_at)
        return results

    # ==================================================================
    # Planned Sessions
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
        psid = planned_session_id or self._gen_id()
        now = self._now()
        data = {
            "id": psid,
            "title": title,
            "notes": notes,
            "scheduled_date": scheduled_date,
            "scheduled_time": scheduled_time,
            "estimated_duration_minutes": estimated_duration_minutes,
            "acquisition_params": acquisition_params,
            "source_session_id": source_session_id,
            "status": "planned",
            "session_id": None,
            "campaign_ids": campaign_ids or [],
            "created_at": now,
            "updated_at": now,
        }
        self._write_yaml(
            self.agent_dir / "planned_sessions" / f"{psid}.yaml",
            data,
        )
        logger.info(
            f"Created planned session {psid} for {scheduled_date}: {title or notes or '(untitled)'}"
        )
        return psid

    def get_planned_session(self, planned_session_id: str) -> PlannedSession | None:
        path = self.agent_dir / "planned_sessions" / f"{planned_session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return None
        return self._dict_to_planned_session(data)

    def get_planned_sessions(
        self,
        status: str | None = None,
        campaign_id: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[PlannedSession]:
        results: list[PlannedSession] = []
        ps_dir = self.agent_dir / "planned_sessions"
        if not ps_dir.exists():
            return results
        for f in ps_dir.iterdir():
            if f.suffix not in (".yaml", ".yml"):
                continue
            data = self._read_yaml(f)
            if not data:
                continue
            if status and data.get("status") != status:
                continue
            if campaign_id and campaign_id not in data.get("campaign_ids", []):
                continue
            sd = data.get("scheduled_date")
            if from_date and (not sd or sd < from_date):
                continue
            if to_date and (not sd or sd > to_date):
                continue
            results.append(self._dict_to_planned_session(data))
        results.sort(key=lambda ps: (ps.scheduled_date or "", ps.scheduled_time or ""))
        return results

    def get_upcoming_sessions(self, limit: int = 10) -> list[PlannedSession]:
        today = datetime.now().strftime("%Y-%m-%d")
        results: list[PlannedSession] = []
        ps_dir = self.agent_dir / "planned_sessions"
        if not ps_dir.exists():
            return results
        for f in ps_dir.iterdir():
            if f.suffix not in (".yaml", ".yml"):
                continue
            data = self._read_yaml(f)
            if not data:
                continue
            if data.get("status") != "planned":
                continue
            sd = data.get("scheduled_date")
            if not sd or sd < today:
                continue
            results.append(self._dict_to_planned_session(data))
        results.sort(key=lambda ps: (ps.scheduled_date or "", ps.scheduled_time or ""))
        return results[:limit]

    def get_todays_sessions(self) -> list[PlannedSession]:
        today = datetime.now().strftime("%Y-%m-%d")
        results: list[PlannedSession] = []
        ps_dir = self.agent_dir / "planned_sessions"
        if not ps_dir.exists():
            return results
        for f in ps_dir.iterdir():
            if f.suffix not in (".yaml", ".yml"):
                continue
            data = self._read_yaml(f)
            if not data:
                continue
            if data.get("scheduled_date") != today:
                continue
            if data.get("status") not in ("planned", "active"):
                continue
            results.append(self._dict_to_planned_session(data))
        results.sort(key=lambda ps: ps.scheduled_time or "")
        return results

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
        path = self.agent_dir / "planned_sessions" / f"{planned_session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return
        for key, val in [
            ("title", title),
            ("notes", notes),
            ("scheduled_date", scheduled_date),
            ("scheduled_time", scheduled_time),
            ("estimated_duration_minutes", estimated_duration_minutes),
            ("source_session_id", source_session_id),
            ("session_id", session_id),
        ]:
            if val is not None:
                data[key] = val
        if acquisition_params is not None:
            data["acquisition_params"] = acquisition_params
        if status is not None:
            data["status"] = status.value
        data["updated_at"] = self._now()
        self._write_yaml(path, data)

    def start_planned_session(self, planned_session_id: str, session_id: str):
        self.update_planned_session(
            planned_session_id,
            status=PlannedSessionStatus.ACTIVE,
            session_id=session_id,
        )

    def complete_planned_session(self, planned_session_id: str):
        self.update_planned_session(
            planned_session_id,
            status=PlannedSessionStatus.COMPLETED,
        )

    def link_planned_session_campaign(self, planned_session_id: str, campaign_id: str):
        path = self.agent_dir / "planned_sessions" / f"{planned_session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return
        cids = data.get("campaign_ids", [])
        if campaign_id not in cids:
            cids.append(campaign_id)
        data["campaign_ids"] = cids
        self._write_yaml(path, data)

    def unlink_planned_session_campaign(self, planned_session_id: str, campaign_id: str):
        path = self.agent_dir / "planned_sessions" / f"{planned_session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return
        cids = data.get("campaign_ids", [])
        if campaign_id in cids:
            cids.remove(campaign_id)
        data["campaign_ids"] = cids
        self._write_yaml(path, data)

    def get_campaign_ids_for_planned_session(self, planned_session_id: str) -> list[str]:
        path = self.agent_dir / "planned_sessions" / f"{planned_session_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return []
        return data.get("campaign_ids", [])

    # ==================================================================
    # Plan Items
    # ==================================================================

    def _read_plan_items_raw(self, campaign_id: str) -> list[dict]:
        """Read the raw plan items list for a campaign."""
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return []
        data = self._read_yaml(folder / "plan" / "current.yaml")
        if not data or not isinstance(data, list):
            return []
        return data

    def _write_plan_items(self, campaign_id: str, items: list[dict]):
        """Write the plan items list for a campaign."""
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return
        self._write_yaml(folder / "plan" / "current.yaml", items)

    def _find_plan_item_location(
        self,
        item_id: str,
    ) -> tuple | None:
        """Find a plan item across all campaigns.

        Returns (campaign_id, items_list, index) or None.
        """
        for cid in self._campaign_index:
            items = self._read_plan_items_raw(cid)
            for i, item in enumerate(items):
                if item.get("id") == item_id:
                    return (cid, items, i)
        return None

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
        pid = item_id or self._gen_id()
        now = self._now()
        items = self._read_plan_items_raw(campaign_id)

        if phase_order < 0:
            max_order = 0
            for it in items:
                order = it.get("phase_order", 0)
                if order > max_order:
                    max_order = order
            phase_order = max_order + 1

        item_data = {
            "id": pid,
            "campaign_id": campaign_id,
            "type": type,
            "title": title,
            "description": description,
            "status": "planned",
            "outcome": None,
            "spec": spec,
            "inherit_from": inherit_from,
            "planned_session_id": planned_session_id,
            "session_id": None,
            "session_ids": [],
            "estimated_days": estimated_days,
            "phase_order": phase_order,
            "references": references,
            "depends_on": depends_on or [],
            "claimed_by": None,
            "claimed_by_hostname": None,
            "created_at": now,
            "updated_at": now,
        }
        items.append(item_data)
        self._write_plan_items(campaign_id, items)
        self._notify_plan_change(campaign_id)
        logger.info(f"Created plan item {pid} [{type}] #{phase_order}: {title}")
        return pid

    def get_plan_item(self, item_id: str) -> PlanItem | None:
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return None
        campaign_id, items, idx = loc
        return self._dict_to_plan_item(items[idx])

    def resolve_plan_item(
        self,
        ref: str,
        campaign_id: str | None = None,
    ) -> PlanItem | None:
        ref = ref.strip().lower()

        # Direct ID match
        item = self.get_plan_item(ref)
        if item:
            return item

        # UUID prefix match
        if len(ref) >= 4 and re.match(r"^[0-9a-f]+$", ref):
            for cid in self._campaign_index:
                for raw in self._read_plan_items_raw(cid):
                    if raw.get("id", "").startswith(ref):
                        return self._dict_to_plan_item(raw)

        # Parse campaign.phase.task / phase.task / task
        phase_num = None
        task_num = None

        # "campaign.phase.task"
        m = re.match(r"^([^.\s]+)\.(\d+)\.(\d+)$", ref)
        if m:
            campaign_label = m.group(1)
            phase_num, task_num = int(m.group(2)), int(m.group(3))
            resolved = self._resolve_campaign_label(campaign_label)
            if resolved:
                campaign_id = resolved

        # "1.3" or "2.1"
        if not task_num:
            m = re.match(r"^(\d+)\.(\d+)$", ref)
            if m:
                phase_num, task_num = int(m.group(1)), int(m.group(2))

        # "task 3 of phase 1"
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

        # Determine root campaign
        root_id = campaign_id
        if not root_id:
            campaigns = self.get_root_campaigns()
            if campaigns:
                root_id = campaigns[0].id

        if not root_id:
            return None

        # Resolve phase -> campaign_id
        if phase_num is not None:
            phases = self.get_subcampaigns(root_id)
            if 1 <= phase_num <= len(phases):
                target_campaign = phases[phase_num - 1].id
            else:
                return None
        else:
            phases = self.get_subcampaigns(root_id)
            if phases:
                all_items: list[PlanItem] = []
                for phase in phases:
                    p_items = self.get_plan_items(campaign_id=phase.id)
                    p_items.sort(key=lambda x: x.phase_order)
                    all_items.extend(p_items)
                if 1 <= task_num <= len(all_items):
                    return all_items[task_num - 1]
                return None
            else:
                target_campaign = root_id

        items = self.get_plan_items(campaign_id=target_campaign)
        items.sort(key=lambda x: x.phase_order)
        if 1 <= task_num <= len(items):
            return items[task_num - 1]

        return None

    def get_plan_items(
        self,
        campaign_id: str | None = None,
        status: str | None = None,
        type: str | None = None,
        include_children: bool = False,
    ) -> list[PlanItem]:
        if campaign_id and include_children:
            cids = self._get_campaign_tree_ids(campaign_id)
        elif campaign_id:
            cids = [campaign_id]
        else:
            cids = list(self._campaign_index.keys())

        result: list[PlanItem] = []
        for cid in cids:
            for raw in self._read_plan_items_raw(cid):
                if status and raw.get("status") != status:
                    continue
                if type and raw.get("type") != type:
                    continue
                result.append(self._dict_to_plan_item(raw))
        result.sort(key=lambda it: (it.phase_order, it.created_at))
        return result

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
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return
        old_campaign_id, items, idx = loc
        item = items[idx]

        for key, val in [
            ("title", title),
            ("description", description),
            ("outcome", outcome),
            ("planned_session_id", planned_session_id),
            ("session_id", session_id),
            ("estimated_days", estimated_days),
        ]:
            if val is not None:
                item[key] = val
        if status is not None:
            item["status"] = status.value
        if spec is not None:
            item["spec"] = spec
        if phase_order is not None:
            item["phase_order"] = phase_order
        if references is not None:
            item["references"] = references
        if campaign_id is not None and campaign_id != old_campaign_id:
            # Move item to a different campaign
            items.pop(idx)
            self._write_plan_items(old_campaign_id, items)
            item["campaign_id"] = campaign_id
            item["updated_at"] = self._now()
            new_items = self._read_plan_items_raw(campaign_id)
            new_items.append(item)
            self._write_plan_items(campaign_id, new_items)
            self._notify_plan_change(campaign_id)
            return

        item["updated_at"] = self._now()
        self._write_plan_items(old_campaign_id, items)
        self._notify_plan_change(old_campaign_id)

    def complete_plan_item(self, item_id: str, outcome: str):
        self.update_plan_item(
            item_id,
            status=PlanItemStatus.COMPLETED,
            outcome=outcome,
        )

    def link_plan_item_session(
        self, item_id: str, session_id: str, set_in_progress: bool = True
    ) -> bool:
        """Attach a session to a plan item — APPENDS (an item may run several times:
        re-runs, multi-sitting, more embryos later). Records the session as the latest
        `session_id` (back-compat), flips a PLANNED item to IN_PROGRESS, emits PLAN_UPDATED.
        Returns False if the item isn't found."""
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return False
        campaign_id, items, idx = loc
        item = items[idx]
        sids = item.get("session_ids") or ([item["session_id"]] if item.get("session_id") else [])
        if session_id and session_id not in sids:
            sids.append(session_id)
        item["session_ids"] = sids
        if sids:
            item["session_id"] = sids[-1]  # most recent run; back-compat for older readers
        if set_in_progress and item.get("status") == "planned":
            item["status"] = PlanItemStatus.IN_PROGRESS.value
        item["updated_at"] = self._now()
        self._write_plan_items(campaign_id, items)
        self._notify_plan_change(campaign_id)
        return True

    def unlink_plan_item_session(self, item_id: str, session_id: str) -> bool:
        """Remove a session from a plan item's session_ids list.

        Mirrors the load/persist/notify pattern of link_plan_item_session.
        Clears the back-compat scalar session_id when it matched the removed
        session (sets it to the most-recent remaining session_id, or None).
        Idempotent: returns False without writing if the session isn't linked.
        Returns True on successful removal.
        """
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return False
        campaign_id, items, idx = loc
        item = items[idx]
        sids = item.get("session_ids") or ([item["session_id"]] if item.get("session_id") else [])
        if session_id not in sids:
            return False
        sids = [s for s in sids if s != session_id]
        item["session_ids"] = sids
        item["session_id"] = sids[-1] if sids else None  # back-compat: most recent remaining
        item["updated_at"] = self._now()
        self._write_plan_items(campaign_id, items)
        self._notify_plan_change(campaign_id)
        return True

    def get_plan_items_for_session(self, session_id: str) -> list["PlanItem"]:
        """Return all plan items linked to a session.

        Iterates active campaigns only (mirrors the normal read path).
        Back-compat: matches items whose scalar session_id equals the query even
        when session_ids is empty (old items written before the list field existed).
        Deduplicates by item id.
        """
        seen: set[str] = set()
        result: list[PlanItem] = []
        for campaign in self.get_active_campaigns():
            for item in self.get_plan_items(campaign.id):
                if item.id in seen:
                    continue
                # session_ids is already populated from back-compat in _dict_to_plan_item
                if session_id in (item.session_ids or []) or item.session_id == session_id:
                    seen.add(item.id)
                    result.append(item)
        return result

    def skip_plan_item(self, item_id: str, reason: str | None = None):
        self.update_plan_item(
            item_id,
            status=PlanItemStatus.SKIPPED,
            outcome=reason or "Skipped",
        )

    def delete_plan_item(self, item_id: str) -> bool:
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return False
        campaign_id, items, idx = loc
        items.pop(idx)
        # Remove this ID from other items' depends_on lists
        for item in items:
            deps = item.get("depends_on", [])
            if item_id in deps:
                deps.remove(item_id)
        # Also clean up depends_on in ALL campaigns
        for cid in self._campaign_index:
            if cid == campaign_id:
                continue
            other_items = self._read_plan_items_raw(cid)
            changed = False
            for oitem in other_items:
                deps = oitem.get("depends_on", [])
                if item_id in deps:
                    deps.remove(item_id)
                    changed = True
            if changed:
                self._write_plan_items(cid, other_items)
        self._write_plan_items(campaign_id, items)
        logger.info(f"Deleted plan item {item_id}")
        return True

    def add_plan_item_dependency(self, item_id: str, depends_on_id: str):
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return
        campaign_id, items, idx = loc
        deps = items[idx].get("depends_on", [])
        if depends_on_id not in deps:
            deps.append(depends_on_id)
        items[idx]["depends_on"] = deps
        self._write_plan_items(campaign_id, items)

    def remove_plan_item_dependency(self, item_id: str, depends_on_id: str):
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return
        campaign_id, items, idx = loc
        deps = items[idx].get("depends_on", [])
        if depends_on_id in deps:
            deps.remove(depends_on_id)
        items[idx]["depends_on"] = deps
        self._write_plan_items(campaign_id, items)

    def get_plan_item_dependencies(self, item_id: str) -> list[str]:
        loc = self._find_plan_item_location(item_id)
        if not loc:
            return []
        _, items, idx = loc
        return list(items[idx].get("depends_on", []))

    def get_plan_item_dependents(self, item_id: str) -> list[str]:
        """Get IDs of items that depend on this item."""
        dependents: list[str] = []
        for cid in self._campaign_index:
            for raw in self._read_plan_items_raw(cid):
                if item_id in raw.get("depends_on", []):
                    dependents.append(raw["id"])
        return dependents

    def get_unblocked_plan_items(self, campaign_id: str) -> list[PlanItem]:
        items = self.get_plan_items(
            campaign_id=campaign_id,
            status="planned",
            include_children=True,
        )
        unblocked: list[PlanItem] = []
        for item in items:
            if not item.depends_on:
                unblocked.append(item)
                continue
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

            type_key = item.type.value
            if type_key not in result["by_type"]:
                result["by_type"][type_key] = {"total": 0, "completed": 0}
            result["by_type"][type_key]["total"] += 1
            if item.status == PlanItemStatus.COMPLETED:
                result["by_type"][type_key]["completed"] += 1

            if item.type == PlanItemType.DECISION_POINT and item.status == PlanItemStatus.PLANNED:
                result["pending_decisions"].append(item)

        result["next_actions"] = self.get_unblocked_plan_items(campaign_id)
        return result

    def resolve_imaging_spec(self, item: PlanItem) -> ImagingSpec | None:
        if item.type != PlanItemType.IMAGING:
            return None
        if not item.inherit_from:
            return item.imaging_spec
        parent = self.get_plan_item(item.inherit_from)
        if not parent:
            return item.imaging_spec
        parent_spec = self.resolve_imaging_spec(parent)
        if not parent_spec:
            return item.imaging_spec
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
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        template_data = self._serialize_campaign_tree(campaign_id)
        tid = self._gen_id()
        now = self._now()
        folder = self._campaign_folder(campaign_id)
        # Store template in the campaign's templates/ folder AND globally
        # accessible via list_plan_templates which scans all campaigns.
        if folder:
            tpl_data = {
                "id": tid,
                "name": name,
                "description": description,
                "template_json": template_data,
                "campaign_id": campaign_id,
                "created_at": now,
                "updated_at": now,
            }
            self._write_yaml(folder / "templates" / f"{name}.yaml", tpl_data)
        logger.info(f"Saved plan template '{name}' ({tid})")
        return tid

    def _serialize_campaign_tree(self, campaign_id: str) -> dict:
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {}
        items = self.get_plan_items(campaign_id=campaign_id)
        items.sort(key=lambda x: x.phase_order)

        all_item_ids = [it.id for it in items]
        serialized_items: list[dict] = []
        for item in items:
            item_data: dict[str, Any] = {
                "type": item.type.value,
                "title": item.title,
                "description": item.description,
                "phase_order": item.phase_order,
            }
            if item.imaging_spec:
                spec_dict = {}
                for f in dataclasses.fields(item.imaging_spec):
                    val = getattr(item.imaging_spec, f.name)
                    if val is not None:
                        spec_dict[f.name] = val
                item_data["spec"] = spec_dict
            elif item.bench_spec:
                spec_dict = {}
                for f in dataclasses.fields(item.bench_spec):
                    val = getattr(item.bench_spec, f.name)
                    if val is not None:
                        spec_dict[f.name] = val
                item_data["spec"] = spec_dict

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

        children = self.get_subcampaigns(campaign_id)
        serialized_children = [self._serialize_campaign_tree(child.id) for child in children]

        return {
            "description": campaign.description,
            "shorthand": campaign.shorthand,
            "target": campaign.target,
            "items": serialized_items,
            "children": serialized_children,
        }

    def list_plan_templates(self) -> list[dict]:
        templates: list[dict] = []
        for _cid, folder in self._campaign_index.items():
            tpl_dir = folder / "templates"
            if not tpl_dir.exists():
                continue
            for f in tpl_dir.iterdir():
                if f.suffix in (".yaml", ".yml"):
                    data = self._read_yaml(f)
                    if data:
                        templates.append(
                            {
                                "id": data.get("id", ""),
                                "name": data.get("name", ""),
                                "description": data.get("description"),
                                "created_at": data.get("created_at", ""),
                                "updated_at": data.get("updated_at", ""),
                            }
                        )
        templates.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return templates

    def get_plan_template(self, id_or_name: str) -> dict | None:
        for _cid, folder in self._campaign_index.items():
            tpl_dir = folder / "templates"
            if not tpl_dir.exists():
                continue
            for f in tpl_dir.iterdir():
                if f.suffix in (".yaml", ".yml"):
                    data = self._read_yaml(f)
                    if data and (data.get("id") == id_or_name or data.get("name") == id_or_name):
                        return data
        return None

    def apply_plan_template(
        self,
        template_id: str,
        overrides: dict | None = None,
    ) -> str:
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
        cid = self.create_campaign(
            description=data.get("description", "Untitled"),
            shorthand=data.get("shorthand"),
            target=data.get("target"),
            parent_id=parent_id,
        )
        items_data = data.get("items", [])
        new_item_ids: list[str] = []

        for item_data in items_data:
            spec = item_data.get("spec")
            if spec and item_data.get("type") == "imaging" and overrides:
                spec = dict(spec)
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

        for idx_item, item_data in enumerate(items_data):
            dep_indices = item_data.get("depends_on_indices", [])
            for dep_idx in dep_indices:
                if 0 <= dep_idx < len(new_item_ids):
                    self.add_plan_item_dependency(
                        new_item_ids[idx_item],
                        new_item_ids[dep_idx],
                    )

        for child_data in data.get("children", []):
            self._instantiate_template_tree(child_data, parent_id=cid, overrides=overrides)

        return cid

    def delete_plan_template(self, template_id: str) -> bool:
        for _cid, folder in self._campaign_index.items():
            tpl_dir = folder / "templates"
            if not tpl_dir.exists():
                continue
            for f in tpl_dir.iterdir():
                if f.suffix in (".yaml", ".yml"):
                    data = self._read_yaml(f)
                    if data and (data.get("id") == template_id or data.get("name") == template_id):
                        f.unlink()
                        return True
        return False

    # ==================================================================
    # Tactic Library
    # ==================================================================

    def save_tactic(self, tactic: dict, name: str | None = None) -> str:
        """Persist a tactic as a reusable template in agent/tactic_library/.

        Strips runtime state (live, state, original id) and assigns a new
        template id + slug. Fires CONTEXT_UPDATED. Returns the template id.
        """
        name = name or tactic.get("name") or "unnamed"
        slug = self._slugify(name)
        tid = self._gen_id()
        now = self._now()

        template = {
            "id": tid,
            "name": name,
            "slug": slug,
            "kind": tactic.get("kind", "unknown"),
            "structure": copy.deepcopy(tactic.get("structure") or {}),
            "scope_hint": copy.deepcopy(tactic.get("scope")),
            "description": tactic.get("description") or tactic.get("rationale"),
            "rationale": tactic.get("rationale"),
            "params": copy.deepcopy(tactic.get("params")),
            "relations": copy.deepcopy(tactic.get("relations") or {}),
            "live_bind": list(tactic.get("live_bind") or []),
            "created_at": now,
            "created_by": tactic.get("created_by", "agent"),
        }

        path = self.agent_dir / "tactic_library" / f"{tid}_{slug}.yaml"
        self._write_yaml(path, template)
        self._notify_context_change("tactic_library")
        logger.info(f"Saved tactic template '{name}' ({tid})")
        return tid

    def list_tactics(self) -> list[dict]:
        """List all saved tactic templates, ordered by created_at descending."""
        tl_dir = self.agent_dir / "tactic_library"
        if not tl_dir.exists():
            return []
        tactics: list[dict] = []
        for f in tl_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data:
                    tactics.append(data)
        tactics.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return tactics

    def get_tactic(self, id_or_name: str) -> dict | None:
        """Return a tactic template by id or name, or None if not found.

        Uses list_tactics() (sorted newest-first by created_at) so that name
        lookups are deterministic: on a name collision, the newest entry wins.
        id lookups are unique by construction so order does not matter.
        """
        tl_dir = self.agent_dir / "tactic_library"
        if not tl_dir.exists():
            return None
        for tactic in self.list_tactics():
            if tactic.get("id") == id_or_name or tactic.get("name") == id_or_name:
                return tactic
        return None

    def apply_tactic(self, id_or_name: str) -> dict | None:
        """Return a fresh planned tactic from a saved template.

        The returned dict has a new run id (distinct from the template id),
        state="planned", and no live/runtime state. Returns None if the
        template is not found.
        """
        tmpl = self.get_tactic(id_or_name)
        if tmpl is None:
            return None
        tactic = copy.deepcopy(tmpl)
        # Assign a new run id — must differ from the template id
        tactic["id"] = self._gen_id()
        tactic["state"] = "planned"
        # Promote scope_hint back to scope for the tactic
        scope_hint = tactic.pop("scope_hint", None)
        if scope_hint is not None:
            tactic["scope"] = scope_hint
        # Strip template-internal metadata
        tactic.pop("slug", None)
        tactic.pop("created_at", None)
        tactic.pop("created_by", None)
        # Strip runtime state (should not exist in a template, but guard anyway)
        tactic.pop("live", None)
        return tactic

    # ==================================================================
    # Plan Snapshots
    # ==================================================================

    def create_plan_snapshot(
        self,
        campaign_id: str,
        label: str | None = None,
        summary: str | None = None,
    ) -> str:
        snapshot_data = self._serialize_campaign_tree(campaign_id)
        if not summary:
            summary = self._generate_snapshot_summary(campaign_id)

        # Determine version number from existing snapshots
        folder = self._campaign_folder(campaign_id)
        if not folder:
            raise ValueError(f"Campaign {campaign_id} not found")

        history_dir = folder / "plan" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_all_snapshots(campaign_id)
        version_number = max((s.get("version_number", 0) for s in existing), default=0) + 1
        parent_version_id = None
        if existing:
            # Most recent snapshot by version_number
            existing.sort(key=lambda s: s.get("version_number", 0), reverse=True)
            parent_version_id = existing[0].get("version_id")

        version_id = self._gen_id()
        now = self._now()
        timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M")

        snapshot = {
            "version_id": version_id,
            "campaign_id": campaign_id,
            "version_number": version_number,
            "snapshot_json": snapshot_data,
            "summary": summary,
            "label": label,
            "parent_version_id": parent_version_id,
            "created_at": now,
        }
        self._write_yaml(history_dir / f"{timestamp_slug}.yaml", snapshot)
        logger.info(
            f"Created plan snapshot v{version_number} ({version_id}) for campaign {campaign_id}"
        )
        return version_id

    def _generate_snapshot_summary(self, campaign_id: str) -> str:
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return "Unknown campaign"
        phases = self.get_subcampaigns(campaign_id)
        items = self.get_plan_items(campaign_id=campaign_id, include_children=True)
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

    def _read_all_snapshots(self, campaign_id: str) -> list[dict]:
        """Read all snapshot files for a campaign."""
        folder = self._campaign_folder(campaign_id)
        if not folder:
            return []
        history_dir = folder / "plan" / "history"
        if not history_dir.exists():
            return []
        snapshots: list[dict] = []
        for f in history_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data:
                    snapshots.append(data)
        return snapshots

    def list_plan_snapshots(
        self,
        campaign_id: str,
        limit: int = 50,
    ) -> list[dict]:
        snapshots = self._read_all_snapshots(campaign_id)
        # Return metadata only (no blob)
        result = []
        for s in snapshots:
            result.append(
                {
                    "version_id": s.get("version_id"),
                    "campaign_id": s.get("campaign_id"),
                    "version_number": s.get("version_number"),
                    "summary": s.get("summary"),
                    "label": s.get("label"),
                    "parent_version_id": s.get("parent_version_id"),
                    "created_at": s.get("created_at"),
                }
            )
        result.sort(key=lambda s: s.get("version_number", 0) or 0, reverse=True)
        return result[:limit]

    def get_plan_snapshot(self, version_id: str) -> dict | None:
        for cid in self._campaign_index:
            for snap in self._read_all_snapshots(cid):
                if snap.get("version_id") == version_id:
                    return snap
        return None

    def restore_plan_snapshot(self, version_id: str) -> str:
        snapshot = self.get_plan_snapshot(version_id)
        if not snapshot:
            raise ValueError(f"Snapshot {version_id} not found")

        campaign_id = snapshot["campaign_id"]
        version_number = snapshot["version_number"]

        # Auto-snapshot current state
        try:
            self.create_plan_snapshot(
                campaign_id,
                label=f"auto: before restore to v{version_number}",
            )
        except Exception:
            pass

        campaign = self.get_campaign(campaign_id)
        parent_id = campaign.parent_id if campaign else None

        self.delete_campaign(campaign_id, cascade=True)

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
        ids = [campaign_id]
        for cid, folder in self._campaign_index.items():
            data = self._read_yaml(folder / "campaign.yaml")
            if data and data.get("parent_id") == campaign_id:
                ids.extend(self._get_campaign_tree_ids(cid))
        return ids

    # ==================================================================
    # Observations
    # ==================================================================

    def add_observation(self, obs: Observation):
        slug = self._slugify(obs.type + "-" + (obs.content[:20] if obs.content else ""))
        data = {
            "id": obs.id,
            "timestamp": obs.timestamp.isoformat(),
            "type": obs.type,
            "content": obs.content,
            "embryo_id": obs.embryo_id,
            "significance": obs.significance.value if obs.significance else "medium",
            "session_id": obs.session_id,
            "gently_refs": obs.gently_refs,
            "relates_to": obs.relates_to,
        }
        self._write_yaml(
            self.agent_dir / "observations" / f"{obs.id}_{slug}.yaml",
            data,
        )

    def get_recent_observations(self, limit: int = 50) -> list[Observation]:
        obs_dir = self.agent_dir / "observations"
        if not obs_dir.exists():
            return []
        all_obs: list[Observation] = []
        for f in obs_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data:
                    all_obs.append(self._dict_to_observation(data))
        all_obs.sort(key=lambda o: o.timestamp, reverse=True)
        # Return in chronological order (oldest first in the window)
        return list(reversed(all_obs[:limit]))

    def get_observations_for_embryo(self, embryo_id: str, limit: int = 20) -> list[Observation]:
        obs_dir = self.agent_dir / "observations"
        if not obs_dir.exists():
            return []
        matches: list[Observation] = []
        for f in obs_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data and data.get("embryo_id") == embryo_id:
                    matches.append(self._dict_to_observation(data))
        matches.sort(key=lambda o: o.timestamp, reverse=True)
        return list(reversed(matches[:limit]))

    # ==================================================================
    # Expectations
    # ==================================================================

    def _notify_context_change(self, kind: str = "context") -> None:
        """Emit CONTEXT_UPDATED on the global bus so the shared-visibility
        surface refreshes live. Best-effort — a bus failure never breaks a write."""
        try:
            from gently.core.event_bus import EventType, emit

            emit(EventType.CONTEXT_UPDATED, {"kind": kind}, source="context_store")
        except Exception:
            pass

    def _notify_plan_change(self, campaign_id: str | None = None) -> None:
        """Emit PLAN_UPDATED so the Plans UI refreshes live when a plan item or
        campaign changes (status, session link, new item, progress). Best-effort."""
        try:
            from gently.core.event_bus import EventType, emit

            emit(EventType.PLAN_UPDATED, {"campaign_id": campaign_id}, source="context_store")
        except Exception:
            pass

    def add_expectation(self, exp: Expectation):
        path = self.agent_dir / "active" / "expectations.yaml"
        items = self._read_yaml(path) or []
        items.append(
            {
                "id": exp.id,
                "target": exp.target,
                "prediction": exp.prediction,
                "expected_time": exp.expected_time.isoformat(),
                "uncertainty": exp.uncertainty,
                "basis": exp.basis,
                "status": exp.status.value,
                "created_at": exp.created_at.isoformat(),
                "resolved_at": None,
            }
        )
        self._write_yaml(path, items)
        self._notify_context_change("expectation")

    def get_pending_expectations(self) -> list[Expectation]:
        path = self.agent_dir / "active" / "expectations.yaml"
        items = self._read_yaml(path) or []
        pending = [self._dict_to_expectation(d) for d in items if d.get("status") == "pending"]
        pending.sort(key=lambda e: e.expected_time)
        return pending

    def get_expectation_for(self, target: str) -> Expectation | None:
        path = self.agent_dir / "active" / "expectations.yaml"
        items = self._read_yaml(path) or []
        candidates = [
            self._dict_to_expectation(d)
            for d in items
            if d.get("target") == target and d.get("status") == "pending"
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda e: e.expected_time)
        return candidates[0]

    def resolve_expectation(self, exp_id: str, status: ExpectationStatus):
        path = self.agent_dir / "active" / "expectations.yaml"
        items = self._read_yaml(path) or []
        now = self._now()
        for item in items:
            if item.get("id") == exp_id:
                item["status"] = status.value
                item["resolved_at"] = now
                break
        self._write_yaml(path, items)
        self._notify_context_change("expectation")

    # ==================================================================
    # Watchpoints
    # ==================================================================

    def add_watchpoint(self, wp: Watchpoint):
        path = self.agent_dir / "active" / "watchpoints.yaml"
        items = self._read_yaml(path) or []
        items.append(
            {
                "id": wp.id,
                "target": wp.target,
                "condition": wp.condition,
                "priority": wp.priority.value if wp.priority else "medium",
                "status": wp.status.value,
                "created_at": wp.created_at.isoformat(),
            }
        )
        self._write_yaml(path, items)
        self._notify_context_change("watchpoint")

    def get_active_watchpoints(self) -> list[Watchpoint]:
        path = self.agent_dir / "active" / "watchpoints.yaml"
        items = self._read_yaml(path) or []
        active = [self._dict_to_watchpoint(d) for d in items if d.get("status") == "active"]
        # Sort: high > medium > low, then by created_at
        priority_order = {"high": 0, "medium": 1, "low": 2}
        active.sort(key=lambda w: (priority_order.get(w.priority.value, 1), w.created_at))
        return active

    def trigger_watchpoint(self, wp_id: str):
        path = self.agent_dir / "active" / "watchpoints.yaml"
        items = self._read_yaml(path) or []
        for item in items:
            if item.get("id") == wp_id:
                item["status"] = "triggered"
                break
        self._write_yaml(path, items)

    def resolve_watchpoint(self, wp_id: str):
        path = self.agent_dir / "active" / "watchpoints.yaml"
        items = self._read_yaml(path) or []
        for item in items:
            if item.get("id") == wp_id:
                item["status"] = "resolved"
                break
        self._write_yaml(path, items)
        self._notify_context_change("watchpoint")

    # ==================================================================
    # Questions
    # ==================================================================

    def add_question(self, q: Question):
        path = self.agent_dir / "active" / "questions.yaml"
        items = self._read_yaml(path) or []
        items.append(
            {
                "id": q.id,
                "content": q.content,
                "status": q.status.value,
                "resolution": None,
                "created_at": q.created_at.isoformat(),
                "resolved_at": None,
            }
        )
        self._write_yaml(path, items)
        self._notify_context_change("question")

    def get_open_questions(self) -> list[Question]:
        path = self.agent_dir / "active" / "questions.yaml"
        items = self._read_yaml(path) or []
        open_qs = [
            self._dict_to_question(d) for d in items if d.get("status") in ("open", "investigating")
        ]
        open_qs.sort(key=lambda q: q.created_at)
        return open_qs

    def resolve_question(self, q_id: str, resolution: str):
        path = self.agent_dir / "active" / "questions.yaml"
        items = self._read_yaml(path) or []
        now = self._now()
        for item in items:
            if item.get("id") == q_id:
                item["status"] = "resolved"
                item["resolution"] = resolution
                item["resolved_at"] = now
                break
        self._write_yaml(path, items)
        self._notify_context_change("question")

    # ==================================================================
    # Learnings
    # ==================================================================

    def add_learning(self, learning: Learning):
        slug = self._slugify(learning.content[:30] if learning.content else "learning")
        data = {
            "id": learning.id,
            "content": learning.content,
            "confidence": learning.confidence.value if learning.confidence else "medium",
            "basis": learning.basis,
            "created_at": learning.created_at.isoformat(),
        }
        self._write_yaml(
            self.agent_dir / "learnings" / f"{learning.id}_{slug}.yaml",
            data,
        )

    def get_learnings(self, limit: int = 50) -> list[Learning]:
        learn_dir = self.agent_dir / "learnings"
        if not learn_dir.exists():
            return []
        all_learnings: list[Learning] = []
        for f in learn_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data:
                    all_learnings.append(self._dict_to_learning(data))
        all_learnings.sort(key=lambda learning: learning.created_at, reverse=True)
        return all_learnings[:limit]

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
        now = self._now()
        path = self.agent_dir / "embryo_understanding" / f"{embryo_id}.yaml"
        existing = self._read_yaml(path)

        if existing:
            notes = existing.get("notes", []) or []
            if note:
                notes.append(note)

            if current_stage is not None:
                existing["current_stage"] = current_stage
            if stage_confidence is not None:
                existing["stage_confidence"] = stage_confidence.value
            if health_assessment is not None:
                existing["health_assessment"] = health_assessment
            existing["notes"] = notes
            existing["last_observed"] = now
            if is_hatched is not None:
                existing["is_hatched"] = is_hatched
            if needs_attention is not None:
                existing["needs_attention"] = needs_attention
            if attention_reason is not None:
                existing["attention_reason"] = attention_reason
            self._write_yaml(path, existing)
        else:
            notes = [note] if note else []
            data = {
                "embryo_id": embryo_id,
                "current_stage": current_stage,
                "stage_confidence": stage_confidence.value if stage_confidence else None,
                "health_assessment": health_assessment,
                "notes": notes,
                "last_observed": now,
                "is_tracked": True,
                "is_hatched": bool(is_hatched) if is_hatched else False,
                "needs_attention": bool(needs_attention) if needs_attention else False,
                "attention_reason": attention_reason,
            }
            self._write_yaml(path, data)

    # ==================================================================
    # Agent State
    # ==================================================================

    def get_state(self, key: str) -> str | None:
        path = self.agent_dir / "state.yaml"
        data = self._read_yaml(path)
        if not data or not isinstance(data, dict):
            return None
        return data.get(key)

    def set_state(self, key: str, value: str):
        path = self.agent_dir / "state.yaml"
        data = self._read_yaml(path)
        if not data or not isinstance(data, dict):
            data = {}
        data[key] = value
        data.setdefault("_updated_at", {})
        data["_updated_at"][key] = self._now()
        self._write_yaml(path, data)

    # ==================================================================
    # Batch Updates
    # ==================================================================

    @property
    def notebook(self):
        """The shared lab notebook, rooted at agent_dir/notebook (lazy)."""
        nb = getattr(self, "_notebook", None)
        if nb is None:
            from .notebook import NotebookStore

            nb = NotebookStore(self.agent_dir / "notebook")
            self._notebook = nb
        return nb

    def apply_updates(self, updates: ContextUpdates):
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

        for exp_id, status in updates.resolved_expectations.items():
            self.resolve_expectation(exp_id, status)
        for wp_id in updates.triggered_watchpoints:
            self.trigger_watchpoint(wp_id)
        for q_id, resolution in updates.resolved_questions.items():
            self.resolve_question(q_id, resolution)

        for embryo_id, update_dict in updates.embryo_updates.items():
            self.update_embryo_understanding(embryo_id, **update_dict)
        for campaign_id, progress in updates.campaign_progress.items():
            self.update_campaign_progress(campaign_id, progress)

        if updates.new_focus is not None:
            self.set_state("current_focus", updates.new_focus)

        # Mirror new observations & learnings into the shared notebook
        # (best-effort — a notebook failure never breaks the legacy write).
        from .notebook import learning_to_note, observation_to_note

        try:
            for obs in updates.new_observations:
                self.notebook.write_note(observation_to_note(obs))
            for learning in updates.new_learnings:
                self.notebook.write_note(learning_to_note(learning))
        except Exception:
            logger.warning("notebook mirror failed", exc_info=True)

    # ==================================================================
    # ML Pipelines
    # ==================================================================

    def create_ml_pipeline(
        self,
        campaign_id: str,
        name: str,
        task: str = "embryo_stage_classification",
        model_config: dict | None = None,
        data_split: dict | None = None,
        training_config: dict | None = None,
    ) -> dict[str, Any]:
        pipeline_id = self._gen_id()
        now = self._now()
        data = {
            "id": pipeline_id,
            "campaign_id": campaign_id,
            "name": name,
            "task": task,
            "status": "planned",
            "model_config": model_config,
            "data_split": data_split,
            "training_config": training_config,
            "best_run_id": None,
            "best_accuracy": 0.0,
            "created_at": now,
            "updated_at": now,
        }
        self._write_yaml(
            self.agent_dir / "ml" / "pipelines" / f"{pipeline_id}.yaml",
            data,
        )
        created = self.get_ml_pipeline(pipeline_id)
        assert created is not None  # just written above
        return created

    def get_ml_pipeline(self, pipeline_id: str) -> dict[str, Any] | None:
        path = self.agent_dir / "ml" / "pipelines" / f"{pipeline_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return None
        return {
            "id": data["id"],
            "campaign_id": data["campaign_id"],
            "name": data["name"],
            "task": data.get("task", "embryo_stage_classification"),
            "status": data.get("status", "planned"),
            "model_config": data.get("model_config"),
            "data_split": data.get("data_split"),
            "training_config": data.get("training_config"),
            "best_run_id": data.get("best_run_id"),
            "best_accuracy": data.get("best_accuracy", 0.0),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        }

    def list_ml_pipelines(self, campaign_id: str | None = None) -> list[dict[str, Any]]:
        pipe_dir = self.agent_dir / "ml" / "pipelines"
        if not pipe_dir.exists():
            return []
        results: list[dict[str, Any]] = []
        for f in pipe_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if not data:
                    continue
                if campaign_id and data.get("campaign_id") != campaign_id:
                    continue
                pipeline = self.get_ml_pipeline(data["id"])
                if pipeline:
                    results.append(pipeline)
        results.sort(key=lambda p: p.get("created_at", ""), reverse=True)
        return results

    def update_ml_pipeline(self, pipeline_id: str, **kwargs) -> dict[str, Any] | None:
        path = self.agent_dir / "ml" / "pipelines" / f"{pipeline_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return self.get_ml_pipeline(pipeline_id)
        allowed = {
            "status",
            "model_config",
            "data_split",
            "training_config",
            "best_run_id",
            "best_accuracy",
            "name",
        }
        changed = False
        for k, v in kwargs.items():
            if k in allowed:
                data[k] = v
                changed = True
        if changed:
            data["updated_at"] = self._now()
            self._write_yaml(path, data)
        return self.get_ml_pipeline(pipeline_id)

    # ------------------------------------------------------------------
    # Training Runs
    # ------------------------------------------------------------------

    def create_training_run(
        self,
        pipeline_id: str,
        model_config: dict | None = None,
        training_config: dict | None = None,
        data_split: dict | None = None,
        peer_instance_id: str = "",
    ) -> dict[str, Any]:
        run_id = self._gen_id()
        data = {
            "id": run_id,
            "pipeline_id": pipeline_id,
            "status": "planned",
            "model_config": model_config,
            "training_config": training_config,
            "data_split": data_split,
            "current_epoch": 0,
            "total_epochs": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "val_accuracy": 0.0,
            "best_val_accuracy": 0.0,
            "model_weights_path": "",
            "metrics_path": "",
            "peer_instance_id": peer_instance_id,
            "started_at": "",
            "completed_at": "",
            "error_message": "",
        }
        self._write_yaml(
            self.agent_dir / "ml" / "runs" / f"{run_id}.yaml",
            data,
        )
        created = self.get_training_run(run_id)
        assert created is not None  # just written above
        return created

    def get_training_run(self, run_id: str) -> dict[str, Any] | None:
        path = self.agent_dir / "ml" / "runs" / f"{run_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return None
        return {
            "id": data["id"],
            "pipeline_id": data["pipeline_id"],
            "status": data.get("status", "planned"),
            "model_config": data.get("model_config"),
            "training_config": data.get("training_config"),
            "data_split": data.get("data_split"),
            "current_epoch": data.get("current_epoch", 0),
            "total_epochs": data.get("total_epochs", 0),
            "train_loss": data.get("train_loss", 0.0),
            "val_loss": data.get("val_loss", 0.0),
            "val_accuracy": data.get("val_accuracy", 0.0),
            "best_val_accuracy": data.get("best_val_accuracy", 0.0),
            "model_weights_path": data.get("model_weights_path", ""),
            "metrics_path": data.get("metrics_path", ""),
            "peer_instance_id": data.get("peer_instance_id", ""),
            "started_at": data.get("started_at", ""),
            "completed_at": data.get("completed_at", ""),
            "error_message": data.get("error_message", ""),
        }

    def list_training_runs(self, pipeline_id: str) -> list[dict[str, Any]]:
        runs_dir = self.agent_dir / "ml" / "runs"
        if not runs_dir.exists():
            return []
        results: list[dict[str, Any]] = []
        for f in runs_dir.iterdir():
            if f.suffix in (".yaml", ".yml"):
                data = self._read_yaml(f)
                if data and data.get("pipeline_id") == pipeline_id:
                    run = self.get_training_run(data["id"])
                    if run:
                        results.append(run)
        return results

    def update_training_run(self, run_id: str, **kwargs) -> dict[str, Any] | None:
        path = self.agent_dir / "ml" / "runs" / f"{run_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return self.get_training_run(run_id)
        allowed = {
            "status",
            "current_epoch",
            "total_epochs",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "best_val_accuracy",
            "model_weights_path",
            "metrics_path",
            "started_at",
            "completed_at",
            "error_message",
        }
        changed = False
        for k, v in kwargs.items():
            if k in allowed:
                data[k] = v
                changed = True
        if changed:
            self._write_yaml(path, data)
        return self.get_training_run(run_id)

    # ------------------------------------------------------------------
    # Data Assessments
    # ------------------------------------------------------------------

    def save_data_assessment(
        self,
        pipeline_id: str | None = None,
        total_sessions: int = 0,
        total_embryos: int = 0,
        total_volumes: int = 0,
        annotated_embryos: int = 0,
        stage_distribution: dict | None = None,
        coverage_gaps: list | None = None,
        quality_notes: str = "",
    ) -> dict[str, Any]:
        assessment_id = self._gen_id()
        now = self._now()
        data = {
            "id": assessment_id,
            "pipeline_id": pipeline_id,
            "total_sessions": total_sessions,
            "total_embryos": total_embryos,
            "total_volumes": total_volumes,
            "annotated_embryos": annotated_embryos,
            "stage_distribution": stage_distribution,
            "coverage_gaps": coverage_gaps,
            "quality_notes": quality_notes,
            "created_at": now,
        }
        self._write_yaml(
            self.agent_dir / "ml" / "assessments" / f"{assessment_id}.yaml",
            data,
        )
        created = self.get_data_assessment(assessment_id)
        assert created is not None  # just written above
        return created

    def get_data_assessment(self, assessment_id: str) -> dict[str, Any] | None:
        path = self.agent_dir / "ml" / "assessments" / f"{assessment_id}.yaml"
        data = self._read_yaml(path)
        if not data:
            return None
        return {
            "id": data["id"],
            "pipeline_id": data.get("pipeline_id"),
            "total_sessions": data.get("total_sessions", 0),
            "total_embryos": data.get("total_embryos", 0),
            "total_volumes": data.get("total_volumes", 0),
            "annotated_embryos": data.get("annotated_embryos", 0),
            "stage_distribution": data.get("stage_distribution"),
            "coverage_gaps": data.get("coverage_gaps"),
            "quality_notes": data.get("quality_notes", ""),
            "created_at": data.get("created_at"),
        }

    # ==================================================================
    # Dict -> dataclass converters
    # ==================================================================

    @staticmethod
    def _dict_to_campaign(d: dict) -> Campaign:
        return Campaign(
            id=d["id"],
            description=d["description"],
            shorthand=d.get("shorthand"),
            summary=d.get("summary"),
            target=d.get("target"),
            progress=d.get("progress"),
            parent_id=d.get("parent_id"),
            status=Status(d.get("status", "active")),
            is_shared=bool(d.get("is_shared", False)),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d["created_at"], str)
            else d["created_at"],
            updated_at=datetime.fromisoformat(d["updated_at"])
            if isinstance(d["updated_at"], str)
            else d["updated_at"],
        )

    @staticmethod
    def _dict_to_project(d: dict) -> Project:
        return Project(
            id=d["id"],
            description=d["description"],
            campaign_id=d.get("campaign_id"),
            status=Status(d.get("status", "active")),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d["created_at"], str)
            else d["created_at"],
            updated_at=datetime.fromisoformat(d["updated_at"])
            if isinstance(d["updated_at"], str)
            else d["updated_at"],
        )

    def _dict_to_session_intent(self, d: dict) -> SessionIntent:
        session_id = d["session_id"]
        campaign_ids = d.get("campaign_ids", [])
        return SessionIntent(
            session_id=session_id,
            planned_intent=d.get("planned_intent"),
            actual_summary=d.get("actual_summary"),
            campaign_ids=campaign_ids,
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d.get("created_at"), str)
            else d.get("created_at", datetime.now()),
            completed_at=datetime.fromisoformat(d["completed_at"])
            if d.get("completed_at") and isinstance(d["completed_at"], str)
            else None,
        )

    @staticmethod
    def _dict_to_planned_session(d: dict) -> PlannedSession:
        return PlannedSession(
            id=d["id"],
            title=d.get("title"),
            notes=d.get("notes"),
            scheduled_date=d.get("scheduled_date"),
            scheduled_time=d.get("scheduled_time"),
            estimated_duration_minutes=d.get("estimated_duration_minutes"),
            acquisition_params=d.get("acquisition_params"),
            source_session_id=d.get("source_session_id"),
            status=PlannedSessionStatus(d.get("status", "planned")),
            session_id=d.get("session_id"),
            campaign_ids=d.get("campaign_ids", []),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d.get("created_at"), str)
            else d.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(d["updated_at"])
            if isinstance(d.get("updated_at"), str)
            else d.get("updated_at", datetime.now()),
        )

    @staticmethod
    def _dict_to_plan_item(d: dict) -> PlanItem:
        item_type = PlanItemType(d["type"])
        spec_data = d.get("spec")
        imaging_spec = None
        bench_spec = None

        # Tolerate specs persisted as JSON strings (older tool calls that passed
        # spec as a string instead of an object) so read-back never crashes.
        if isinstance(spec_data, str):
            try:
                spec_data = json.loads(spec_data)
            except (json.JSONDecodeError, TypeError):
                spec_data = None

        if spec_data:
            if item_type == PlanItemType.IMAGING:
                valid = {f.name for f in dataclasses.fields(ImagingSpec)}
                imaging_spec = ImagingSpec(**{k: v for k, v in spec_data.items() if k in valid})
            else:
                valid = {f.name for f in dataclasses.fields(BenchSpec)}
                bench_spec = BenchSpec(**{k: v for k, v in spec_data.items() if k in valid})

        references = d.get("references") or []
        if isinstance(references, str):
            try:
                references = json.loads(references) or []
            except (json.JSONDecodeError, TypeError):
                references = []

        return PlanItem(
            id=d["id"],
            campaign_id=d["campaign_id"],
            type=item_type,
            title=d["title"],
            description=d.get("description"),
            status=PlanItemStatus(d.get("status", "planned")),
            depends_on=d.get("depends_on", []),
            outcome=d.get("outcome"),
            claimed_by=d.get("claimed_by"),
            claimed_by_hostname=d.get("claimed_by_hostname"),
            references=references,
            imaging_spec=imaging_spec,
            bench_spec=bench_spec,
            planned_session_id=d.get("planned_session_id"),
            session_id=d.get("session_id"),
            session_ids=d.get("session_ids") or ([d["session_id"]] if d.get("session_id") else []),
            inherit_from=d.get("inherit_from"),
            estimated_days=d.get("estimated_days"),
            phase_order=d.get("phase_order", 0),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d.get("created_at"), str)
            else d.get("created_at", datetime.now()),
            updated_at=datetime.fromisoformat(d["updated_at"])
            if isinstance(d.get("updated_at"), str)
            else d.get("updated_at", datetime.now()),
        )

    @staticmethod
    def _dict_to_observation(d: dict) -> Observation:
        return Observation(
            id=d["id"],
            timestamp=datetime.fromisoformat(d["timestamp"])
            if isinstance(d.get("timestamp"), str)
            else d.get("timestamp", datetime.now()),
            type=d["type"],
            content=d["content"],
            embryo_id=d.get("embryo_id"),
            significance=Significance(d.get("significance", "medium")),
            session_id=d.get("session_id"),
            gently_refs=d.get("gently_refs"),
            relates_to=d.get("relates_to"),
        )

    @staticmethod
    def _dict_to_expectation(d: dict) -> Expectation:
        return Expectation(
            id=d["id"],
            target=d["target"],
            prediction=d["prediction"],
            expected_time=datetime.fromisoformat(d["expected_time"])
            if isinstance(d.get("expected_time"), str)
            else d.get("expected_time", datetime.now()),
            uncertainty=d.get("uncertainty"),
            basis=d.get("basis"),
            status=ExpectationStatus(d.get("status", "pending")),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d.get("created_at"), str)
            else d.get("created_at", datetime.now()),
            resolved_at=datetime.fromisoformat(d["resolved_at"])
            if d.get("resolved_at") and isinstance(d["resolved_at"], str)
            else None,
        )

    @staticmethod
    def _dict_to_watchpoint(d: dict) -> Watchpoint:
        return Watchpoint(
            id=d["id"],
            target=d["target"],
            condition=d["condition"],
            priority=Significance(d.get("priority", "medium")),
            status=WatchpointStatus(d.get("status", "active")),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d.get("created_at"), str)
            else d.get("created_at", datetime.now()),
        )

    @staticmethod
    def _dict_to_question(d: dict) -> Question:
        return Question(
            id=d["id"],
            content=d["content"],
            status=QuestionStatus(d.get("status", "open")),
            resolution=d.get("resolution"),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d.get("created_at"), str)
            else d.get("created_at", datetime.now()),
            resolved_at=datetime.fromisoformat(d["resolved_at"])
            if d.get("resolved_at") and isinstance(d["resolved_at"], str)
            else None,
        )

    @staticmethod
    def _dict_to_learning(d: dict) -> Learning:
        return Learning(
            id=d["id"],
            content=d["content"],
            confidence=Confidence(d.get("confidence", "medium")),
            basis=d.get("basis"),
            created_at=datetime.fromisoformat(d["created_at"])
            if isinstance(d.get("created_at"), str)
            else d.get("created_at", datetime.now()),
        )

    @staticmethod
    def _dict_to_embryo_understanding(d: dict) -> EmbryoUnderstanding:
        return EmbryoUnderstanding(
            embryo_id=d["embryo_id"],
            current_stage=d.get("current_stage"),
            stage_confidence=Confidence(d["stage_confidence"])
            if d.get("stage_confidence")
            else None,
            health_assessment=d.get("health_assessment"),
            notes=d.get("notes") or [],
            last_observed=datetime.fromisoformat(d["last_observed"])
            if d.get("last_observed") and isinstance(d["last_observed"], str)
            else None,
            is_tracked=d.get("is_tracked", True),
            is_hatched=bool(d.get("is_hatched", False)),
            needs_attention=bool(d.get("needs_attention", False)),
            attention_reason=d.get("attention_reason"),
        )
