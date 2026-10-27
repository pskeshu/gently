"""Campaign routes - browse experimental plans and campaign hierarchy."""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from gently.harness.memory.acquisition import acquisition_from_legacy, effective_acquisition
from gently.harness.memory.model import PlanItem, PlanItemStatus
from gently.harness.memory.notebook import NoteKind, NoteStatus, note_to_dict
from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)


def _serialize_datetime(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_datetime(v) for v in obj]
    return obj


def _serialize_enum(obj: Any) -> Any:
    """Recursively convert enum values to strings."""
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, dict):
        return {k: _serialize_enum(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_enum(v) for v in obj]
    return obj


def _serialize(obj: Any) -> Any:
    """Full serialization: dataclass → dict → enums → datetimes. A plan item's
    spec goes out with the acquisition it means, in the pane's shape, whether
    it was written that way or only in the older flat fields."""
    d = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    d = _serialize_enum(d)
    d = _serialize_datetime(d)
    if (
        isinstance(obj, PlanItem)
        and isinstance(d, dict)
        and isinstance(d.get("imaging_spec"), dict)
    ):
        spec = d["imaging_spec"]
        explicit = bool(spec.get("acquisition"))
        acq = effective_acquisition(spec)
        spec["acquisition"] = acq
        spec["acquisition_derived"] = bool(acq) and not explicit
    return d


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _get_store():
        cs = getattr(server, "context_store", None)
        if cs is None:
            raise HTTPException(status_code=503, detail="Context store not available")
        return cs

    def _resolve(cs, campaign_id: str):
        """Resolve campaign by ID or shorthand, raise 404 if not found."""
        campaign = cs.resolve_campaign(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return campaign

    def _gently_store():
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        return getattr(agent, "store", None) or getattr(server, "gently_store", None)

    def _active_session_id():
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        return getattr(agent, "session_id", None) if agent is not None else None

    def _member_sessions(root_id: str) -> list[dict]:
        store = _gently_store()
        if store is None:
            return []
        try:
            return store.list_sessions(project_id=root_id)
        except Exception:
            return []

    def _project_summary(cs, root_id: str) -> dict:
        """What Home and the project list say about a project: how many sessions,
        whether one is live, when the last ran, what waits on a person."""
        sessions = _member_sessions(root_id)
        active = _active_session_id()
        out = {
            "sessions": len(sessions),
            "live": any(s.get("session_id") == active for s in sessions),
            "last_run_at": max((s.get("last_active") or "" for s in sessions), default="") or None,
            "proposed_items": 0,
            "proposed_findings": 0,
        }
        try:
            out["proposed_items"] = int(cs.get_plan_status(root_id).get("proposed", 0))
        except Exception:
            pass
        nb = getattr(cs, "notebook", None)
        if nb is not None:
            try:
                out["proposed_findings"] = len(
                    nb.query_notes(
                        kind=NoteKind.FINDING, status=NoteStatus.PROPOSED, project=root_id
                    )
                )
            except Exception:
                pass
        return out

    def _root(cs, campaign) -> str:
        return cs.root_campaign_id(campaign.id) or campaign.id

    @router.get("/api/campaigns")
    async def list_campaigns():
        """List all campaigns as a tree."""
        cs = _get_store()
        roots = cs.get_root_campaigns()
        trees = []
        for root in roots:
            tree = _build_campaign_tree(cs, root.id)
            tree["project"] = _project_summary(cs, root.id)
            trees.append(tree)
        return {"campaigns": trees}

    # ------------------------------------------------------------------
    # Projects: a root campaign is a project. These are the writes a person
    # makes from the Projects tab; the agent's tools call the same store
    # methods, so both announce (PLAN_UPDATED) and both land in the history.
    # ------------------------------------------------------------------

    @router.post("/api/campaigns", dependencies=[Depends(require_control)])
    async def create_campaign(request: Request):
        """A new project (no ``parent_id``) or a phase of one (``parent_id``)."""
        body = await request.json()
        cs = _get_store()
        description = (body.get("description") or "").strip()
        if not description:
            raise HTTPException(status_code=400, detail="description required")
        parent_id = body.get("parent_id") or None
        if parent_id:
            _resolve(cs, parent_id)
        cid = cs.create_campaign(
            description=description,
            shorthand=(body.get("shorthand") or None),
            target=(body.get("target") or None),
            parent_id=parent_id,
            strains=body.get("strains") or None,
            references=body.get("references") or None,
        )
        return {"ok": True, "campaign": _serialize(cs.get_campaign(cid))}

    @router.patch("/api/campaigns/{campaign_id}", dependencies=[Depends(require_control)])
    async def update_campaign(campaign_id: str, request: Request):
        body = await request.json()
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        fields = {
            k: body.get(k) for k in ("description", "shorthand", "summary", "target", "status")
        }
        if all(v is None for v in fields.values()):
            raise HTTPException(status_code=400, detail="No editable fields supplied")
        try:
            cs.update_campaign(campaign.id, **{k: v for k, v in fields.items() if v is not None})
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        cs._notify_plan_change(campaign.id)
        return {"ok": True, "campaign": _serialize(cs.get_campaign(campaign.id))}

    @router.post("/api/campaigns/{campaign_id}/references", dependencies=[Depends(require_control)])
    async def add_reference(campaign_id: str, request: Request):
        """Keep a search result on the project: {source, key, title, url,
        subtitle}; ``as_strain`` files it with the strains."""
        body = await request.json()
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        if not (body.get("key") or body.get("title")):
            raise HTTPException(status_code=400, detail="key or title required")
        as_strain = bool(body.get("as_strain")) or body.get("files_as") == "strain"
        from gently.ui.web.auth import current_username

        body["added_by"] = current_username(request) or body.get("added_by") or "human"
        entry = cs.add_campaign_reference(_root(cs, campaign), body, as_strain=as_strain)
        return {"ok": True, "entry": entry, "as_strain": as_strain}

    @router.delete(
        "/api/campaigns/{campaign_id}/references", dependencies=[Depends(require_control)]
    )
    async def remove_reference(campaign_id: str, source: str, key: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        return {
            "ok": True,
            "removed": cs.remove_campaign_reference(_root(cs, campaign), source, key),
        }

    @router.post("/api/campaigns/{campaign_id}/items", dependencies=[Depends(require_control)])
    async def create_item(campaign_id: str, request: Request):
        """A plan item by hand (``origin: manual``) in the campaign or one of its
        phases (``phase_id``). ``status`` may be ``proposed`` for a suggestion."""
        body = await request.json()
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        target = campaign.id
        if body.get("phase_id"):
            target = _resolve(cs, body["phase_id"]).id
        title = (body.get("title") or "").strip()
        item_type = body.get("type") or "imaging"
        if not title:
            raise HTTPException(status_code=400, detail="title required")
        spec = body.get("spec") if isinstance(body.get("spec"), dict) else None
        if spec is None and item_type == "imaging":
            # The sentence states some of the spec outright; take those fields
            # now, with their provenance, and leave the rest for the person or
            # the model (POST …/transcribe).
            from gently.harness.memory.intent_spec import merge_spec, parse_intent

            read = parse_intent(f"{title}. {body.get('description') or ''}")
            spec = merge_spec(None, read) if read["spec"] else {"provenance": {}}
            if body.get("draft", True) is not False:
                from gently.harness.memory.intent_spec import draft_record

                spec["draft"] = draft_record("drafting", sentence=title)
        try:
            item_id = cs.create_plan_item(
                campaign_id=target,
                type=item_type,
                title=title,
                description=body.get("description") or None,
                spec=spec,
                depends_on=body.get("depends_on") or None,
                estimated_days=body.get("estimated_days"),
                status=body.get("status") or "planned",
                origin=body.get("origin") or "manual",
            )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        if spec is not None and (spec.get("draft") or {}).get("status") == "drafting":
            asyncio.create_task(_draft_in_background(cs, item_id))
        return {"ok": True, "item": _serialize(cs.get_plan_item(item_id))}

    @router.delete(
        "/api/campaigns/{campaign_id}/items/{item_id}", dependencies=[Depends(require_control)]
    )
    async def delete_item(campaign_id: str, item_id: str):
        cs = _get_store()
        _resolve(cs, campaign_id)
        if not cs.get_plan_item(item_id):
            raise HTTPException(status_code=404, detail="Plan item not found")
        return {"ok": True, "deleted": bool(cs.delete_plan_item(item_id))}

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/move", dependencies=[Depends(require_control)]
    )
    async def move_item(campaign_id: str, item_id: str, request: Request):
        """Move an item to another phase (``campaign_id``) and/or position (``phase_order``)."""
        body = await request.json()
        cs = _get_store()
        _resolve(cs, campaign_id)
        if not cs.get_plan_item(item_id):
            raise HTTPException(status_code=404, detail="Plan item not found")
        to = body.get("campaign_id")
        if to:
            to = _resolve(cs, to).id
        order = body.get("phase_order")
        cs.update_plan_item(
            item_id,
            campaign_id=to or None,
            phase_order=int(order) if order is not None else None,
        )
        return {"ok": True, "item": _serialize(cs.get_plan_item(item_id))}

    def _reaches(cs, start: str, target: str, limit: int = 200) -> bool:
        """Does following depends_on from ``start`` arrive at ``target``?"""
        seen: set[str] = set()
        stack = [start]
        while stack and len(seen) < limit:
            cur = stack.pop()
            if cur == target:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            it = cs.get_plan_item(cur)
            if it:
                stack.extend(it.depends_on or [])
        return False

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/depends",
        dependencies=[Depends(require_control)],
    )
    async def set_dependency(campaign_id: str, item_id: str, request: Request):
        body = await request.json()
        cs = _get_store()
        _resolve(cs, campaign_id)
        dep = body.get("depends_on")
        if not dep or not cs.get_plan_item(item_id) or not cs.get_plan_item(dep):
            raise HTTPException(status_code=404, detail="Plan item not found")
        if body.get("remove"):
            cs.remove_plan_item_dependency(item_id, dep)
        else:
            if dep == item_id or _reaches(cs, dep, item_id):
                raise HTTPException(
                    status_code=400,
                    detail="That would make a cycle: the other item already waits on this one.",
                )
            cs.add_plan_item_dependency(item_id, dep)
        return {"ok": True, "item": _serialize(cs.get_plan_item(item_id))}

    @router.get("/api/plan/intent")
    async def read_intent(text: str = ""):
        """What a sentence states of an imaging spec, read deterministically,
        with the words each field came from. For the form's live preview."""
        from gently.harness.memory.intent_spec import parse_intent

        return parse_intent(text or "")

    async def _model_client():
        """The Anthropic client to draft with, or None when there is no key."""
        client = getattr(server, "claude_async", None)
        if client is not None:
            return client
        try:
            import anthropic

            return anthropic.AsyncAnthropic()
        except Exception:
            return None

    async def draft_item(
        cs,
        item_id: str,
        *,
        note: str | None = None,
        overwrite: bool = False,
        use_model: bool = True,
    ) -> dict:
        """Read the item's sentence into its spec, then ask Gently to draft the
        rest. Writes the merged spec and the draft record; returns a summary."""
        from dataclasses import asdict

        from gently.harness.memory.intent_spec import (
            draft_record,
            draft_spec_with_model,
            merge_spec,
            parse_intent,
            project_context,
        )
        from gently.settings import settings

        item = cs.get_plan_item(item_id)
        if not item:
            raise KeyError(item_id)
        sentence = (
            item.description
            if item.imaging_spec and item.imaging_spec.draft and item.description
            else item.title
        )
        text = (
            f"{sentence}. {item.description or ''}".strip() if sentence == item.title else sentence
        )
        read = parse_intent(text + (f". {note}" if note else ""))
        layers = [read]
        existing = asdict(item.imaging_spec) if item.imaging_spec else {}
        rationale = None
        assumptions: list[str] = []
        title = None
        status = "drafted"
        client = await _model_client() if use_model else None
        if use_model and client is None:
            status = "unavailable"
            assumptions = ["no model to ask: ANTHROPIC_API_KEY is not set in this run"]
        elif use_model:
            try:
                drafted = await draft_spec_with_model(
                    client,
                    settings.models.main,
                    text,
                    known=read["spec"],
                    context=project_context(cs, item.campaign_id),
                    note=note,
                )
                layers.append(drafted)
                rationale = drafted.get("rationale")
                assumptions = drafted.get("assumptions") or []
                title = drafted.get("title")
            except Exception as err:  # noqa: BLE001 — the model being unreachable is an answer
                status = "unavailable"
                assumptions = [f"the model could not be asked: {str(err)[:140]}"]
        merged = merge_spec(existing, *layers, overwrite=overwrite)
        if isinstance(merged.get("acquisition"), dict) and read["spec"]:
            # What the sentence said outright, in the pane's shape, over the draft.
            merged["acquisition"].update(acquisition_from_legacy(read["spec"]))
        merged["draft"] = draft_record(
            status, rationale=rationale, assumptions=assumptions, sentence=sentence
        )
        kwargs: dict = {"spec": merged}
        if title and title.strip() and title.strip() != item.title:
            # The draft's short title goes on the card; the sentence stays as the description.
            kwargs["title"] = title.strip()
            if not item.description or item.description == sentence:
                kwargs["description"] = sentence
        cs.update_plan_item(item_id=item_id, **kwargs)
        return {
            "ok": True,
            "item": _serialize(cs.get_plan_item(item_id)),
            "status": status,
            "from_sentence": sorted(read["spec"]),
            "filled": sorted(
                k
                for k in merged
                if k not in ("provenance", "tactics", "draft")
                and merged[k] not in (None, "", [], {})
            ),
            "rationale": rationale,
            "assumptions": assumptions,
        }

    async def _draft_in_background(cs, item_id: str) -> None:
        try:
            await draft_item(cs, item_id)
        except Exception:
            logger.debug("background draft failed for %s", item_id, exc_info=True)

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/draft",
        dependencies=[Depends(require_control)],
    )
    async def draft_item_route(campaign_id: str, item_id: str, request: Request):
        """Draft (or redraft) the item's spec from its sentence. Body: ``note`` is
        a constraint the researcher adds; ``model: false`` reads the sentence
        only; ``overwrite`` lets the draft replace typed fields."""
        cs = _get_store()
        _resolve(cs, campaign_id)
        if not cs.get_plan_item(item_id):
            raise HTTPException(status_code=404, detail="Plan item not found")
        body = await request.json() if int(request.headers.get("content-length") or 0) > 0 else {}
        return await draft_item(
            cs,
            item_id,
            note=(body.get("note") or None),
            overwrite=bool(body.get("overwrite")),
            use_model=body.get("model", True) is not False,
        )

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/spec/accept",
        dependencies=[Depends(require_control)],
    )
    async def accept_spec(campaign_id: str, item_id: str):
        """A person accepts Gently's draft: the drafted fields become theirs."""
        from dataclasses import asdict

        from gently.harness.memory.intent_spec import draft_record

        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item or not item.imaging_spec:
            raise HTTPException(status_code=404, detail="Plan item or spec not found")
        spec = asdict(item.imaging_spec)
        prov = dict(spec.get("provenance") or {})
        for field, p in prov.items():
            if p.get("source") == "model":
                prov[field] = {**p, "source": "accepted"}
        spec["provenance"] = prov
        old = spec.get("draft") or {}
        spec["draft"] = draft_record(
            "accepted",
            rationale=old.get("rationale"),
            assumptions=old.get("assumptions"),
            sentence=old.get("sentence"),
        )
        cs.update_plan_item(item_id=item_id, spec=spec)
        return {"ok": True, "item": _serialize(cs.get_plan_item(item_id))}

    @router.get("/api/campaigns/{campaign_id}/ledger")
    async def get_ledger(campaign_id: str):
        """The project's ledger: what is next (unblocked plan items), what the
        agent suggested, what was predicted (expectations), what was measured
        (outcomes and findings), what is open (questions). Every entry is read
        back from the store; expectations and questions belong to the project
        through the session they were made in, or by being made for it."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        root = _root(cs, campaign)
        members = {s.get("session_id") for s in _member_sessions(root)}
        items = cs.get_plan_items(campaign_id=root, include_children=True)
        by_status: dict[str, list] = {}
        for it in items:
            by_status.setdefault(it.status.value, []).append(it)
        next_items = cs.get_unblocked_plan_items(root)
        measured_items = [
            it for it in items if it.status in (PlanItemStatus.COMPLETED, PlanItemStatus.SKIPPED)
        ]
        expectations = cs.list_expectations(project_id=root, session_ids=members)
        questions = cs.list_questions(project_id=root, session_ids=members)
        findings: list[dict] = []
        nb = getattr(cs, "notebook", None)
        if nb is not None:
            seen: set[str] = set()
            pool = nb.query_notes(kind=NoteKind.FINDING, project=root)
            pool += [n for n in nb.query_notes(kind=NoteKind.FINDING) if set(n.sessions) & members]
            for n in pool:
                if n.id in seen or n.status == NoteStatus.SUPERSEDED:
                    continue
                seen.add(n.id)
                findings.append(note_to_dict(n))
            findings.sort(key=lambda d: d.get("created_at") or "", reverse=True)
        return {
            "project_id": root,
            "members": sorted(m for m in members if m),
            "next": [_serialize(i) for i in next_items],
            "in_progress": [_serialize(i) for i in by_status.get("in_progress", [])],
            "proposed": [_serialize(i) for i in by_status.get("proposed", [])],
            "blocked": [
                _serialize(i)
                for i in by_status.get("planned", [])
                if i.id not in {n.id for n in next_items}
            ],
            "predicted": [_serialize(e) for e in expectations],
            "measured": {
                "outcomes": [_serialize(i) for i in measured_items],
                "findings": findings,
            },
            "open": [
                _serialize(q) for q in questions if q.status.value in ("open", "investigating")
            ],
            "answered": [
                _serialize(q) for q in questions if q.status.value not in ("open", "investigating")
            ],
        }

    @router.post("/api/campaigns/{campaign_id}/plan-mode", dependencies=[Depends(require_control)])
    async def plan_with_gently(campaign_id: str):
        """Enter plan mode with this project as the focus, so the chat picks up
        where the plan is. The agent's own items arrive as suggestions."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        root = _root(cs, campaign)
        try:
            cs.set_state("current_focus", root)
        except Exception:
            pass
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        if agent is None or not hasattr(agent, "enter_plan_mode"):
            return {"ok": False, "detail": "agent not running", "project_id": root}
        message = agent.enter_plan_mode()
        return {"ok": True, "project_id": root, "message": message}

    @router.get("/api/campaigns/{campaign_id}")
    async def get_campaign(campaign_id: str):
        """Get a single campaign with its plan items."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)

        items = cs.get_plan_items(campaign_id=campaign.id)
        status = cs.get_plan_status(campaign.id)

        return {
            "campaign": _serialize(campaign),
            "items": [_serialize(item) for item in items],
            "status": {
                "total": status["total"],
                "completed": status["completed"],
                "in_progress": status["in_progress"],
                "planned": status["planned"],
                "skipped": status["skipped"],
                "blocked": status["blocked"],
                "by_type": status["by_type"],
                "next_actions": [_serialize(i) for i in status["next_actions"]],
                "pending_decisions": [_serialize(i) for i in status["pending_decisions"]],
            },
        }

    @router.get("/api/campaigns/{campaign_id}/tree")
    async def get_campaign_tree(campaign_id: str):
        """Get a campaign and all descendants as a tree."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        tree = _build_campaign_tree(cs, campaign.id)
        if not tree:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return tree

    @router.get("/api/campaigns/{campaign_id}/document")
    async def get_campaign_document(campaign_id: str):
        """Full plan as a structured document for the review page."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        tree = _build_campaign_tree(cs, campaign.id)
        if not tree:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Collect all items across the tree and enrich with deps/dependents
        bibliography: list[Any] = []
        ref_index = {}  # dedup by (source, key)

        # Pre-index every item in the tree once. The naive enrichment used to call
        # cs.get_plan_item(...) per dep + per dependent, each one walking the on-disk
        # campaign index — O(items × deps × campaigns) YAML reads per request.
        items_by_id: dict[str, dict] = {}
        dependents_map: dict[str, list[str]] = {}

        def _index(node):
            for it in node.get("items", []):
                items_by_id[it["id"]] = it
                for dep_id in it.get("depends_on") or []:
                    dependents_map.setdefault(dep_id, []).append(it["id"])
            for child in node.get("children", []):
                _index(child)

        _index(tree)

        def _resolve_title(target_id: str) -> str:
            hit = items_by_id.get(target_id)
            if hit is not None:
                return hit.get("title") or target_id[:8]
            # Cross-tree fallback (rare: dependency points outside this campaign tree)
            external = cs.get_plan_item(target_id)
            return external.title if external else target_id[:8]

        def _enrich_tree(node):
            """Walk tree, enrich each item with dependencies/dependents, collect refs."""
            for item in node.get("items", []):
                item_id = item["id"]
                dep_ids = list(item.get("depends_on") or [])
                item["dependencies"] = [{"id": d, "title": _resolve_title(d)} for d in dep_ids]
                dnt_ids = dependents_map.get(item_id, [])
                item["dependents"] = [{"id": d, "title": _resolve_title(d)} for d in dnt_ids]

                # Collect references into bibliography
                for ref in item.get("references") or []:
                    source = ref.get("source", "")
                    key = ref.get("key", ref.get("id", ref.get("title", "")))
                    dedup_key = (source, key)
                    if dedup_key not in ref_index:
                        ref_entry = {**ref, "number": len(bibliography) + 1}
                        bibliography.append(ref_entry)
                        ref_index[dedup_key] = ref_entry
                    item.setdefault("ref_numbers", []).append(ref_index[dedup_key]["number"])

            for child in node.get("children", []):
                _enrich_tree(child)

        _enrich_tree(tree)

        # Overall status
        status = cs.get_plan_status(campaign.id)

        return {
            "document": tree,
            "bibliography": bibliography,
            "status": {
                "total": status["total"],
                "completed": status["completed"],
                "in_progress": status["in_progress"],
                "planned": status["planned"],
                "skipped": status.get("skipped", 0),
                "blocked": status.get("blocked", 0),
                "by_type": status["by_type"],
            },
        }

    @router.get("/api/campaigns/{campaign_id}/versions")
    async def list_versions(campaign_id: str):
        """List plan snapshots for a campaign."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        snapshots = cs.list_plan_snapshots(campaign.id)
        return {"versions": _serialize(snapshots)}

    @router.get("/api/campaigns/{campaign_id}/versions/{version_id}")
    async def get_version(campaign_id: str, version_id: str):
        """Get a single plan snapshot."""
        cs = _get_store()
        _resolve(cs, campaign_id)  # validate campaign exists
        snapshot = cs.get_plan_snapshot(version_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        # Parse snapshot_json if it's a string
        result = _serialize(snapshot)
        if isinstance(result.get("snapshot_json"), str):
            try:
                result["snapshot_json"] = json.loads(result["snapshot_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return {"version": result}

    @router.get("/api/campaigns/{campaign_id}/items/{item_id}")
    async def get_item_detail(campaign_id: str, item_id: str):
        """Detailed single item with resolved deps, dependents, refs, sessions."""
        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        # Dependencies with titles
        dep_ids = cs.get_plan_item_dependencies(item_id)
        dependencies = []
        for did in dep_ids:
            dep = cs.get_plan_item(did)
            dependencies.append(
                {
                    "id": did,
                    "title": dep.title if dep else did[:8],
                    "status": dep.status.value if dep else None,
                }
            )

        # Dependents with titles
        dnt_ids = cs.get_plan_item_dependents(item_id)
        dependents = []
        for did in dnt_ids:
            dnt = cs.get_plan_item(did)
            dependents.append(
                {
                    "id": did,
                    "title": dnt.title if dnt else did[:8],
                    "status": dnt.status.value if dnt else None,
                }
            )

        # Sessions — return only those linked to this specific item (item.session_ids),
        # not all campaign sessions. The frontend uses item.session_ids as the canonical
        # list and this pool as metadata (name, created_at) for display.
        item_sids = set(item.session_ids or [])
        all_sessions = cs.get_sessions_for_campaign(item.campaign_id)
        sessions = [s for s in all_sessions if s.session_id in item_sids]

        return {
            "item": _serialize(item),
            "dependencies": dependencies,
            "dependents": dependents,
            "sessions": [_serialize(s) for s in sessions],
        }

    @router.post("/api/campaigns/{campaign_id}/items/{item_id}/sessions")
    async def link_session_to_item(campaign_id: str, item_id: str, request: Request):
        """Link a session to a plan item (appends) and record it against the campaign."""
        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        body = await request.json()
        session_id = body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        from gently.app.filing import file_session_under
        from gently.ui.web.auth import current_username

        file_session_under(
            _gently_store(), cs, session_id, item.campaign_id, item_id, by=current_username(request)
        )

        # Re-fetch item so session_ids reflects the just-added link; then filter
        # exactly as get_item_detail does — POST and GET return the same scope.
        item = cs.get_plan_item(item_id)
        item_sids = set(item.session_ids or [])
        all_sessions = cs.get_sessions_for_campaign(item.campaign_id)
        sessions = [s for s in all_sessions if s.session_id in item_sids]
        return {"sessions": [_serialize(s) for s in sessions]}

    @router.delete("/api/campaigns/{campaign_id}/items/{item_id}/sessions/{session_id}")
    async def unlink_session_from_item(campaign_id: str, item_id: str, session_id: str):
        """Remove a session link from a plan item. Returns {unlinked: bool}."""
        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        unlinked = cs.unlink_plan_item_session(item_id, session_id)
        return {"unlinked": unlinked}

    @router.get("/api/campaigns/{campaign_id}/planned-sessions")
    async def get_planned_sessions(campaign_id: str):
        """Planned sessions linked to a campaign."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        sessions = cs.get_planned_sessions(campaign_id=campaign.id)
        return {"sessions": [_serialize(s) for s in sessions]}

    # ------------------------------------------------------------------
    # Mesh campaign coordination endpoints (auth required for remote)
    # ------------------------------------------------------------------

    def _make_campaign_auth(required_scope: str):
        """Create a FastAPI dependency that checks auth + scope for campaigns."""

        async def _require(request: Request):
            mesh_svc = getattr(server, "mesh_service", None)
            pairing_mgr = getattr(mesh_svc, "pairing_manager", None) if mesh_svc else None
            _audit = getattr(mesh_svc, "audit_log", None) if mesh_svc else None
            if pairing_mgr is None:
                return  # no pairing manager = open access
            host = request.client.host if request.client else ""
            if host in ("127.0.0.1", "::1", "localhost"):
                return
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                peer_id = pairing_mgr.verify_token(auth[7:])
                if peer_id:
                    scopes = pairing_mgr.get_scopes_for_peer(peer_id)
                    if required_scope not in scopes:
                        if _audit:
                            from gently.mesh.audit import AuditEvent

                            _audit.log(
                                AuditEvent.SCOPE_DENIED,
                                outcome="deny",
                                peer_id=peer_id,
                                ip=host,
                                detail=f"scope={required_scope} path={request.url.path}",
                            )
                        raise HTTPException(
                            status_code=403,
                            detail=f"Missing scope: {required_scope}",
                        )
                    if _audit:
                        from gently.mesh.audit import AuditEvent

                        _audit.log(
                            AuditEvent.AUTH_SUCCESS,
                            outcome="allow",
                            peer_id=peer_id,
                            ip=host,
                        )
                    return
            if _audit:
                from gently.mesh.audit import AuditEvent

                _audit.log(
                    AuditEvent.AUTH_FAILURE,
                    outcome="deny",
                    ip=host,
                    detail=f"path={request.url.path}",
                )
            raise HTTPException(status_code=403, detail="Mesh authentication required")

        return _require

    @router.patch(
        "/api/campaigns/{campaign_id}/items/{item_id}",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def update_item(campaign_id: str, item_id: str, request: Request):
        """Edit plan-item fields and/or imaging-spec fields inline.

        Send only the fields you're changing. Spec edits are *merged* into the
        existing spec, so the UI can PATCH a single field (e.g. laser_power_pct)
        without losing the rest. An empty string clears a spec field to null.
        Persists via update_plan_item, which fires PLAN_UPDATED for live refresh.
        """
        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        kwargs: dict[str, Any] = {}
        for f in ("title", "description", "outcome"):
            if isinstance(body.get(f), str):
                kwargs[f] = body[f]
        if body.get("estimated_days") is not None:
            kwargs["estimated_days"] = body["estimated_days"]

        if body.get("status"):
            try:
                kwargs["status"] = PlanItemStatus(body["status"])
            except ValueError as err:
                raise HTTPException(
                    status_code=400, detail=f"Invalid status: {body['status']}"
                ) from err

        spec_patch = body.get("spec")
        if isinstance(spec_patch, dict):
            current = item.imaging_spec or item.bench_spec
            merged = asdict(current) if current else {}
            prov = dict(merged.get("provenance") or {})
            for k, v in spec_patch.items():
                merged[k] = None if v == "" else v
                prov[k] = {"source": "typed", "text": ""}
            merged["provenance"] = prov
            kwargs["spec"] = merged

        if not kwargs:
            raise HTTPException(status_code=400, detail="No editable fields supplied")

        cs.update_plan_item(item_id=item_id, **kwargs)  # fires PLAN_UPDATED
        updated = cs.get_plan_item(item_id)
        return {"ok": True, "item": _serialize(updated)}

    @router.post(
        "/api/campaigns/{campaign_id}/share",
        dependencies=[Depends(_make_campaign_auth("campaigns:admin"))],
    )
    async def share_campaign(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        cs.share_campaign(campaign.id)
        return {"ok": True}

    @router.post(
        "/api/campaigns/{campaign_id}/unshare",
        dependencies=[Depends(_make_campaign_auth("campaigns:admin"))],
    )
    async def unshare_campaign(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        cs.unshare_campaign(campaign.id)
        return {"ok": True}

    @router.get(
        "/api/campaigns/{campaign_id}/export",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def export_campaign(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        tree = cs._serialize_campaign_tree(campaign.id)
        _enrich_export_with_claims(tree, cs, campaign.id)
        return tree

    @router.post(
        "/api/campaigns/{campaign_id}/join",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def join_campaign(campaign_id: str, request: Request):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        body = await request.json()
        instance_id = body.get("instance_id", "")
        hostname = body.get("hostname", "")
        if not instance_id:
            raise HTTPException(status_code=400, detail="instance_id required")
        cs.add_campaign_participant(campaign.id, instance_id, hostname)
        return {"ok": True}

    @router.get(
        "/api/campaigns/{campaign_id}/participants",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def get_participants(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        participants = cs.get_campaign_participants(campaign.id)
        return {"participants": participants}

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/claim",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def claim_item(campaign_id: str, item_id: str, request: Request):
        cs = _get_store()
        _resolve(cs, campaign_id)
        body = await request.json()
        instance_id = body.get("instance_id", "")
        hostname = body.get("hostname", "")
        if not instance_id:
            raise HTTPException(status_code=400, detail="instance_id required")
        ok = cs.claim_plan_item(item_id, instance_id, hostname)
        if not ok:
            raise HTTPException(status_code=409, detail="Item already claimed by another node")
        return {"ok": True}

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/unclaim",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def unclaim_item(campaign_id: str, item_id: str):
        cs = _get_store()
        _resolve(cs, campaign_id)
        cs.unclaim_plan_item(item_id)
        return {"ok": True}

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/status",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def update_item_status(campaign_id: str, item_id: str, request: Request):
        cs = _get_store()
        _resolve(cs, campaign_id)
        body = await request.json()
        status_str = body.get("status")
        outcome = body.get("outcome")
        if not status_str:
            raise HTTPException(status_code=400, detail="status required")
        try:
            item_status = PlanItemStatus(status_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_str}") from None
        cs.update_plan_item(item_id, status=item_status, outcome=outcome)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_campaign_tree(cs, campaign_id: str) -> dict | None:
        """Recursively build campaign tree with plan items and status."""
        campaign = cs.get_campaign(campaign_id)
        if not campaign:
            return None

        items = cs.get_plan_items(campaign_id=campaign_id)
        status = cs.get_plan_status(campaign_id)
        children = cs.get_subcampaigns(campaign_id)

        return {
            "campaign": _serialize(campaign),
            "items": sorted(
                [_serialize(item) for item in items],
                key=lambda x: x.get("phase_order", 0),
            ),
            "status": {
                "total": status["total"],
                "completed": status["completed"],
                "in_progress": status["in_progress"],
                "planned": status["planned"],
            },
            "children": [_build_campaign_tree(cs, child.id) for child in children],
        }

    def _enrich_export_with_claims(tree: dict, cs, campaign_id: str):
        """Walk a serialized campaign tree and annotate items with IDs and claim info."""
        items = cs.get_plan_items(campaign_id=campaign_id)
        items.sort(key=lambda x: x.phase_order)

        # Match serialized items to real items by index (same ordering)
        for idx, serialized_item in enumerate(tree.get("items", [])):
            if idx < len(items):
                real = items[idx]
                serialized_item["id"] = real.id
                serialized_item["status"] = real.status.value
                serialized_item["claimed_by"] = real.claimed_by
                serialized_item["claimed_by_hostname"] = real.claimed_by_hostname

        # Recurse into children
        children = cs.get_subcampaigns(campaign_id)
        for child_idx, child_tree in enumerate(tree.get("children", [])):
            if child_idx < len(children):
                _enrich_export_with_claims(child_tree, cs, children[child_idx].id)

    return router
