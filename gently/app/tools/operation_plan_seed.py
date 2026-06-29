"""
Seed an Operation Plan from the plan item linked to a session.

Called at session start (or lazily on first declare/render) to pre-populate
the plan's goal + planned tactics from the ImagingSpec.tactics outline of
the linked plan item.

Hook point
----------
gently/app/agent.py ~455-463, after ``attach_session_to_plan`` has linked
the session to its campaign.  At that point call::

    from gently.app.tools.operation_plan_seed import seed_operation_plan_from_plan_item
    seed_operation_plan_from_plan_item(agent.context_store, agent.session_id)

Resolution path (multi-hop)
----------------------------
1. ``context_store.get_campaign_ids_for_session(session_id)``
   → list of campaign IDs from the session_intent YAML
2. For each campaign_id →  ``context_store.get_plan_items(campaign_id=cid)``
   → PlanItem list
3. Find the first PlanItem where ``session_id`` is in ``plan_item.session_ids``
   (or ``plan_item.session_id == session_id``) AND
   ``plan_item.imaging_spec.tactics`` is non-empty.
4. ``context_store.get_campaign(plan_item.campaign_id)`` for the goal text.
"""

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_plan_item_with_tactics(context_store, session_id: str):
    """Locate the first plan item linked to *session_id* that has a tactics outline.

    Returns ``(plan_item, campaign)`` or ``(None, None)`` if nothing matches.
    """
    campaign_ids = context_store.get_campaign_ids_for_session(session_id)
    if not campaign_ids:
        return None, None

    for cid in campaign_ids:
        items = context_store.get_plan_items(campaign_id=cid)
        for item in items:
            # Match items that list this session
            session_ids = item.session_ids or ([item.session_id] if item.session_id else [])
            if session_id not in session_ids and item.session_id != session_id:
                continue
            # Must carry an imaging_spec with at least one tactic outline entry
            if not item.imaging_spec or not item.imaging_spec.tactics:
                continue
            campaign = context_store.get_campaign(cid)
            return item, campaign

    return None, None


def _outline_entry_to_tactic(entry: dict, idx: int) -> dict:
    """Convert a lightweight ImagingSpec.tactics outline entry to a full planned tactic."""
    kind = entry.get("kind") or "custom"
    name = entry.get("name") or f"Tactic {idx + 1}"
    tactic_id = f"seed_{uuid.uuid4().hex[:8]}"
    return {
        "id": tactic_id,
        "name": name,
        "kind": kind,
        "state": "planned",
        "target": entry.get("target"),
        "scope": entry.get("scope"),
        "structure": entry.get("structure"),
        "rationale": None,
        "live_bind": [],
        "relations": {},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def seed_operation_plan_from_plan_item(context_store, session_id: str) -> dict | None:
    """Seed the Operation Plan for *session_id* from its linked plan item's outline.

    Idempotency rules
    -----------------
    * **No existing plan** — seed fully; write to store; return plan.
    * **Plan exists with ≥1 active or done tactic** — do NOT overwrite; return ``None``.
    * **Plan exists with only planned tactics** — add any outline entries whose
      *name* is not already present; if nothing new, return ``None``; otherwise
      write the merged plan and return it.

    Parameters
    ----------
    context_store:
        A ``FileContextStore`` instance.
    session_id:
        The active session ID.

    Returns
    -------
    dict | None
        The seeded (or merged) plan dict if something was written, else ``None``.
    """
    item, campaign = _resolve_plan_item_with_tactics(context_store, session_id)
    if item is None:
        return None

    outline: list[dict] = item.imaging_spec.tactics  # type: ignore[union-attr]
    if not outline:
        return None

    # --- Idempotency: check existing plan ---
    existing = context_store.get_operation_plan(session_id)
    if existing:
        existing_tactics: list[dict] = existing.get("tactics") or []
        # Guard: at least one active/done tactic — do not clobber
        has_live = any(t.get("state") in ("active", "done") for t in existing_tactics)
        if has_live:
            logger.debug(
                "seed_operation_plan: session %s already has live tactics — skipping",
                session_id,
            )
            return None

        # Plan exists with only planned tactics — add missing outline entries by name
        existing_names = {t.get("name") for t in existing_tactics}
        new_tactics = list(existing_tactics)
        added = False
        for i, entry in enumerate(outline):
            if entry.get("name") not in existing_names:
                new_tactics.append(_outline_entry_to_tactic(entry, i))
                added = True
        if not added:
            return None

        plan = dict(existing)
        plan["tactics"] = new_tactics
        plan["updated_at"] = datetime.now(timezone.utc).isoformat()
        plan["updated_reason"] = "seeded: added planned tactics from plan-item outline"
        context_store.set_operation_plan(session_id, plan)
        logger.info(
            "seed_operation_plan: added %d planned tactic(s) to existing plan for session %s",
            sum(1 for t in new_tactics if t not in existing_tactics),
            session_id,
        )
        return plan

    # --- No existing plan: seed from scratch ---
    goal = ""
    if campaign:
        goal = campaign.target or campaign.description or ""

    tactics = [_outline_entry_to_tactic(entry, i) for i, entry in enumerate(outline)]

    plan = {
        "session_id": session_id,
        "title": item.title,
        "goal": goal,
        "plan_item_id": item.id,
        "campaign_id": item.campaign_id,
        "tactics": tactics,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_reason": "seeded from plan-item tactical outline",
    }
    context_store.set_operation_plan(session_id, plan)
    logger.info(
        "seed_operation_plan: seeded %d planned tactic(s) for session %s from plan item %s",
        len(tactics),
        session_id,
        item.id,
    )
    return plan
