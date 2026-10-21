"""Filing a session under a project: the one door for the agent, the routes and the gate.

A session has one home project, ``session.yaml.project_id``, the id of a root
campaign. Plan items link sessions from any project (``PlanItem.session_ids``),
which is how one session serves a second project without a second home. Every
caller that used to write the session-intent record directly goes through here,
so the home is set in exactly one place.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def file_session_under(
    store,
    cs,
    session_id: str,
    campaign_id: str,
    plan_item_id: str | None = None,
    *,
    move: bool = False,
) -> dict[str, Any]:
    """Link ``session_id`` to ``campaign_id``, a project or one of its phases.

    The session's home project is set when it has none. An existing, different
    home is kept and only the plan item is linked, unless ``move`` is given,
    which re-files the session. Returns what happened.
    """
    if cs is None:
        raise RuntimeError("context store unavailable")
    root = cs.root_campaign_id(campaign_id)
    if root is None:
        raise KeyError(f"campaign {campaign_id} not found")
    out: dict[str, Any] = {
        "session_id": session_id,
        "project_id": root,
        "home_set": False,
        "home_kept": None,
        "plan_item_id": plan_item_id,
    }
    if store is not None:
        info = store.get_session(session_id) or {}
        current = info.get("project_id")
        if not current or (move and current != root):
            store.set_session_project(session_id, root)
            out["home_set"] = True
        elif current != root:
            out["home_kept"] = current
    # The intent record is still what the resolution tools read at startup.
    cs.link_session_campaign(session_id, campaign_id)
    if plan_item_id:
        cs.link_plan_item_session(plan_item_id, session_id)
    return out


def unfile_session(store, cs, session_id: str) -> bool:
    """Take a session out of its project. Item links stay; they say what ran."""
    if store is None:
        return False
    info = store.get_session(session_id) or {}
    current = info.get("project_id")
    store.set_session_project(session_id, None)
    if current and cs is not None:
        try:
            cs.unlink_session_campaign(session_id, current)
        except Exception:
            logger.debug("intent unlink failed for %s", session_id, exc_info=True)
    return bool(current)


def backfill_session_projects(store, cs) -> int:
    """Give every unfiled session the home its plan links imply: the root of the
    first campaign in its intent record, else of the first plan item that ran
    it. Idempotent; sessions with no link stay unfiled. Returns how many."""
    if store is None or cs is None:
        return 0
    filed = 0
    for s in store.list_sessions(unfiled=True):
        sid = s.get("session_id")
        if not sid:
            continue
        root = None
        try:
            for cid in cs.get_campaign_ids_for_session(sid):
                root = cs.root_campaign_id(cid)
                if root:
                    break
            if root is None:
                for item in cs.get_plan_items_for_session(sid):
                    root = cs.root_campaign_id(item.campaign_id)
                    if root:
                        break
        except Exception:
            logger.debug("backfill lookup failed for %s", sid, exc_info=True)
            continue
        if root:
            store.set_session_project(sid, root)
            filed += 1
    return filed
