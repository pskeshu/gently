"""Filing a session under a project: the one door for the agent, the routes and the gate.

A session's membership is one record, ``projects:`` in its ``session.yaml``,
a list of links each saying which project, which phase if any, in what role,
who made the link and when. A session informs several questions and informs
them differently, so the record is many-to-many with a role per link. The
project side is derived by reading the sessions. ``project_id`` beside it is
the home, the first link, for readers that take one id. The intent record in
the context store is the agent's own account of what a session was for, not
membership; it is written here too so the two never disagree about which
campaign a session was filed under.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ROLES = ("control", "test", "reference", "pilot")


def file_session_under(
    store,
    cs,
    session_id: str,
    campaign_id: str,
    plan_item_id: str | None = None,
    *,
    move: bool = False,
    role: str | None = None,
    by: str | None = None,
) -> dict[str, Any]:
    """Link ``session_id`` to ``campaign_id``, a project or one of its phases,
    in ``role`` if given. The first link is the session's home; a later link
    to another project is added beside it unless ``move`` makes it the home.
    Returns what happened."""
    if cs is None:
        raise RuntimeError("context store unavailable")
    root = cs.root_campaign_id(campaign_id)
    if root is None:
        raise KeyError(f"campaign {campaign_id} not found")
    if role is not None and role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    phase = campaign_id if campaign_id != root else None
    out: dict[str, Any] = {
        "session_id": session_id,
        "project_id": root,
        "home_set": False,
        "home_kept": None,
        "plan_item_id": plan_item_id,
        "links": [],
    }
    if store is not None:
        before = store.session_links(session_id)
        home = before[0]["id"] if before else None
        make_home = move or home is None
        links = store.link_session(session_id, root, phase=phase, role=role, by=by, home=make_home)
        out["links"] = links
        out["home_set"] = make_home and home != root
        out["home_kept"] = home if (home and home != root and not move) else None
        out["project_id"] = links[0]["id"] if links else root
    cs.link_session_campaign(session_id, campaign_id)
    if plan_item_id:
        cs.link_plan_item_session(plan_item_id, session_id)
    return out


def unfile_session(store, cs, session_id: str, project_id: str | None = None) -> bool:
    """Take a session out of one project, or out of all of them. The intent
    records for that project and its phases go too, so nothing files it back
    on the next start. Item links stay; they say what ran."""
    if store is None:
        return False
    before = store.session_links(session_id)
    gone = [x["id"] for x in before] if project_id is None else [project_id]
    store.unlink_session(session_id, project_id)
    if cs is not None:
        for cid in list(cs.get_campaign_ids_for_session(session_id)):
            try:
                if cs.root_campaign_id(cid) in gone or cid in gone:
                    cs.unlink_session_campaign(session_id, cid)
            except Exception:
                logger.debug("intent unlink failed for %s", session_id, exc_info=True)
    return bool(before)


def backfill_session_projects(store, cs) -> int:
    """Give a session that was never filed the home its old links imply: the
    root of the first campaign in its intent record, else of the first plan
    item that ran it. A session with a ``projects`` record, even an empty
    one, has been filed or unfiled on purpose and is left alone. Returns how
    many were filed."""
    if store is None or cs is None:
        return 0
    filed = 0
    for s in store.list_sessions(unfiled=True):
        sid = s.get("session_id")
        if not sid or "projects" in (store.get_session(sid) or {}):
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
            store.link_session(sid, root, by="backfill", home=True)
            filed += 1
    return filed
