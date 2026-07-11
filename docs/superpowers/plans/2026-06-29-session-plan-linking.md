# Session ↔ plans link/delink (F) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Let a session link to multiple plan items, with link/delink from both the Plans tab and the session view. Source of truth = `PlanItem.session_ids` (reverse-query for a session's plans). Base plans deferred.

**Architecture:** Data-layer `unlink_plan_item_session` + `get_plan_items_for_session` (reverse query); link/delink HTTP endpoints; link/delink UI on the Plans-tab item detail + a "Linked plans" panel on the session/Operations view.

## Global Constraints
- Source of truth for session↔plan-item linkage = `PlanItem.session_ids` (a list; `link_plan_item_session` appends — `file_store.py:1415`). A session's plans = reverse query. Do NOT add a plan_item list to SessionIntent.
- Campaign edge stays `SessionIntent.campaign_ids` (`link/unlink_session_campaign`). Linking a session to a plan item also `link_session_campaign(item.campaign_id)`. Delink (v1) only removes the plan-item edge.
- Endpoints mirror `gently/ui/web/routes/campaigns.py` (the item-detail route at :212, PATCH at :324). UI mirrors `campaigns.js` item-detail rendering (~:789-802 Sessions section).
- Repo/base plans DEFERRED (spec §4) — do not build.
- Git hygiene: stage only your files by explicit path; never `git add -A`.

---

### Task 1: Data layer — unlink + reverse-query
**Files:** Modify `gently/harness/memory/file_store.py` — add `unlink_plan_item_session(item_id, session_id)` (remove session from the item's `session_ids`, clear back-compat `session_id` if equal, persist to the campaign plan, fire `_notify_plan_change`; idempotent if absent) and `get_plan_items_for_session(session_id) -> list[PlanItem]` (iterate `get_active_campaigns()` → `get_plan_items(campaign_id)` → filter `session_id in (item.session_ids or [])`). Test: `tests/test_session_plan_linking_store.py`.
- [ ] Confirm `link_plan_item_session` (:1415) — how it appends to session_ids + persists the plan item to `plan/current.yaml` — and mirror the persistence for unlink. Confirm `get_active_campaigns`/`get_plan_items`.
- [ ] TDD: link a session to 2 items (different campaigns) → `get_plan_items_for_session` returns both; `unlink_plan_item_session` removes it from one (other remains); unlink absent → no-op; back-compat `session_id` cleared when it matched. `pytest tests/test_session_plan_linking_store.py -v`; `pytest -q` clean. Commit `feat(f): unlink_plan_item_session + get_plan_items_for_session`.

### Task 2: Link/delink endpoints
**Files:** Modify `gently/ui/web/routes/campaigns.py` — `POST /api/campaigns/{cid}/items/{item_id}/sessions` (body `{session_id}` → `link_plan_item_session` + `link_session_campaign`; return updated sessions) and `DELETE /api/campaigns/{cid}/items/{item_id}/sessions/{session_id}` (→ `unlink_plan_item_session`); add `GET /api/sessions/{id}/plans` in `gently/ui/web/routes/sessions.py` (→ `get_plan_items_for_session`, return `[{id,title,campaign_id,status}]`). Test: `tests/test_session_plan_linking_routes.py`.
- [ ] Mirror the campaigns route store resolution (`server.context_store`/`gently_store`) + the item-detail route (:212). The session-plans route mirrors `sessions.py` resolution. Graceful (404 item/session; empty list).
- [ ] TDD (TestClient + mock store): POST links (store.link_plan_item_session + link_session_campaign called); DELETE delinks (unlink called); GET returns the session's plans; bad ids handled. `pytest tests/test_session_plan_linking_routes.py -v`; `pytest -q` clean. Commit `feat(f): session↔plan link/delink endpoints`.

### Task 3: Plans-tab link/delink UI
**Files:** Modify `gently/ui/web/static/js/campaigns.js` (the item-detail Sessions section ~:789-802 — add a `[+ link session]` picker (sessions from `/api/sessions`) + a `[delink]` button per session, calling the POST/DELETE endpoints then re-rendering the item) + `gently/ui/web/static/css/campaigns.css` (the control styles). 
- [ ] Render the link control + per-session delink in the item-detail Sessions list; wire to the endpoints; optimistic re-render / refetch the item on success; keep the "No linked sessions" empty state with the link control. `node --check`; build a Chrome-MCP harness (real campaigns.js + a stubbed item-detail + /api/sessions) for the controller to audit. Commit `feat(f): Plans-tab session link/delink controls`.

### Task 4: Session "Linked plans" panel
**Files:** Modify the session/Operations view JS (`gently/ui/web/static/js/experiment-overview.js` or the session view `review.js` — whichever renders the active session header; CONFIRM which surface) — add a "Linked plans" panel listing `/api/sessions/{id}/plans` rows (title · campaign · status · `[delink]`) + a `[+ link to a plan]` picker (plan items from active campaigns via `/api/campaigns` tree); wire to the same endpoints; CSS as needed.
- [ ] Confirm the session-scoped render surface + how it knows the current session_id. Render the panel reading `/api/sessions/{id}/plans`; link/delink hit the same endpoints + refresh; symmetric with Task 3. Backward compat: no linked plans → a tidy empty state, no crash. `node --check`; Chrome-MCP harness audit. Commit `feat(f): session Linked-plans panel (link/delink)`.

## Self-Review
- Data→T1; endpoints→T2; Plans-tab UI→T3; session panel→T4. ✓
- Open confirmations: link_plan_item_session persistence (T1), campaigns/sessions route patterns (T2), the campaigns.js item-detail render seam (T3), the session-scoped render surface + current session_id (T4).
- Type consistency: plan-item linkage via `PlanItem.session_ids`; the endpoints + both UIs hit the same POST/DELETE/GET; a link from one surface appears on the other after refresh.
