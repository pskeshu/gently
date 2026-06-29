# Design: Session ↔ plans link/delink (sub-project F)

Status: design 2026-06-29 (after recon + user steering). Branch `feature/session-plan-linking` (off D2).
Lets a session link to MULTIPLE plan items, with link/delink from BOTH the Plans tab and the session
view. "Repo/base plans" is DEFERRED (user has ideas — separate follow-on).

## 0. What exists (recon)
- Session↔**campaign**: many-to-many (`SessionIntent.campaign_ids` list; `link/unlink_session_campaign`).
- Session↔**plan-item**: `PlanItem.session_ids` is a LIST (a session can already appear under multiple
  items in storage) — but a session's *own* notion of "its plans" is a single `active_plan_item_id`
  pointer, and `SessionIntent` stores no plan-item ids.
- `attach_session_to_plan` appends to `item.session_ids` (via `link_plan_item_session`) but overwrites
  the single active pointer; `detach_session_from_plan` only clears the pointer (NOT a data delink).
- Plans tab (`campaigns.js:789-802`) shows a per-item read-only Sessions list ("No linked sessions").
- NO link/delink endpoint or UI anywhere; NO `unlink_plan_item_session`; session endpoint returns no linkage.

## 1. The model — source of truth = `PlanItem.session_ids`
A session's linked plan items = the reverse query over plan items whose `session_ids` includes the
session. No new field on SessionIntent (avoids dual source of truth). Multi-plan falls out naturally
(a session can be in many items' `session_ids`). The campaign edge stays on `SessionIntent.campaign_ids`.

- **Link** session↔plan-item: `link_plan_item_session(item_id, session_id)` (exists, appends) +
  `link_session_campaign(session_id, item.campaign_id)` (exists).
- **Delink**: NEW `unlink_plan_item_session(item_id, session_id)` — remove the session from
  `item.session_ids` (+ clear the back-compat `session_id` if it pointed there); fire `_notify_plan_change`.
  Campaign edge: leave it unless no other item of that campaign links the session (refinement — for v1,
  delink only touches the plan-item edge; campaign delink stays the existing separate control).
- **Session's plans**: NEW `get_plan_items_for_session(session_id) -> list[PlanItem]` (reverse query
  across `get_active_campaigns` → `get_plan_items` → filter `session_id in item.session_ids`).

## 2. Endpoints (mirror `routes/campaigns.py`)
- `POST /api/campaigns/{cid}/items/{item_id}/sessions` body `{session_id}` → link (link_plan_item_session
  + link_session_campaign). Returns the updated item sessions.
- `DELETE /api/campaigns/{cid}/items/{item_id}/sessions/{session_id}` → delink (unlink_plan_item_session).
- `GET /api/sessions/{id}/plans` → the session's linked plan items (id, title, campaign_id, status) via
  `get_plan_items_for_session`. (A new sub-route; leaves the existing session payload untouched.)

## 3. UI — both surfaces
### 3.1 Plans tab item-detail (`campaigns.js` ~:789-802)
The existing per-item Sessions list gains: a **[+ link session]** control (a picker of recent sessions
from `/api/sessions`) and a **[delink]** button per listed session. Calls the POST/DELETE endpoints,
re-renders the item detail. Empty state keeps "No linked sessions" + the link control.

### 3.2 Session / Operations view — "Linked plans" panel
A panel (in the Operations/experiment view header or a session detail strip) listing the session's
linked plan items (from `/api/sessions/{id}/plans`): each row `plan item title · campaign · status ·
[delink]`, plus **[+ link to a plan]** (a picker of plan items from the active campaigns). Symmetric
with 3.1 — link/delink from either side; both hit the same endpoints + refresh.

## 4. Out of scope (deferred)
- **Repo/base plans** — the user has ideas; a separate follow-on (a repo plans library / seed). Noted,
  not built here.
- Campaign-edge auto-cleanup on plan delink (v1 leaves the campaign link; refine later).
- Reworking `attach_session_to_plan`/`detach` agent tools beyond what's needed — the data-layer
  delink (`unlink_plan_item_session`) is added; wiring a `detach` that calls it is a small optional add.

## 5. Testing
- Data layer: `unlink_plan_item_session` removes the session (+ back-compat session_id); idempotent on
  absent; `get_plan_items_for_session` reverse-query returns the right items across campaigns; multi-plan
  (a session under 2 items) round-trips.
- Endpoints: POST links (item.session_ids gains it + campaign linked); DELETE delinks; GET returns the
  session's plans; mirror `tests/test_*route*` with a mock store.
- UI: link/delink controls on both surfaces (node --check + Chrome audit of the Plans-tab item detail +
  the session "Linked plans" panel); link from one side shows on the other after refresh.
