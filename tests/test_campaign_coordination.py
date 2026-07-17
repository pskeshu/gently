"""
Tests for distributed campaign coordination via mesh.

Runs against a live Gently instance (default http://localhost:8080).
Set GENTLY_URL env var to override.

Usage:
    python tests/test_campaign_coordination.py
    GENTLY_URL=http://192.168.1.50:8080 python tests/test_campaign_coordination.py
"""

import json
import os
import sys
import urllib.error
import urllib.request

if "pytest" in sys.modules and os.environ.get("GENTLY_RUN_LIVE_CAMPAIGN_TESTS") != "1":
    import pytest

    pytest.skip(
        "live campaign coordination script requires a running Gently server",
        allow_module_level=True,
    )

BASE = os.environ.get("GENTLY_URL", "http://localhost:8080")
FAKE_PEER = "test-peer-001"
FAKE_HOST = "test-machine"
CONFLICT_PEER = "other-peer-002"

passed = 0
failed = 0


def req(method, path, body=None):
    """Make an HTTP request, return (status_code, parsed_json)."""
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# ------------------------------------------------------------------
# Find a campaign to test with
# ------------------------------------------------------------------
print("=== Campaign Coordination Tests ===\n")

status, data = req("GET", "/api/campaigns")
campaigns = data.get("campaigns", [])
if not campaigns:
    print("No campaigns found. Create one first (e.g. via /plan).")
    sys.exit(1)

# Use the first root campaign
root = campaigns[0]["campaign"]
cid = root["id"]
shorthand = root.get("shorthand") or cid[:8]
print(f"Using campaign: {shorthand} ({cid})\n")


# ------------------------------------------------------------------
# 1. Share
# ------------------------------------------------------------------
print("--- Share / Unshare ---")

status, data = req("POST", f"/api/campaigns/{cid}/share")
check("POST /share returns ok", status == 200 and data.get("ok"))

status, data = req("GET", "/api/mesh/status")
shared = data.get("shared_campaigns", [])
shared_ids = [c["id"] for c in shared]
check("Mesh status includes shared campaign", cid in shared_ids)

# Verify shared campaign has expected fields
if shared:
    sc: dict = next((c for c in shared if c["id"] == cid), {})
    check("Shared campaign has item_count", "item_count" in sc)
    check("Shared campaign has completed_count", "completed_count" in sc)


# ------------------------------------------------------------------
# 2. Export
# ------------------------------------------------------------------
print("\n--- Export ---")

status, tree = req("GET", f"/api/campaigns/{cid}/export")
check("GET /export returns 200", status == 200)
check(
    "Export has children",
    len(tree.get("children", [])) > 0 or len(tree.get("items", [])) > 0,
)

# Find the first item with an ID (enriched)
first_item_id = None


def find_first_item(node):
    global first_item_id
    for item in node.get("items", []):
        if item.get("id") and first_item_id is None:
            first_item_id = item["id"]
            return
    for child in node.get("children", []):
        find_first_item(child)


find_first_item(tree)
check(
    "Export items are enriched with IDs",
    first_item_id is not None,
    f"id={first_item_id}",
)

# Check enrichment fields
if first_item_id:

    def find_item(node, target_id):
        for item in node.get("items", []):
            if item.get("id") == target_id:
                return item
        for child in node.get("children", []):
            result = find_item(child, target_id)
            if result:
                return result
        return None

    enriched = find_item(tree, first_item_id)
    check("Enriched item has status field", "status" in (enriched or {}))
    check("Enriched item has claimed_by field", "claimed_by" in (enriched or {}))


# ------------------------------------------------------------------
# 3. Join
# ------------------------------------------------------------------
print("\n--- Join ---")

status, data = req(
    "POST",
    f"/api/campaigns/{cid}/join",
    {
        "instance_id": FAKE_PEER,
        "hostname": FAKE_HOST,
    },
)
check("POST /join returns ok", status == 200 and data.get("ok"))

status, data = req("GET", f"/api/campaigns/{cid}/participants")
check("GET /participants returns list", status == 200 and "participants" in data)
peer_ids = [p["instance_id"] for p in data["participants"]]
check("Participant is registered", FAKE_PEER in peer_ids)

# Join is idempotent
status, data = req(
    "POST",
    f"/api/campaigns/{cid}/join",
    {
        "instance_id": FAKE_PEER,
        "hostname": FAKE_HOST,
    },
)
check("Re-join is idempotent", status == 200)

# Missing instance_id
status, data = req("POST", f"/api/campaigns/{cid}/join", {"hostname": "x"})
check("Join without instance_id returns 400", status == 400)


# ------------------------------------------------------------------
# 4. Claim
# ------------------------------------------------------------------
print("\n--- Claim ---")

if first_item_id:
    # Claim the item
    status, data = req(
        "POST",
        f"/api/campaigns/{cid}/items/{first_item_id}/claim",
        {
            "instance_id": FAKE_PEER,
            "hostname": FAKE_HOST,
        },
    )
    check("POST /claim returns ok", status == 200 and data.get("ok"))

    # Re-claim by same peer (idempotent)
    status, data = req(
        "POST",
        f"/api/campaigns/{cid}/items/{first_item_id}/claim",
        {
            "instance_id": FAKE_PEER,
            "hostname": FAKE_HOST,
        },
    )
    check("Re-claim by same peer succeeds", status == 200)

    # Conflicting claim by different peer
    status, data = req(
        "POST",
        f"/api/campaigns/{cid}/items/{first_item_id}/claim",
        {
            "instance_id": CONFLICT_PEER,
            "hostname": "conflict-machine",
        },
    )
    check("Conflicting claim returns 409", status == 409)

    # Verify claim shows in export
    status, tree = req("GET", f"/api/campaigns/{cid}/export")
    claimed_item = find_item(tree, first_item_id)
    check(
        "Claimed item shows claimed_by",
        (claimed_item or {}).get("claimed_by") == FAKE_PEER,
    )
    check(
        "Claimed item shows hostname",
        (claimed_item or {}).get("claimed_by_hostname") == FAKE_HOST,
    )


# ------------------------------------------------------------------
# 5. Status update
# ------------------------------------------------------------------
print("\n--- Status Update ---")

if first_item_id:
    status, data = req(
        "POST",
        f"/api/campaigns/{cid}/items/{first_item_id}/status",
        {
            "status": "in_progress",
        },
    )
    check("POST /status returns ok", status == 200 and data.get("ok"))

    # Verify in export
    status, tree = req("GET", f"/api/campaigns/{cid}/export")
    updated = find_item(tree, first_item_id)
    check(
        "Item status updated to in_progress",
        (updated or {}).get("status") == "in_progress",
    )

    # With outcome
    status, data = req(
        "POST",
        f"/api/campaigns/{cid}/items/{first_item_id}/status",
        {
            "status": "completed",
            "outcome": "Test outcome from coordination test",
        },
    )
    check("POST /status with outcome returns ok", status == 200)

    # Invalid status
    status, data = req(
        "POST",
        f"/api/campaigns/{cid}/items/{first_item_id}/status",
        {
            "status": "bogus_status",
        },
    )
    check("Invalid status returns 400", status == 400)


# ------------------------------------------------------------------
# 6. Unclaim
# ------------------------------------------------------------------
print("\n--- Unclaim ---")

if first_item_id:
    status, data = req("POST", f"/api/campaigns/{cid}/items/{first_item_id}/unclaim")
    check("POST /unclaim returns ok", status == 200 and data.get("ok"))

    # Verify claim cleared in export
    status, tree = req("GET", f"/api/campaigns/{cid}/export")
    unclaimed = find_item(tree, first_item_id)
    check(
        "Unclaimed item has claimed_by=None",
        (unclaimed or {}).get("claimed_by") is None,
    )


# ------------------------------------------------------------------
# 7. Unshare
# ------------------------------------------------------------------
print("\n--- Unshare ---")

status, data = req("POST", f"/api/campaigns/{cid}/unshare")
check("POST /unshare returns ok", status == 200 and data.get("ok"))

status, data = req("GET", "/api/mesh/status")
shared = data.get("shared_campaigns", [])
shared_ids = [c["id"] for c in shared]
check("Campaign removed from mesh status", cid not in shared_ids)


# ------------------------------------------------------------------
# 8. Reset item to planned (cleanup)
# ------------------------------------------------------------------
print("\n--- Cleanup ---")

if first_item_id:
    req(
        "POST",
        f"/api/campaigns/{cid}/items/{first_item_id}/status",
        {
            "status": "planned",
            "outcome": None,
        },
    )
    # Remove participant
    # (no delete endpoint, but that's fine — participants table is cleaned up on campaign delete)
    print(f"  Reset item {first_item_id[:8]} to planned")


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
total = passed + failed
print(f"\n{'=' * 40}")
print(f"  {passed}/{total} passed, {failed} failed")
print(f"{'=' * 40}")

sys.exit(1 if failed else 0)
