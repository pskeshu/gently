"""The Projects page defines the helpers it calls.

"+ add a session → Link it here" failed with "refreshSummaries is not defined":
the commit that took the Ledger's helpers out of campaigns.js took
refreshSummaries with them, and five other places call it. The link was
written; the page just never redrew. The page's own helpers read refresh…,
reload…, load…, render…, schedule…; each one called must be declared there.
"""

from __future__ import annotations

import re
from pathlib import Path

CAMPAIGNS_JS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "campaigns.js"
).read_text(encoding="utf-8")


def test_every_page_helper_the_projects_page_calls_is_declared():
    declared = set(re.findall(r"(?m)^\s*(?:async\s+)?function\s+(\w+)\s*\(", CAMPAIGNS_JS))
    helper = r"(?<![\w.])((?:refresh|reload|load|render|schedule)[A-Z]\w*)\s*\("
    called = set(re.findall(helper, CAMPAIGNS_JS))
    assert "refreshSummaries" in called  # the one that went missing
    assert called <= declared, sorted(called - declared)
