"""A project's question and short name can be changed after it is made.

"i also noticed that it is not possible to edit the name of the project once
set, and also the short name is not editable or there is no way to edit it."
The route existed: PATCH /api/campaigns/{id} takes description and shorthand.
The header drew both as plain text and never called it.
"""

from __future__ import annotations

import re
from pathlib import Path

from gently.harness.memory.file_store import FileContextStore

CAMPAIGNS_JS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "campaigns.js"
).read_text(encoding="utf-8")


def test_the_header_offers_an_edit_that_saves_both_fields_to_the_project():
    assert 'data-action="head-edit"' in CAMPAIGNS_JS
    save = re.search(r"case 'head-save': \{(.*?)\n {12}\}", CAMPAIGNS_JS, re.S)
    assert save is not None
    assert "api('PATCH', `/api/campaigns/${P}`" in save.group(1)
    assert "description: d.description" in save.group(1)
    assert "shorthand: d.shorthand" in save.group(1)


def test_an_empty_short_name_clears_it(tmp_path):
    cs = FileContextStore(tmp_path / "agent")
    pid = cs.create_campaign("Does LED dose change time to hatching?", shorthand="dose")
    cs.update_campaign(pid, description="Does LED dose change hatching time?", shorthand="")
    c = cs.get_campaign(pid)
    assert c.description == "Does LED dose change hatching time?"
    assert not c.shorthand


def test_the_folder_on_disk_follows_the_name(tmp_path):
    cs = FileContextStore(tmp_path / "agent")
    pid = cs.create_campaign("Does LED dose change time to hatching?", shorthand="dose")
    before = cs.campaign_dir(pid)
    assert before.name == f"{pid}_dose"
    (before / "reading.md").write_text("KNOWN. Nothing yet.", encoding="utf-8")
    cs.update_campaign(pid, shorthand="led-dose")
    after = cs.campaign_dir(pid)
    assert after.name == f"{pid}_led-dose" and after.is_dir() and not before.exists()
    assert (after / "reading.md").read_text(encoding="utf-8") == "KNOWN. Nothing yet."
    assert cs.get_campaign(pid).shorthand == "led-dose"
    # a fresh store finds it by the id in its yaml, under the new name
    assert FileContextStore(tmp_path / "agent").campaign_dir(pid) == after
    # no short name: the question names the folder
    cs.update_campaign(pid, shorthand="")
    assert cs.campaign_dir(pid).name == f"{pid}_does-led-dose-change-time-to-h"
