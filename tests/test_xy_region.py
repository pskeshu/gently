"""The XY working region, and the list of the ones it replaced.

The region used to be persisted into `config.local.yml`, resolved from a
`config_path` whose default is the RELATIVE `config/config.yml` — so it lived
wherever the device layer happened to be started from. On the rig there turned
out to be no such file anywhere, meaning boot had been writing full travel
every time and nobody was told.

That is survivable for one number someone can re-walk. It is not survivable for
a history, which is the whole point of being able to go back. So these tests
pin the two properties that make going back trustworthy: the record lives in
the storage root, and applying never drops what it replaced.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from gently.core import xy_region

BOX = {"x_min": -900.0, "x_max": 400.0, "y_min": -800.0, "y_max": 100.0}
WIDER = {"x_min": -1200.0, "x_max": 600.0, "y_min": -900.0, "y_max": 200.0}


@pytest.fixture(autouse=True)
def _storage(tmp_path, monkeypatch):
    """Every test gets its own storage root; none of them touch a real one."""
    monkeypatch.setattr(
        xy_region, "settings", SimpleNamespace(storage=SimpleNamespace(base_path=tmp_path))
    )
    return tmp_path


def _at(minutes: int) -> datetime:
    return datetime(2026, 9, 24, 10, 0) + timedelta(minutes=minutes)


def test_the_region_lives_in_the_storage_root_not_beside_the_cwd():
    path = xy_region.region_path()
    assert path.is_absolute()
    assert path == xy_region.settings.storage.base_path / "config" / "xy_region.yaml"


def test_no_file_means_no_region_rather_than_a_guess():
    record = xy_region.load()
    assert record.current is None
    assert record.history == []


def test_applying_records_the_box_and_when():
    record = xy_region.apply(BOX, note="walked the corners", now=_at(0))
    assert record.current is not None
    assert record.current.box == BOX
    assert record.current.applied_at == "2026-09-24T10:00:00"
    assert record.current.note == "walked the corners"
    assert xy_region.load().current.box == BOX, "and it survives the process"


def test_applying_keeps_the_region_it_replaced():
    xy_region.apply(BOX, now=_at(0))
    record = xy_region.apply(WIDER, now=_at(5))
    assert record.current.box == WIDER
    assert [h.box for h in record.history] == [BOX]


def test_restoring_brings_a_past_region_back_as_the_current_one():
    xy_region.apply(BOX, now=_at(0))
    xy_region.apply(WIDER, now=_at(5))
    record = xy_region.restore("2026-09-24T10:00:00")
    assert record is not None
    assert record.current.box == BOX
    assert "restored from" in record.current.note


def test_a_restore_can_itself_be_undone():
    # Going back is an apply like any other, so the region it replaced joins
    # the history and you can go forward again. "Back and forth", not "back".
    xy_region.apply(BOX, now=_at(0))
    xy_region.apply(WIDER, now=_at(5))
    xy_region.restore("2026-09-24T10:00:00")
    assert [h.box for h in xy_region.load().history] == [BOX, WIDER]


def test_restoring_something_that_is_not_there_says_so():
    xy_region.apply(BOX, now=_at(0))
    assert xy_region.restore("1999-01-01T00:00:00") is None
    assert xy_region.load().current.box == BOX, "and changes nothing"


def test_the_history_is_capped_but_the_current_region_never_is():
    for i in range(xy_region.MAX_HISTORY + 10):
        xy_region.apply({**BOX, "x_max": float(i)}, now=_at(i))
    record = xy_region.load()
    assert len(record.history) == xy_region.MAX_HISTORY
    assert record.current.x_max == float(xy_region.MAX_HISTORY + 9)


def test_an_unreadable_file_reads_as_no_region():
    # No region means full travel, which is the safe direction. A half-parsed
    # one would be a fence in the wrong place.
    path = xy_region.region_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{{{ not yaml", encoding="utf-8")
    assert xy_region.load().current is None


def test_a_history_entry_missing_a_bound_is_dropped_not_fatal():
    # One bad row should cost you that row, not the list.
    import yaml

    xy_region.apply(BOX, now=_at(0))
    xy_region.apply(WIDER, now=_at(5))
    path = xy_region.region_path()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["history"].append({"x_min": -1.0, "y_min": -1.0, "y_max": 1.0})  # no x_max
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    record = xy_region.load()
    assert record.current.box == WIDER
    assert [h.box for h in record.history] == [BOX], "the good row survives"
