"""The store does not read the same unchanged file twice, nor walk twice.

From the Sessions tab audit (2026-10-30): one list call over 300 sessions
parsed 900 YAML files in pure Python, two of them the same session.yaml, and
listed each projections folder twice. libyaml is installed; a file that has
not changed is remembered by its mtime and size; one scan gives the count and
the newest frame.
"""

from __future__ import annotations

import builtins
import os

import yaml

from gently.core import file_store
from gently.core.file_store import FileStore, _read_yaml, _write_yaml
from gently.core.session_summary import what_a_session_holds


def _count_opens(monkeypatch, suffix):
    seen = {"n": 0}
    real = builtins.open

    def counting(file, *a, **k):
        if str(file).endswith(suffix):
            seen["n"] += 1
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", counting)
    return seen


def test_libyaml_is_the_loader_when_installed():
    if yaml.__with_libyaml__:
        assert file_store._LOADER is yaml.CSafeLoader


def test_an_unchanged_file_is_parsed_once_and_callers_get_their_own_copy(tmp_path, monkeypatch):
    path = tmp_path / "session.yaml"
    _write_yaml(path, {"name": "monday", "tags": ["a"]})
    opens = _count_opens(monkeypatch, "session.yaml")
    first = _read_yaml(path)
    assert first == {"name": "monday", "tags": ["a"]} and opens["n"] == 1
    first["tags"].append("mutated by a caller")
    second = _read_yaml(path)
    assert second == {"name": "monday", "tags": ["a"]} and opens["n"] == 1
    # a rewrite is a new mtime or size: read again
    _write_yaml(path, {"name": "tuesday", "tags": ["a", "b"]})
    assert _read_yaml(path)["name"] == "tuesday" and opens["n"] == 2
    assert _read_yaml(tmp_path / "missing.yaml") is None


def test_what_a_session_holds_scans_each_projection_folder_once(tmp_path, monkeypatch):
    store = FileStore(tmp_path)
    store.create_session("s1", name="monday")
    for eid in ("embryo_1", "embryo_2"):
        store.register_embryo("s1", eid, position_x=0, position_y=0)
        pdir = store._embryo_dir("s1", eid) / "projections"
        pdir.mkdir(parents=True, exist_ok=True)
        for tp in (1, 2, 3):
            (pdir / f"t{tp:04d}.jpg").write_bytes(b"jpg")
    scans = {"n": 0}
    real = os.scandir

    def counting(path=".", *a, **k):
        if str(path).endswith("projections"):
            scans["n"] += 1
        return real(path, *a, **k)

    monkeypatch.setattr(os, "scandir", counting)
    held = what_a_session_holds(store, "s1")
    assert held["embryo_count"] == 2 and held["timepoints"] == 6
    assert held["last_image_at"] is not None
    assert scans["n"] == 2
    assert store.list_projection_timepoints("s1", "embryo_1") == [1, 2, 3]
    assert store.projection_timepoints_and_latest("s1", "nobody") == ([], None)


def test_a_symlink_planted_as_an_embryo_name_is_refused(tmp_path):
    store = FileStore(tmp_path)
    store.create_session("s1")
    outside = tmp_path / "outside"
    outside.mkdir()
    sd = store._session_dir("s1")
    (sd / "embryos").mkdir(exist_ok=True)
    try:
        os.symlink(outside, sd / "embryos" / "evil")
    except (OSError, NotImplementedError):
        return  # no symlinks here (Windows without privilege)
    try:
        store._embryo_dir("s1", "evil")
    except ValueError as e:
        assert "escapes" in str(e)
    else:
        raise AssertionError("a symlinked embryo dir must be refused")
