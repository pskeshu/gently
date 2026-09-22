"""The SAM checkpoint is found from a stable place, not from the cwd.

Detect failed on the rig with:

    Detect failed (502 — SAM checkpoint not found: sam_vit_b_01ec64.pth)

The checkpoint was a bare filename, so it resolved against the **device
layer process's cwd** — whatever directory the launcher happened to start it
from. The file is a 375 MB untracked blob that exists in exactly one checkout,
so the moment the device layer is spawned from anywhere else (a worktree, the
Tauri shell, a service), detection dies at the point of use. And the message
named a file but no directory, so it did not even say where to put it.

Two properties, both of which the old code failed:

* the search does not depend on the cwd alone; and
* when it fails it says every place it looked.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gently.hardware.dispim.sam_detection import (
    CHECKPOINT_NAME,
    checkpoint_missing_detail,
    find_checkpoint,
)


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A cwd with no checkpoint in it — the rig's failing situation."""
    empty = tmp_path / "some-worktree"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.delenv("GENTLY_SAM_CHECKPOINT", raising=False)
    return empty


def test_an_explicit_env_path_wins(elsewhere, tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "models" / CHECKPOINT_NAME
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not really a checkpoint")
    monkeypatch.setenv("GENTLY_SAM_CHECKPOINT", str(ckpt))
    found, _ = find_checkpoint(None)
    assert found == ckpt


def test_the_storage_root_is_searched(elsewhere, tmp_path, monkeypatch) -> None:
    """Machine-level and checkout-independent — where a shared blob belongs."""
    import gently.hardware.dispim.sam_detection as mod

    root = tmp_path / "Gently3"
    ckpt = root / "models" / CHECKPOINT_NAME
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    # settings is a frozen dataclass; swap the module's reference to it.
    monkeypatch.setattr(mod, "settings", SimpleNamespace(storage=SimpleNamespace(base_path=root)))
    found, searched = find_checkpoint(None)
    assert found == ckpt, f"storage root not searched; looked in {searched}"
    assert searched.index(ckpt) < searched.index(elsewhere / CHECKPOINT_NAME), (
        "the cwd is consulted before the storage root again"
    )


def test_the_repo_root_still_works(elsewhere) -> None:
    """A normal clone keeps the checkpoint beside pyproject.toml."""
    repo_root = Path(find_checkpoint.__globals__["__file__"]).resolve().parents[3]
    _, searched = find_checkpoint(None)
    assert repo_root / CHECKPOINT_NAME in searched, (
        "the repo root is no longer searched; an existing clone would stop working"
    )


def test_the_cwd_is_still_searched_last(elsewhere) -> None:
    """The old behaviour keeps working — it is just no longer the only one.

    Only the ORDER is asserted: this box has a checkpoint at the repo root, so
    which candidate wins depends on the machine, but where the cwd sits in the
    list does not.
    """
    ckpt = elsewhere / CHECKPOINT_NAME
    ckpt.write_bytes(b"x")
    _, searched = find_checkpoint(None)
    assert ckpt in searched, "the cwd is no longer searched at all"
    assert searched[-1] == ckpt, "the cwd should be the last resort, not the first"


def test_an_explicit_path_is_honoured(elsewhere, tmp_path) -> None:
    ckpt = tmp_path / "custom" / "sam.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"x")
    found, _ = find_checkpoint(str(ckpt))
    assert found == ckpt


def test_a_failure_names_every_directory_it_tried(elsewhere) -> None:
    # Not asserted to be missing: this box happens to have one at the repo
    # root. What matters is that the message accounts for every candidate.
    _, searched = find_checkpoint(None)
    detail = checkpoint_missing_detail(searched)
    assert len(searched) >= 3, "the search collapsed to a single location again"
    for path in searched:
        assert str(path) in detail, (
            "the error does not name where it looked, which is the whole reason "
            "the original failure was unactionable"
        )


def test_the_device_layer_does_not_pin_a_cwd_relative_name() -> None:
    """The regression that caused the 502, pinned in the device layer itself."""
    src = (
        Path(__file__).resolve().parents[1] / "gently" / "hardware" / "dispim" / "device_layer.py"
    ).read_text(encoding="utf-8")
    assert f'self._sam_checkpoint = "{CHECKPOINT_NAME}"' not in src, (
        "the device layer pins the bare checkpoint filename again — it will "
        "resolve against whatever directory the process was started in"
    )


def test_no_env_leaks_between_tests() -> None:
    assert os.getenv("GENTLY_SAM_CHECKPOINT") is None
