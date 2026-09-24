"""
Shared fixtures for gently tests.
"""

import pytest

from gently.core.event_bus import EventBus


@pytest.fixture(autouse=True)
def _storage_root_is_never_the_real_one(tmp_path, monkeypatch):
    """No test writes to the operator's data, whatever it calls.

    This was not hypothetical. A route handler gained one line —
    `xy_region.apply(box, ...)`, which persists the XY working region under
    `settings.storage.base_path` — and three tests that POST to that handler
    began writing into `D:\Gently3` on the microscope PC. They had carefully
    redirected the handler's config sidecar to `tmp_path` and had no idea a
    second destination existed. The suite left eighteen entries in the rig's
    real region history, and the last of them would have become the software
    fence at the next boot: a box from a test fixture, quietly bounding a real
    stage.

    The fix cannot be "remember to patch it", because the test that broke this
    was written before the line that broke it. So the root is redirected for
    every test, and a test that genuinely wants the real path has to say so by
    undoing this itself.

    Off Windows it also sidesteps the `D:` trap in the settings default: that
    string is not an absolute path there, so it resolves against the cwd and a
    directory literally named `D:` appears in the repo.
    """
    import dataclasses

    from gently.settings import settings

    root = tmp_path / "storage"
    root.mkdir(parents=True, exist_ok=True)

    # Both settings objects are frozen dataclasses, so this goes around the
    # freeze rather than through it — and puts the original back afterwards.
    # Everything reads `settings.storage.base_path` off this one shared
    # instance, including modules that imported it by name.
    original = settings.storage
    object.__setattr__(settings, "storage", dataclasses.replace(original, base_path=root))
    yield root
    object.__setattr__(settings, "storage", original)


@pytest.fixture
def config_dir(tmp_path):
    """Temporary config directory for mesh/transfer state files."""
    d = tmp_path / "config"
    d.mkdir()
    return d


@pytest.fixture
def store(tmp_path):
    """Fresh GentlyStore for tests."""
    from gently.core.store import GentlyStore

    s = GentlyStore(tmp_path / "gently_test")
    yield s
    s.close()


@pytest.fixture
def context_store(tmp_path):
    """Fresh ContextStore for tests."""
    from gently.harness.memory.store import ContextStore

    cs = ContextStore(tmp_path / "context_test.db")
    yield cs
    cs.close()


@pytest.fixture
def file_store(tmp_path):
    """Fresh FileStore for tests."""
    from gently.core.file_store import FileStore

    s = FileStore(tmp_path / "gently3_test")
    yield s
    s.close()


@pytest.fixture
def file_context_store(tmp_path):
    """Fresh FileContextStore for tests."""
    from gently.harness.memory.file_store import FileContextStore

    cs = FileContextStore(tmp_path / "agent_test")
    yield cs
    cs.close()


@pytest.fixture
def event_bus():
    """Fresh EventBus, isolated from global singleton."""
    return EventBus(history_size=50)
