"""
Shared fixtures for gently tests.
"""

import pytest

from gently.core.event_bus import EventBus


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
