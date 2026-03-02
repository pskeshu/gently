"""
Shared fixtures for gently tests.
"""

import pytest
from pathlib import Path

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
    from gently.store import GentlyStore
    s = GentlyStore(tmp_path / "gently_test")
    yield s
    s.close()


@pytest.fixture
def context_store(tmp_path):
    """Fresh ContextStore for tests."""
    from gently.context.store import ContextStore
    cs = ContextStore(tmp_path / "context_test.db")
    yield cs
    cs.close()


@pytest.fixture
def event_bus():
    """Fresh EventBus, isolated from global singleton."""
    return EventBus(history_size=50)
