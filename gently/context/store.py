"""Backward-compatibility shim — use gently.harness.memory.store instead."""
from gently.harness.memory.store import *  # noqa: F401,F403
from gently.harness.memory.store import ContextStore  # explicit re-export
