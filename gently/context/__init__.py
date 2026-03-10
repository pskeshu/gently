"""Backward-compatibility shim — use gently.harness.memory instead."""
from gently.harness.memory import *  # noqa: F401,F403
from gently.harness.memory import ContextStore  # explicit re-export
