"""
ML Subagent — Autonomous background agent for ML training tasks.

Spawned by the copilot via asyncio.create_task(). Runs its own
conversation loop with Claude, using ML-specific tools.
"""

from .agent import MLSubagent

__all__ = ["MLSubagent"]
