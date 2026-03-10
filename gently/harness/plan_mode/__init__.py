"""
Plan Mode — Experimental design collaborator.

A dedicated mode where the agent acts as a scientific advisor,
helping design complete experimental plans before touching hardware.

Produces structured plans (campaigns, plan items, imaging specs)
that the agent tracks across sessions and auto-configures when
it's time to execute.
"""

from .prompt import build_plan_prompt

__all__ = ["build_plan_prompt"]
