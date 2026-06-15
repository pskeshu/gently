"""
StoreProtocol — typing-only interface for members the memory mixins expect
from their host class (ContextStore) and from each other.

IntentionsMixin, PlansMixin, UnderstandingMixin, and MlPipelinesMixin are
combined into ContextStore via multiple inheritance. Each mixin calls
methods/attributes defined either on ContextStore itself (_conn, _tx, _now,
_gen_id) or on one of the sibling mixins (e.g. PlansMixin.get_plan_items
called from IntentionsMixin). Declaring this Protocol as a base lets mypy
see those members without introducing a runtime dependency between mixins.
"""

import sqlite3
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from .model import Campaign, PlanItem


@runtime_checkable
class StoreProtocol(Protocol):
    _conn: sqlite3.Connection

    def _tx(self) -> AbstractContextManager[sqlite3.Connection]: ...

    def _now(self) -> str: ...

    def _gen_id(self) -> str: ...

    def get_state(self, key: str) -> str | None: ...

    def get_campaign(self, campaign_id: str) -> Campaign | None: ...

    def get_root_campaigns(self, status: str | None = "active") -> list[Campaign]: ...

    def get_subcampaigns(self, campaign_id: str) -> list[Campaign]: ...

    def create_campaign(
        self,
        description: str,
        shorthand: str | None = None,
        summary: str | None = None,
        target: str | None = None,
        parent_id: str | None = None,
        campaign_id: str | None = None,
    ) -> str: ...

    def delete_campaign(self, campaign_id: str, cascade: bool = True) -> dict[str, int]: ...

    def update_campaign_progress(self, campaign_id: str, progress: str) -> None: ...

    def get_plan_items(
        self,
        campaign_id: str | None = None,
        status: str | None = None,
        type: str | None = None,
        include_children: bool = False,
    ) -> list[PlanItem]: ...

    def _resolve_campaign_label(self, label: str) -> str | None: ...
