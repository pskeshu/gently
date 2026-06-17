"""
Startup Wizard — conversational context-gap filler.

Runs at startup between the WebSocket ``connected`` message and the
main REPL loop.  Each step presents a choice picker with an option
to type a custom response, so the researcher always has something
to click.

The wizard drives the conversation by calling ``send_fn`` to push
messages to the client, ``wait_for_input`` to collect free-text,
and ``wait_for_choice`` to present pickers.
"""

import logging
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from gently.settings import settings

from .gap_assessment import ContextGapReport, assess_gaps
from .model import Confidence, Learning
from .onboarding import (
    process_onboarding_response,
)

try:
    from .file_store import FileContextStore as ContextStore
except ImportError:
    from .store import ContextStore  # type: ignore[assignment]  # legacy fallback

logger = logging.getLogger(__name__)

SKIP_PHRASES = {"skip", "/skip", "later"}


class StartupWizard:
    """
    Gap-driven startup wizard.

    Parameters
    ----------
    context_store : ContextStore
        The agent's mind database.
    session_id : str
        Current agent session ID.
    claude_client : optional
        Anthropic client for LLM extraction.  May be ``None``
        (falls back to basic keyword extraction).
    """

    def __init__(
        self,
        context_store: ContextStore,
        session_id: str,
        claude_client: Any | None = None,
    ):
        self.context_store = context_store
        self.session_id = session_id
        self.claude_client = claude_client
        self._gap_report: ContextGapReport | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def gap_report(self) -> ContextGapReport:
        if self._gap_report is None:
            self._gap_report = assess_gaps(self.context_store)
        return self._gap_report

    @property
    def needed(self) -> bool:
        """True when the wizard has steps to run."""
        return self.gap_report.conversation_weight != "none"

    @property
    def gap_summary(self) -> dict:
        """Metadata dict for the ``connected`` message."""
        report = self.gap_report
        return {
            "wizard_needed": self.needed,
            "conversation_weight": report.conversation_weight,
            "is_first_launch": report.is_first_launch,
            "readiness": report.readiness,
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        send_fn: Callable[[dict], Coroutine],
        wait_for_input: Callable[[], Coroutine],
        wait_for_choice: Callable[[dict], Coroutine],
    ) -> None:
        """Run the wizard to completion."""
        report = self.gap_report

        # --- First launch: organism + research program ---
        if report.needs_lab_onboarding:
            await self._step_first_launch(send_fn, wait_for_input, wait_for_choice)
            await self._finish(send_fn)
            return

        # --- Returning user: campaign selection ---
        if report.needs_campaign:
            if report.has_campaigns:
                await self._step_campaign_select(send_fn, wait_for_input, wait_for_choice)
            else:
                await self._step_campaign_create(send_fn, wait_for_input, wait_for_choice)

        # --- Planned session matching ---
        todays = self.context_store.get_todays_sessions()
        if todays:
            await self._step_planned_session(todays, send_fn, wait_for_choice)

        await self._finish(send_fn)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    async def _step_first_launch(self, send_fn, wait_for_input, wait_for_choice):
        """First launch: just ask the organism. Research program emerges from conversation."""

        # What organism?
        await self._say(send_fn, "Hi! I'm your microscopy agent. What organism do you work with?")

        organism_choices = {
            "_type": "single",
            "question": "Organism",
            "options": [
                {
                    "id": "elegans",
                    "label": "C. elegans",
                    "description": "Nematode embryos",
                },
                {
                    "id": "zebrafish",
                    "label": "Zebrafish",
                    "description": "Danio rerio",
                },
                {
                    "id": "drosophila",
                    "label": "Drosophila",
                    "description": "Fruit fly",
                },
                {
                    "id": "cell_lines",
                    "label": "Cell lines",
                    "description": "Cultured cells",
                },
            ],
            "allow_multiple": False,
        }
        organism = await wait_for_choice(organism_choices)

        if not organism:
            return  # cancelled — skip onboarding

        # Known organism IDs
        organism_names = {
            "elegans": "C. elegans",
            "zebrafish": "Zebrafish (Danio rerio)",
            "drosophila": "Drosophila melanogaster",
            "cell_lines": "Cell lines",
        }

        if organism in organism_names:
            # Known pick
            organism_label = organism_names[organism]
            self._store_learning(f"Lab organism: {organism_label}")
        elif organism != "__custom__" and not _is_skip(organism):
            # Custom text typed inline in the picker
            organism_label = organism.strip()
            self._store_learning(f"Lab organism: {organism_label}")
        else:
            return  # skip or cancelled

    async def _step_campaign_select(self, send_fn, wait_for_input, wait_for_choice):
        """Show active campaigns in a picker."""
        campaigns = self.context_store.get_active_campaigns()
        options = [
            {"id": c.id, "label": c.display_name, "description": c.target or ""} for c in campaigns
        ]
        options.append(
            {
                "id": "__new__",
                "label": "Start something new",
                "description": "Describe a new research direction",
            }
        )

        await self._say(send_fn, "Welcome back.")

        choice_data = {
            "_type": "single",
            "question": "Which campaign is this session for?",
            "options": options,
            "allow_multiple": False,
        }
        selected = await wait_for_choice(choice_data)

        if not selected:
            return

        if selected == "__new__":
            await self._step_campaign_create(send_fn, wait_for_input, wait_for_choice)
            return

        self.context_store.link_session_campaign(self.session_id, selected)

    async def _step_campaign_create(self, send_fn, wait_for_input, wait_for_choice):
        """Prompt for a new campaign."""
        await self._say(send_fn, "What are you working on? A sentence or two is plenty.")

        response = await wait_for_input()
        if _is_skip(response):
            return

        result = await self._extract(send_fn, response, "campaign")
        if not result:
            # Fallback: store directly if LLM extraction failed
            cid = self.context_store.create_campaign(description=response[:200])
            self.context_store.link_session_campaign(self.session_id, cid)

    async def _step_planned_session(self, planned_sessions, send_fn, wait_for_choice):
        """Match a planned session from today's calendar."""
        options = [
            {
                "id": ps.id,
                "label": ps.display_title,
                "description": (ps.notes[:60] if ps.notes else ""),
            }
            for ps in planned_sessions
        ]
        options.append(
            {
                "id": "__other__",
                "label": "Something else",
                "description": "Not one of these",
            }
        )

        count = len(planned_sessions)
        await self._say(
            send_fn,
            f"You have {count} session{'s' if count != 1 else ''} planned for today.",
        )

        choice_data = {
            "_type": "single",
            "question": "Starting one of these?",
            "options": options,
            "allow_multiple": False,
        }
        selected = await wait_for_choice(choice_data)

        if not selected or selected == "__other__":
            return

        self.context_store.start_planned_session(selected, self.session_id)

        ps = self.context_store.get_planned_session(selected)
        if ps and ps.notes:
            self.context_store.create_session_intent(
                session_id=self.session_id,
                planned_intent=ps.notes,
                campaign_ids=ps.campaign_ids,
            )

    async def _step_session_intent(self, send_fn, wait_for_input, wait_for_choice):
        """Ask what the session is about — picker with common options."""
        campaigns = self.context_store.get_active_campaigns()

        if campaigns:
            label = campaigns[0].display_name
            await self._say(send_fn, f'Continuing "{label}" — what\'s the plan?')
        else:
            await self._say(send_fn, "What's the plan for this session?")

        choice_data = {
            "_type": "single",
            "question": "Session plan",
            "options": [
                {
                    "id": "timelapse",
                    "label": "Run a timelapse",
                    "description": "Time-series embryo imaging",
                },
                {
                    "id": "continue",
                    "label": "Continue previous work",
                    "description": "Pick up where I left off",
                },
                {
                    "id": "setup",
                    "label": "Set up / calibrate",
                    "description": "Alignment, testing, sample prep",
                },
            ],
            "allow_multiple": False,
        }
        selected = await wait_for_choice(choice_data)

        if not selected:
            return

        intent_map = {
            "timelapse": "Run a timelapse of embryo development",
            "continue": "Continue previous work",
            "setup": "Set up and calibrate the microscope",
        }
        intent_text = intent_map.get(selected)

        if not intent_text and selected != "__custom__" and not _is_skip(selected):
            # Custom text typed inline in the picker
            result = await self._extract(send_fn, selected, "session")
            if not result:
                self.context_store.create_session_intent(
                    session_id=self.session_id,
                    planned_intent=selected,
                )
            if campaigns:
                for c in campaigns:
                    self.context_store.link_session_campaign(self.session_id, c.id)
            return

        if intent_text:
            self.context_store.create_session_intent(
                session_id=self.session_id,
                planned_intent=intent_text,
            )
            if campaigns:
                for c in campaigns:
                    self.context_store.link_session_campaign(self.session_id, c.id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _store_learning(self, content: str, basis: str = "onboarding:identity"):
        """Write a learning directly — no LLM round-trip."""
        self.context_store.add_learning(
            Learning(
                id=str(uuid.uuid4())[:8],
                content=content,
                confidence=Confidence.HIGH,
                basis=basis,
            )
        )

    async def _extract(self, send_fn, response: str, topic: str):
        """Run LLM extraction silently — no acknowledgment message.

        The _finish() method handles the wrap-up message after all
        extraction steps are done.
        """
        await self._think(send_fn)
        try:
            result = await process_onboarding_response(
                response=response,
                topic=topic,
                context_store=self.context_store,
                claude_client=self.claude_client,
                session_id=self.session_id,
            )
            # Close the thinking indicator without a message
            await send_fn({"type": "stream_end", "tokens": _empty_tokens()})
            return result
        except Exception as e:
            logger.warning(f"LLM extraction failed for {topic}: {e}")
            await send_fn({"type": "stream_end", "tokens": _empty_tokens()})
            return None

    async def _think(self, send_fn):
        """Show the thinking spinner in the TUI."""
        await send_fn({"type": "thinking"})

    async def _say(self, send_fn, text: str):
        """Send a text message and close the stream (one bubble)."""
        await send_fn({"type": "text", "text": text})
        await send_fn({"type": "stream_end", "tokens": _empty_tokens()})

    async def _finish(self, send_fn):
        """Emit welcome summary and wizard_complete flag."""
        import asyncio

        campaigns = self.context_store.get_active_campaigns()
        intent = self.context_store.get_current_session_intent()
        learnings = self.context_store.get_learnings()

        campaign_name = campaigns[0].display_name if campaigns else None
        plan = intent.planned_intent if intent else None
        organism = None
        for learning in learnings:
            if learning.content.startswith("Lab organism:"):
                organism = learning.content.split(":", 1)[1].strip()
                break

        # Try LLM-generated summary
        await self._think(send_fn)
        summary = None
        if self.claude_client and (campaign_name or plan or organism):
            context_parts = []
            if organism:
                context_parts.append(f"Organism: {organism}")
            if campaign_name:
                context_parts.append(f"Campaign: {campaign_name}")
            if plan:
                context_parts.append(f"Session plan: {plan}")
            context_str = "\n".join(context_parts)
            prompt = (
                f"You are a microscopy agent. Onboarding just finished. "
                f"Here's what you know:\n\n{context_str}\n\n"
                f"Write a brief (2-3 sentences max) ready message. Summarize "
                f"what you understood and offer to help. Be specific to their "
                f"organism/research, not generic. Don't use bullet points."
            )
            try:
                resp = await asyncio.to_thread(
                    self.claude_client.messages.create,
                    model=settings.models.medium,
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}],
                )
                summary = resp.content[0].text.strip()
            except Exception:
                pass

        if not summary:
            if organism:
                summary = (
                    f"Got it — {organism}. I'm ready to help with your imaging session."
                    " What would you like to do?"
                )
            else:
                summary = "All set. What can I help with?"

        await send_fn({"type": "text", "text": summary})
        await send_fn(
            {
                "type": "stream_end",
                "tokens": _empty_tokens(),
                "wizard_complete": True,
            }
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_skip(text: str) -> bool:
    return text.strip().lower() in SKIP_PHRASES


def _empty_tokens() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }
