"""
Onboarding — cold start conversation logic.

Bridges gap assessment → interaction → context seeding.
Generates the right questions and processes the answers
into structured context, based on what the agent lacks.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from gently.settings import settings

from .gap_assessment import ContextGapReport, GapLayer
from .model import (
    Confidence,
    Learning,
    Watchpoint,
)

try:
    from .file_store import FileContextStore as ContextStore
except ImportError:
    from .store import ContextStore  # type: ignore[assignment]  # legacy fallback


@dataclass
class OnboardingMessage:
    """A message to surface to the researcher during onboarding."""

    message: str
    layer: GapLayer
    priority: str = "normal"  # "high", "normal", "low"
    reason: str = ""


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Onboarding prompts — what the daemon says at each layer
# ---------------------------------------------------------------------------

LAB_ONBOARDING_GREETING = """\
This looks like our first time working together. I'd like to learn about \
your lab and research so I can be a better assistant.

Could you tell me a bit about your work? For example:
- What organism do you study?
- What are your main research questions?
- What microscope setup do you use?

Just talk naturally — I'll pick up on the details."""

CAMPAIGN_PROMPT_FRESH = """\
I don't have a current research campaign set up. What are you working on \
right now? This could be a specific project, a data collection goal, or \
a question you're trying to answer.

You can also share papers or protocols if you'd like me to help plan — \
just use `/ingest <url or file path>`."""

CAMPAIGN_PROMPT_RETURNING = """\
Your last campaign was: "{campaign_description}"
(Status: {campaign_status})

Are you continuing this work, or starting something new?"""

SESSION_PROMPT_WITH_CAMPAIGN = """\
We're working on: {campaign_description}

{history_context}\
What's the plan for this session?"""

SESSION_PROMPT_NO_CAMPAIGN = """\
What are you planning to do today?"""


def generate_onboarding_messages(
    gap_report: ContextGapReport,
    session_id: str | None = None,
) -> list[OnboardingMessage]:
    """
    Generate onboarding messages based on the gap assessment.

    Parameters
    ----------
    gap_report : ContextGapReport
        Output of assess_gaps().
    session_id : str, optional
        Current session ID.

    Returns
    -------
    List[OnboardingMessage]
        Messages to surface to the researcher.
    """
    messages = []

    if gap_report.needs_lab_onboarding:
        messages.append(
            OnboardingMessage(
                message=LAB_ONBOARDING_GREETING,
                layer=GapLayer.LAB,
                priority="high",
                reason="First launch — need to learn about the lab.",
            )
        )

    if gap_report.needs_campaign:
        if gap_report.past_campaign_count > 0:
            prompt = CAMPAIGN_PROMPT_RETURNING.format(
                campaign_description="(previous campaigns completed)",
                campaign_status="completed",
            )
        else:
            prompt = CAMPAIGN_PROMPT_FRESH

        messages.append(
            OnboardingMessage(
                message=prompt,
                layer=GapLayer.CAMPAIGN,
                priority="normal",
                reason="No active campaign — need research direction.",
            )
        )

    if gap_report.needs_session_intent and session_id:
        if gap_report.has_campaigns:
            history = ""
            if gap_report.session_count > 0:
                history = f"This is session #{gap_report.session_count + 1}. "
            # Use the first active campaign's display name for the prompt
            campaign_name = gap_report.active_campaigns[0].display_name
            prompt = SESSION_PROMPT_WITH_CAMPAIGN.format(
                campaign_description=campaign_name,
                history_context=history,
            )
        else:
            prompt = SESSION_PROMPT_NO_CAMPAIGN

        messages.append(
            OnboardingMessage(
                message=prompt,
                layer=GapLayer.SESSION,
                priority="normal",
                reason="Need to establish session intent.",
            )
        )

    if messages:
        logger.info(
            f"Generated {len(messages)} onboarding messages "
            f"(conversation_weight={gap_report.conversation_weight})"
        )

    return messages


def get_onboarding_messages(
    gap_report: ContextGapReport,
    session_id: str | None = None,
) -> list[str]:
    """
    Get plain-text onboarding messages for direct CLI display.

    Parameters
    ----------
    gap_report : ContextGapReport
        Output of assess_gaps().
    session_id : str, optional
        Current session ID.

    Returns
    -------
    List[str]
        Messages to display.
    """
    messages = []

    if gap_report.needs_lab_onboarding:
        messages.append(LAB_ONBOARDING_GREETING)

    if gap_report.needs_campaign:
        if gap_report.past_campaign_count > 0:
            messages.append(
                CAMPAIGN_PROMPT_RETURNING.format(
                    campaign_description="(previous campaigns completed)",
                    campaign_status="completed",
                )
            )
        else:
            messages.append(CAMPAIGN_PROMPT_FRESH)

    if gap_report.needs_session_intent and session_id:
        if gap_report.has_campaigns:
            history = ""
            if gap_report.session_count > 0:
                history = f"This is session #{gap_report.session_count + 1}. "
            campaign_name = gap_report.active_campaigns[0].display_name
            messages.append(
                SESSION_PROMPT_WITH_CAMPAIGN.format(
                    campaign_description=campaign_name,
                    history_context=history,
                )
            )
        else:
            messages.append(SESSION_PROMPT_NO_CAMPAIGN)

    return messages


# ---------------------------------------------------------------------------
# Response processing — convert conversation into context
# ---------------------------------------------------------------------------

ONBOARDING_EXTRACTION_PROMPT = """\
The researcher said the following in response to an onboarding question about {topic}:

"{response}"

Extract structured context from this response. Return JSON with any that apply:

{{
  "learnings": [
    {{"content": "factual insight", "confidence": "high|medium|low", "basis": "onboarding"}}
  ],
  "campaign": {{
    "description": "research goal description",
    "target": "measurable target if mentioned"
  }},
  "watchpoints": [
    {{"target": "what to watch", "condition": "what to look for"}}
  ],
  "session_intent": "what they plan to do today",
  "organism": "if mentioned",
  "microscope": "if mentioned"
}}

Extract only what's actually stated or clearly implied. Don't invent.
Use null for fields not mentioned.
"""


async def process_onboarding_response(
    response: str,
    topic: str,
    context_store: ContextStore,
    claude_client: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Process a researcher's response during onboarding.

    Extracts structured context and writes it to the context store.

    Parameters
    ----------
    response : str
        What the researcher said.
    topic : str
        What the question was about ("lab", "campaign", "session").
    context_store : ContextStore
        Where to write the extracted context.
    claude_client : optional
        Claude API client for extraction. If None, does basic keyword extraction.
    session_id : str, optional
        Current session ID for session intent.

    Returns
    -------
    dict
        Summary of what was extracted and stored.
    """
    extracted = {"entries_created": 0, "summary": ""}

    if claude_client:
        extracted = await _extract_with_llm(
            response, topic, context_store, claude_client, session_id
        )
    else:
        extracted = _extract_basic(response, topic, context_store, session_id)

    logger.info(
        f"Onboarding response processed ({topic}): {extracted['entries_created']} entries created"
    )
    return extracted


async def _extract_with_llm(
    response: str,
    topic: str,
    context_store: ContextStore,
    claude_client: Any,
    session_id: str | None,
) -> dict[str, Any]:
    """Use Claude to extract structured context from a response."""
    import asyncio
    import json

    prompt = ONBOARDING_EXTRACTION_PROMPT.format(topic=topic, response=response)

    try:
        api_response = await asyncio.to_thread(
            claude_client.messages.create,
            model=settings.models.medium,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = api_response.content[0].text.strip()

        # Parse JSON (handle markdown code blocks)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(text)

    except Exception as e:
        logger.warning(f"LLM extraction failed, falling back to basic: {e}")
        return _extract_basic(response, topic, context_store, session_id)

    entries = 0

    # Store learnings
    for item in data.get("learnings") or []:
        if item and item.get("content"):
            context_store.add_learning(
                Learning(
                    id=str(uuid.uuid4())[:8],
                    content=item["content"],
                    confidence=Confidence(item.get("confidence", "medium")),
                    basis=item.get("basis", f"onboarding:{topic}"),
                )
            )
            entries += 1

    # Store campaign
    campaign_data = data.get("campaign")
    if campaign_data and campaign_data.get("description"):
        context_store.create_campaign(
            description=campaign_data["description"],
            target=campaign_data.get("target"),
        )
        entries += 1

    # Store watchpoints
    for item in data.get("watchpoints") or []:
        if item and item.get("target"):
            context_store.add_watchpoint(
                Watchpoint(
                    id=str(uuid.uuid4())[:8],
                    target=item["target"],
                    condition=item.get("condition", "monitor"),
                )
            )
            entries += 1

    # Store session intent
    intent_text = data.get("session_intent")
    if intent_text and session_id:
        context_store.create_session_intent(
            session_id=session_id,
            planned_intent=intent_text,
        )
        entries += 1

    # Store organism/microscope as lab identity learnings
    for field_name in ("organism", "microscope"):
        value = data.get(field_name)
        if value:
            context_store.add_learning(
                Learning(
                    id=str(uuid.uuid4())[:8],
                    content=f"Lab {field_name}: {value}",
                    confidence=Confidence.HIGH,
                    basis="onboarding:identity",
                )
            )
            entries += 1

    return {
        "entries_created": entries,
        "summary": f"Extracted {entries} context entries from {topic} onboarding.",
        "data": data,
    }


def _extract_basic(
    response: str,
    topic: str,
    context_store: ContextStore,
    session_id: str | None,
) -> dict[str, Any]:
    """
    Basic keyword-based extraction when no LLM is available.

    Stores the full response as a learning — better than nothing.
    """
    entries = 0

    # Store the raw response as a learning
    context_store.add_learning(
        Learning(
            id=str(uuid.uuid4())[:8],
            content=f"Researcher ({topic}): {response[:500]}",
            confidence=Confidence.MEDIUM,
            basis=f"onboarding:{topic}",
        )
    )
    entries += 1

    # If this is a session topic and we have a session ID, create intent
    if topic == "session" and session_id:
        context_store.create_session_intent(
            session_id=session_id,
            planned_intent=response[:500],
        )
        entries += 1

    return {
        "entries_created": entries,
        "summary": f"Stored {topic} response as learning (no LLM for extraction).",
    }


# ---------------------------------------------------------------------------
# Ingestion result → context store
# ---------------------------------------------------------------------------


def apply_ingestion_to_context(
    result: Any,  # was "IngestionResult"; that type no longer exists (dead code path)
    context_store: ContextStore,
) -> int:
    """
    Write extracted knowledge from an IngestionResult into the context store.

    Parameters
    ----------
    result : IngestionResult
        Output from the ingestion capability.
    context_store : ContextStore
        Where to write.

    Returns
    -------
    int
        Number of entries written.
    """
    entries = 0

    # Campaign proposal
    if result.campaign_proposal and result.campaign_proposal.get("description"):
        context_store.create_campaign(
            description=result.campaign_proposal["description"],
            target=result.campaign_proposal.get("target"),
        )
        entries += 1

    # Learnings
    for item in result.learnings:
        if item.get("content"):
            context_store.add_learning(
                Learning(
                    id=str(uuid.uuid4())[:8],
                    content=item["content"],
                    confidence=Confidence(item.get("confidence", "medium")),
                    basis=f"ingestion:{result.source}",
                )
            )
            entries += 1

    # Imaging parameters as learnings
    if result.imaging_parameters:
        for key, value in result.imaging_parameters.items():
            if value is not None and key != "notes":
                context_store.add_learning(
                    Learning(
                        id=str(uuid.uuid4())[:8],
                        content=f"Recommended {key}: {value}",
                        confidence=Confidence.MEDIUM,
                        basis=f"ingestion:{result.source}",
                    )
                )
                entries += 1
        # Store notes separately if present
        notes = result.imaging_parameters.get("notes")
        if notes:
            context_store.add_learning(
                Learning(
                    id=str(uuid.uuid4())[:8],
                    content=f"Imaging notes: {notes}",
                    confidence=Confidence.MEDIUM,
                    basis=f"ingestion:{result.source}",
                )
            )
            entries += 1

    # Sample requirements as a learning
    if result.sample_requirements:
        parts = []
        for key, value in result.sample_requirements.items():
            if value and key != "notes":
                parts.append(f"{key}: {value}")
        if parts:
            context_store.add_learning(
                Learning(
                    id=str(uuid.uuid4())[:8],
                    content=f"Sample requirements: {', '.join(parts)}",
                    confidence=Confidence.MEDIUM,
                    basis=f"ingestion:{result.source}",
                )
            )
            entries += 1

    # Watchpoints
    for item in result.watchpoints:
        if item.get("target"):
            context_store.add_watchpoint(
                Watchpoint(
                    id=str(uuid.uuid4())[:8],
                    target=item["target"],
                    condition=item.get("condition", "monitor"),
                )
            )
            entries += 1

    logger.info(f"Applied {entries} entries from ingestion of {result.source}")
    return entries
