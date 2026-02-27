"""
Ingestion capability — how external knowledge enters the daemon's mind.

Reads papers (URLs, PDFs), protocols, notes, and past session data.
Extracts structured context: campaign plans, imaging parameters, sample
requirements, expectations, watchpoints.

This is the mechanism that compresses the cold start curve.
"""

import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..settings import settings

from ..context.model import (
    Campaign,
    Expectation,
    Learning,
    Watchpoint,
    Confidence,
    Significance,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """What was extracted from ingested material."""
    source: str  # URL, file path, or description
    source_type: str  # "paper", "protocol", "notes", "session_history"
    summary: str  # Human-readable summary of what was extracted

    # Extracted context entries
    campaign_proposal: Optional[Dict[str, Any]] = None
    learnings: List[Dict[str, str]] = field(default_factory=list)
    expectations: List[Dict[str, str]] = field(default_factory=list)
    watchpoints: List[Dict[str, str]] = field(default_factory=list)
    imaging_parameters: Optional[Dict[str, Any]] = None
    sample_requirements: Optional[Dict[str, Any]] = None

    @property
    def entry_count(self) -> int:
        count = len(self.learnings) + len(self.expectations) + len(self.watchpoints)
        if self.campaign_proposal:
            count += 1
        return count


# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

PAPER_EXTRACTION_PROMPT = """\
You are helping a microscopy researcher plan experiments. Read this paper and extract
information relevant to planning imaging experiments.

Extract the following, if present:

1. **Campaign proposal**: What experiments could be designed based on this paper?
   Include: goal, target sample count, expected duration, success criteria.

2. **Imaging parameters**: Any mentioned or implied parameters:
   - Time intervals between acquisitions
   - Z-stack depth/step size
   - Exposure times
   - Channels/wavelengths
   - Temperature conditions

3. **Sample requirements**: What organism, strain, stage, preparation is needed?

4. **Developmental expectations**: Timeline predictions — when do key transitions
   happen? How long between stages? What's the expected rate?

5. **Things to watch for**: Known failure modes, subtle phenotypes, quality
   indicators, common pitfalls.

6. **Key learnings**: Important biological facts or experimental insights that
   would help an imaging assistant understand what it's looking at.

Respond in JSON format:
{
  "summary": "Brief summary of the paper's relevance",
  "campaign_proposal": {
    "description": "Proposed campaign based on this paper",
    "target": "What to achieve (e.g., '30 embryos through comma stage')",
    "duration_estimate": "Estimated time to complete",
    "success_criteria": "How to know we're done"
  },
  "imaging_parameters": {
    "interval_minutes": null,
    "z_step_um": null,
    "z_range_um": null,
    "exposure_ms": null,
    "temperature_c": null,
    "notes": "any notes about imaging"
  },
  "sample_requirements": {
    "organism": "",
    "strain": "",
    "stage_at_start": "",
    "preparation": "",
    "notes": ""
  },
  "expectations": [
    {"target": "what", "prediction": "what will happen", "timeframe": "when"}
  ],
  "watchpoints": [
    {"target": "what to watch", "condition": "what to look for"}
  ],
  "learnings": [
    {"content": "the insight", "confidence": "high|medium|low"}
  ]
}

If a field is not mentioned or not applicable, use null or empty list.
"""

PROTOCOL_EXTRACTION_PROMPT = """\
You are helping a microscopy researcher plan experiments. Read this protocol and extract
structured experimental information.

Extract: imaging parameters, sample preparation steps, expected outcomes, timing,
quality checks, and common failure points.

Respond in the same JSON format as for paper extraction.
"""

SESSION_SYNTHESIS_PROMPT = """\
You are reviewing past experimental sessions to build understanding. Here is data
from {session_count} past sessions.

Synthesize:
1. What stages were most commonly observed?
2. What imaging parameters were used?
3. What patterns emerged across sessions?
4. What problems or surprises occurred?
5. What should the daemon know for future sessions?

Respond in JSON with "learnings" (list) and "watchpoints" (list).
"""


class IngestionCapability:
    """
    Ingests external knowledge and converts it to daemon context.

    Supports:
    - Papers (via URL fetch or PDF reading)
    - Protocols (text or file)
    - Notes (direct text input)
    - Past session data (from GentlyStore)
    """

    def __init__(self, claude_client: Optional[Any] = None):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic, optional
            Claude API client for extraction. If None, ingestion is disabled.
        """
        self.claude = claude_client
        self.model = settings.models.medium  # Fast, good at extraction

    @property
    def available(self) -> bool:
        return self.claude is not None

    async def ingest_url(self, url: str) -> IngestionResult:
        """
        Fetch a URL and extract structured knowledge.

        Parameters
        ----------
        url : str
            URL to a paper, protocol, or resource.

        Returns
        -------
        IngestionResult
        """
        if not self.available:
            return IngestionResult(
                source=url,
                source_type="paper",
                summary="Ingestion unavailable — no Claude client.",
            )

        logger.info(f"Ingesting URL: {url}")

        # Fetch content via Claude's web capabilities
        content = await self._fetch_url_content(url)
        if not content:
            return IngestionResult(
                source=url,
                source_type="paper",
                summary=f"Could not fetch content from {url}",
            )

        return await self._extract(content, url, "paper")

    async def ingest_pdf(self, path: str) -> IngestionResult:
        """
        Read a PDF file and extract structured knowledge.

        Parameters
        ----------
        path : str
            Path to PDF file.

        Returns
        -------
        IngestionResult
        """
        if not self.available:
            return IngestionResult(
                source=path,
                source_type="paper",
                summary="Ingestion unavailable — no Claude client.",
            )

        file_path = Path(path)
        if not file_path.exists():
            return IngestionResult(
                source=path,
                source_type="paper",
                summary=f"File not found: {path}",
            )

        logger.info(f"Ingesting PDF: {path}")

        # Read PDF content — use Claude's PDF support via base64
        import base64
        pdf_bytes = file_path.read_bytes()
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        return await self._extract_from_pdf(pdf_b64, path)

    async def ingest_text(self, text: str, source: str = "notes") -> IngestionResult:
        """
        Ingest plain text (notes, protocol descriptions, etc.)

        Parameters
        ----------
        text : str
            The text to ingest.
        source : str
            Description of the source.

        Returns
        -------
        IngestionResult
        """
        if not self.available:
            return IngestionResult(
                source=source,
                source_type="notes",
                summary="Ingestion unavailable — no Claude client.",
            )

        logger.info(f"Ingesting text from: {source}")
        return await self._extract(text, source, "notes")

    async def ingest_session_history(self, store: Any, limit: int = 20) -> IngestionResult:
        """
        Read past sessions from GentlyStore and synthesize learnings.

        Parameters
        ----------
        store : GentlyStore
            The data store to read from.
        limit : int
            Maximum number of sessions to review.

        Returns
        -------
        IngestionResult
        """
        if not self.available:
            return IngestionResult(
                source="session_history",
                source_type="session_history",
                summary="Ingestion unavailable — no Claude client.",
            )

        # Gather session data
        sessions = store.list_sessions(limit=limit)
        if not sessions:
            return IngestionResult(
                source="session_history",
                source_type="session_history",
                summary="No past sessions found.",
            )

        logger.info(f"Ingesting {len(sessions)} past sessions")

        # Build a text summary of sessions for Claude to synthesize
        session_summaries = []
        for session in sessions:
            embryos = store.list_embryos(session["id"])
            summary_parts = [
                f"Session {session['id']} ({session.get('created_at', 'unknown date')})",
                f"  Embryos: {len(embryos)}",
            ]
            for embryo in embryos[:10]:
                summary_parts.append(
                    f"  - {embryo.get('id', '?')}: "
                    f"stage={embryo.get('current_stage', '?')}"
                )
            session_summaries.append("\n".join(summary_parts))

        text = "\n\n".join(session_summaries)
        prompt = SESSION_SYNTHESIS_PROMPT.format(session_count=len(sessions))

        return await self._extract(text, "session_history", "session_history", prompt)

    # -------------------------------------------------------------------
    # Internal extraction
    # -------------------------------------------------------------------

    async def _extract(
        self,
        content: str,
        source: str,
        source_type: str,
        custom_prompt: Optional[str] = None,
    ) -> IngestionResult:
        """Run extraction via Claude API."""
        import asyncio

        prompt = custom_prompt or (
            PAPER_EXTRACTION_PROMPT if source_type == "paper"
            else PROTOCOL_EXTRACTION_PROMPT
        )

        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.model,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": content[:50000]}],  # Truncate if huge
            )

            text = response.content[0].text
            return self._parse_extraction(text, source, source_type)

        except Exception as e:
            logger.error(f"Extraction failed for {source}: {e}")
            return IngestionResult(
                source=source,
                source_type=source_type,
                summary=f"Extraction failed: {e}",
            )

    async def _extract_from_pdf(self, pdf_b64: str, source: str) -> IngestionResult:
        """Extract from a PDF using Claude's document understanding."""
        import asyncio

        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.model,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": PAPER_EXTRACTION_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Please extract experimental information from this paper.",
                        },
                    ],
                }],
            )

            text = response.content[0].text
            return self._parse_extraction(text, source, "paper")

        except Exception as e:
            logger.error(f"PDF extraction failed for {source}: {e}")
            return IngestionResult(
                source=source,
                source_type="paper",
                summary=f"PDF extraction failed: {e}",
            )

    async def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch URL content for extraction."""
        import asyncio

        try:
            # Use Claude to read and summarize web content
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.model,
                max_tokens=8192,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Please read and return the full text content of this URL: {url}\n\n"
                                "Focus on methods, results, and any experimental parameters."
                            ),
                        },
                    ],
                }],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"URL fetch failed for {url}: {e}")
            return None

    def _parse_extraction(
        self,
        raw_text: str,
        source: str,
        source_type: str,
    ) -> IngestionResult:
        """Parse Claude's JSON extraction response."""
        # Try to find JSON in the response
        try:
            # Handle case where Claude wraps JSON in markdown code block
            text = raw_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]  # Remove first line
                text = text.rsplit("```", 1)[0]  # Remove last ```
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse JSON from extraction response")
            return IngestionResult(
                source=source,
                source_type=source_type,
                summary=raw_text[:500],
            )

        result = IngestionResult(
            source=source,
            source_type=source_type,
            summary=data.get("summary", "Extraction complete."),
        )

        # Campaign proposal
        if data.get("campaign_proposal"):
            result.campaign_proposal = data["campaign_proposal"]

        # Imaging parameters
        if data.get("imaging_parameters"):
            result.imaging_parameters = data["imaging_parameters"]

        # Sample requirements
        if data.get("sample_requirements"):
            result.sample_requirements = data["sample_requirements"]

        # Learnings
        for item in data.get("learnings", []):
            if isinstance(item, dict) and item.get("content"):
                result.learnings.append(item)

        # Expectations
        for item in data.get("expectations", []):
            if isinstance(item, dict) and item.get("target"):
                result.expectations.append(item)

        # Watchpoints
        for item in data.get("watchpoints", []):
            if isinstance(item, dict) and item.get("target"):
                result.watchpoints.append(item)

        logger.info(
            f"Extracted from {source}: {result.entry_count} entries "
            f"({len(result.learnings)} learnings, "
            f"{len(result.expectations)} expectations, "
            f"{len(result.watchpoints)} watchpoints"
            f"{', campaign proposal' if result.campaign_proposal else ''})"
        )

        return result
