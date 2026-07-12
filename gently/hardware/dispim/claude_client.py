#!/usr/bin/env python3
"""
Async Claude API Client for Embryo Calibration
===============================================

Provides async wrapper around Anthropic Claude API for use in Bluesky plans
without blocking the RunEngine.

Key Features:
- Async/await API calls that don't block RunEngine
- Image encoding utilities
- Embryo-specific prompts for calibration workflow
- Automatic fallback and error handling
- Timeout support
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import cast

import anthropic
from anthropic.types import MessageParam, TextBlock

from gently.settings import settings

from .plans.calibration import EMBRYO_CENTERING_PROMPT, EMBRYO_EDGE_PROMPT

_MEDIA_TYPE_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


# ============================================================================
# ASYNC CLAUDE CLIENT
# ============================================================================


class AsyncClaudeClient:
    """
    Async Claude API client for embryo calibration workflows.

    Provides async methods for calling Claude API without blocking the
    Bluesky RunEngine. Supports image analysis for embryo centering and
    edge detection.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. If None, reads from ANTHROPIC_API_KEY env variable.
    model : str, optional
        Claude model to use (default: claude-opus-4-5-20251101)
    max_tokens : int, optional
        Maximum tokens in response (default: 100)
    timeout : float, optional
        Request timeout in seconds (default: 30.0)

    Examples
    --------
    >>> client = AsyncClaudeClient()
    >>> visible, description = await client.check_embryo_centered("image.png")
    >>> print(f"Embryo visible: {visible}, {description}")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = settings.models.perception,
        max_tokens: int = 100,
        timeout: float = 30.0,
    ):
        """Initialize async Claude client."""
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment. "
                    "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
                )

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    @staticmethod
    def encode_image(image_path: Path) -> str:
        """
        Encode image file to base64 string for Claude API.

        Parameters
        ----------
        image_path : Path
            Path to image file (PNG, JPG, etc.)

        Returns
        -------
        str
            Base64-encoded image data
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.standard_b64encode(image_data).decode("utf-8")

    @staticmethod
    def parse_yes_no_response(response_text: str) -> tuple[bool, str]:
        """
        Parse Claude's yes/no response format.

        Expected format:
        Line 1: "yes" or "no"
        Line 2+: Description

        Parameters
        ----------
        response_text : str
            Claude's response text

        Returns
        -------
        bool
            True if "yes", False if "no"
        str
            Description from remaining lines
        """
        lines = response_text.strip().split("\n", 1)

        if len(lines) == 0:
            return False, "Empty response"

        # Parse first line for yes/no
        first_line = lines[0].strip().lower()
        is_yes = "yes" in first_line

        # Get description from remaining lines
        description = lines[1].strip() if len(lines) > 1 else "No description provided"

        return is_yes, description

    async def check_embryo_centered(
        self, image_path: Path, custom_prompt: str | None = None
    ) -> tuple[bool, str]:
        """
        Check if embryo is centered and visible in image.

        Uses EMBRYO_CENTERING_PROMPT to determine if an embryo structure
        is visible and suitable for calibration.

        Parameters
        ----------
        image_path : Path
            Path to microscopy image (PNG)
        custom_prompt : str, optional
            Custom prompt to use instead of default

        Returns
        -------
        bool
            True if embryo is visible and centered
        str
            Description of what Claude sees

        Examples
        --------
        >>> client = AsyncClaudeClient()
        >>> visible, desc = await client.check_embryo_centered(Path("embryo.png"))
        >>> if visible:
        ...     print(f"Embryo found: {desc}")
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return False, f"Image file not found: {image_path}"

        # Encode image
        image_data = self.encode_image(image_path)

        # Get media type from file extension
        media_type = _MEDIA_TYPE_MAP.get(image_path.suffix.lower(), "image/png")

        # Prepare prompt
        prompt = custom_prompt if custom_prompt else EMBRYO_CENTERING_PROMPT

        try:
            # Make async API call with timeout
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=cast(
                        list[MessageParam],
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": image_data,
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    ),
                ),
                timeout=self.timeout,
            )

            # Extract text response
            response_text = cast(TextBlock, response.content[0]).text

            # Parse yes/no response
            is_visible, description = self.parse_yes_no_response(response_text)

            return is_visible, description

        except asyncio.TimeoutError:
            return False, f"Claude API timeout after {self.timeout}s"
        except Exception as e:
            return False, f"Claude API error: {str(e)}"

    async def detect_embryo_presence(
        self, image_path: Path, custom_prompt: str | None = None
    ) -> tuple[bool, int, str]:
        """
        Detect if embryo is present at current Z position (for edge detection).

        Uses EMBRYO_EDGE_PROMPT to determine if any embryo structure is
        visible, even if faint or at the edge of the sample. Also returns
        a feature richness score for focus position selection.

        Parameters
        ----------
        image_path : Path
            Path to microscopy image (PNG)
        custom_prompt : str, optional
            Custom prompt to use instead of default

        Returns
        -------
        bool
            True if any embryo structure is visible
        int
            Feature richness score (0-10). Higher = better for focus calibration.
            0 if no embryo visible, 1-3 sparse edge, 7-10 dense features.
        str
            Description of what Claude sees

        Examples
        --------
        >>> client = AsyncClaudeClient()
        >>> present, score, desc = await client.detect_embryo_presence(Path("edge.png"))
        >>> if not present:
        ...     print("Reached embryo edge")
        >>> elif score >= 7:
        ...     print("Good candidate for focus calibration")
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return False, 0, f"Image file not found: {image_path}"

        # Encode image
        image_data = self.encode_image(image_path)

        # Get media type
        media_type = _MEDIA_TYPE_MAP.get(image_path.suffix.lower(), "image/png")

        # Prepare prompt
        prompt = custom_prompt if custom_prompt else EMBRYO_EDGE_PROMPT

        try:
            # Make async API call with timeout
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=cast(
                        list[MessageParam],
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": image_data,
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    ),
                ),
                timeout=self.timeout,
            )

            # Extract text response
            response_text = cast(TextBlock, response.content[0]).text

            # Parse response - now 3 lines: yes/no, score, description
            lines = response_text.strip().split("\n")

            # Line 1: yes/no
            is_present = "yes" in lines[0].lower() if lines else False

            # Line 2: feature score (1-10)
            feature_score = 0
            if len(lines) > 1:
                try:
                    # Extract number from second line
                    score_line = lines[1].strip()
                    # Handle cases like "8" or "Score: 8" or "8/10"
                    import re

                    match = re.search(r"\d+", score_line)
                    if match:
                        feature_score = min(10, max(0, int(match.group())))
                except (ValueError, IndexError):
                    feature_score = 5 if is_present else 0  # Default

            # Line 3+: description
            description = "\n".join(lines[2:]).strip() if len(lines) > 2 else "No description"

            # If not present, ensure score is 0
            if not is_present:
                feature_score = 0

            return is_present, feature_score, description

        except asyncio.TimeoutError:
            return False, 0, f"Claude API timeout after {self.timeout}s"
        except Exception as e:
            return False, 0, f"Claude API error: {str(e)}"

    async def validate_focus_montage(
        self,
        montage_path: Path,
        selected_position_um: float,
        prompt_template: str | None = None,
    ) -> tuple[str, str]:
        """
        Validate algorithmic focus selection by analyzing montage.

        Shows Claude a montage of focus sweep images and asks if the
        algorithmically selected position looks correct.

        Parameters
        ----------
        montage_path : Path
            Path to montage image showing all focus positions
        selected_position_um : float
            Piezo position selected by FFT algorithm
        prompt_template : str, optional
            Custom prompt template (must contain {position} placeholder)

        Returns
        -------
        str
            "CONFIRM" or "REJECT"
        str
            Reasoning for decision

        Examples
        --------
        >>> client = AsyncClaudeClient()
        >>> decision, reason = await client.validate_focus_montage(
        ...     Path("montage.png"), selected_position_um=5.2
        ... )
        >>> if decision == "CONFIRM":
        ...     print(f"Focus validated: {reason}")
        """
        montage_path = Path(montage_path)

        if not montage_path.exists():
            return "REJECT", f"Montage file not found: {montage_path}"

        # Default validation prompt
        if prompt_template is None:
            prompt_template = """You are an expert microscopist reviewing focus quality in
embryo images.

This montage shows a focus sweep through an embryo sample. Each panel is labeled with its
Z position in micrometers.

Our FFT-based algorithm selected position: {position:.2f} µm as optimal focus.

YOUR TASK:
Review the montage and determine if the selected position looks correctly focused.

CRITERIA for good focus:
- Sharp, well-defined boundaries
- Maximum detail and contrast
- No blur or motion artifacts
- Biological structure clearly visible

RESPOND FORMAT:
Line 1: "CONFIRM" if selection looks good, "REJECT" if clearly wrong
Line 2: Brief reasoning (1-2 sentences)

Example:
CONFIRM
Selected position shows sharp embryo boundaries with maximum contrast and detail."""

        # Format prompt with selected position
        prompt = prompt_template.format(position=selected_position_um)

        # Encode image
        image_data = self.encode_image(montage_path)
        media_type = "image/png"

        try:
            # Make async API call
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=150,  # Slightly longer for reasoning
                    messages=cast(
                        list[MessageParam],
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": image_data,
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    ),
                ),
                timeout=self.timeout,
            )

            # Extract response
            response_text = cast(TextBlock, response.content[0]).text
            lines = response_text.strip().split("\n", 1)

            decision = lines[0].strip().upper()
            reasoning = lines[1].strip() if len(lines) > 1 else "No reasoning provided"

            # Normalize decision
            if "CONFIRM" in decision:
                decision = "CONFIRM"
            elif "REJECT" in decision:
                decision = "REJECT"
            else:
                decision = "REJECT"  # Default to reject if unclear
                reasoning = f"Unclear response: {decision}. {reasoning}"

            return decision, reasoning

        except asyncio.TimeoutError:
            return "REJECT", f"Claude API timeout after {self.timeout}s"
        except Exception as e:
            return "REJECT", f"Claude API error: {str(e)}"

    async def select_best_focus(
        self,
        montage_path: Path,
        offsets: list[float],
        labels: list[str] | None = None,
    ) -> tuple[int, str, str]:
        """
        Select the best-focused image from a montage using Vision.

        Used by hybrid focus selection when FFT scores are ambiguous.
        Claude Vision analyzes the montage and picks the sharpest image.

        Parameters
        ----------
        montage_path : Path
            Path to montage image showing focus positions side-by-side
        offsets : list[float]
            Piezo offsets in µm for each image position
        labels : list[str], optional
            Labels for each position (e.g., ['A', 'B', 'C'])

        Returns
        -------
        int
            Index of best-focused image (0-based)
        str
            Label of selected position (e.g., 'B')
        str
            Brief reasoning for selection

        Examples
        --------
        >>> client = AsyncClaudeClient()
        >>> idx, label, reason = await client.select_best_focus(
        ...     Path("focus_montage.png"),
        ...     offsets=[-2.0, 0.0, 2.0]
        ... )
        >>> print(f"Best focus at position {label} (offset {offsets[idx]}µm)")
        """
        montage_path = Path(montage_path)

        if not montage_path.exists():
            return 1, "B", "Montage file not found, defaulting to center"

        # Default labels
        if labels is None:
            labels = [chr(ord("A") + i) for i in range(len(offsets))]

        # Build offset description for prompt
        offset_desc = ", ".join(f"{labels[i]}={offsets[i]:+.1f}µm" for i in range(len(offsets)))

        prompt = f"""You are an expert microscopist comparing focus quality in embryo images.

This montage shows the same embryo at different focus positions (piezo offsets):
{offset_desc}

YOUR TASK:
Select which position shows the SHARPEST focus with:
- Clearest cell membrane boundaries
- Most distinct nuclear structures
- Best overall image clarity and contrast

RESPOND FORMAT:
Line 1: Just the letter ({", ".join(labels)})
Line 2: Brief reasoning (1 sentence)

Example:
B
Center position shows sharpest nuclear boundaries with maximum contrast."""

        # Encode image
        image_data = self.encode_image(montage_path)
        media_type = "image/png"

        try:
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=cast(
                        list[MessageParam],
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": image_data,
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    ),
                ),
                timeout=self.timeout,
            )

            # Parse response
            response_text = cast(TextBlock, response.content[0]).text.strip()
            lines = response_text.split("\n", 1)

            # Extract selected label (first non-empty character that's a valid label)
            selected = None
            for char in lines[0].upper():
                if char in labels:
                    selected = char
                    break

            if selected is None:
                # Default to center if can't parse
                selected = labels[len(labels) // 2]

            # Get index
            idx = labels.index(selected)
            reasoning = lines[1].strip() if len(lines) > 1 else "No reasoning provided"

            return idx, selected, reasoning

        except asyncio.TimeoutError:
            # Default to center on timeout
            center_idx = len(offsets) // 2
            return (
                center_idx,
                labels[center_idx],
                "Claude API timeout, defaulting to center",
            )
        except Exception as e:
            center_idx = len(offsets) // 2
            return (
                center_idx,
                labels[center_idx],
                f"Claude API error: {str(e)}, defaulting to center",
            )


# ============================================================================
# SYNCHRONOUS WRAPPER FOR BACKWARDS COMPATIBILITY
# ============================================================================


class ClaudeClient:
    """
    Synchronous wrapper around AsyncClaudeClient for backwards compatibility.

    Use this when you can't use async/await (e.g., in non-async contexts).
    For new code in Bluesky plans, prefer using AsyncClaudeClient directly.
    """

    def __init__(self, **kwargs):
        """Initialize sync client (wraps async client)."""
        self.async_client = AsyncClaudeClient(**kwargs)

    def check_embryo_centered(self, image_path: Path) -> tuple[bool, str]:
        """Sync version of check_embryo_centered."""
        return asyncio.run(self.async_client.check_embryo_centered(image_path))

    def detect_embryo_presence(self, image_path: Path) -> tuple[bool, int, str]:
        """Sync version of detect_embryo_presence."""
        return asyncio.run(self.async_client.detect_embryo_presence(image_path))

    def validate_focus_montage(
        self, montage_path: Path, selected_position_um: float
    ) -> tuple[str, str]:
        """Sync version of validate_focus_montage."""
        return asyncio.run(
            self.async_client.validate_focus_montage(montage_path, selected_position_um)
        )
