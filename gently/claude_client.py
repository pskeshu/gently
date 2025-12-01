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
from typing import Optional, Tuple, Dict
import anthropic


# ============================================================================
# CLAUDE VISION PROMPTS (from calibration_plans.py)
# ============================================================================

EMBRYO_CENTERING_PROMPT = """You are an expert microscopist examining a diSPIM light sheet microscopy image of a biological embryo sample.

This image shows ONE camera view from the diSPIM system. You should look for an embryo structure somewhere in the field of view. The embryo will appear as a brighter structure against a dark background, but the signal may be MODERATE (not necessarily super bright).

IMPORTANT CONTEXT:
- This is a REAL microscopy image with typical noise and artifacts
- Room lighting may cause some background glow (this is normal - ignore it)
- Embryos appear as irregularly-shaped bright regions, NOT perfectly uniform
- The embryo does NOT need to be perfectly centered, just reasonably visible in the frame
- Signal levels are moderate - don't expect extremely bright fluorescence

YOUR TASK:
Determine if an embryo structure is visible in this image that we can use for calibration.

WHAT TO LOOK FOR (be forgiving):
✓ EMBRYO VISIBLE:
  - ANY distinct structure brighter than the background (doesn't need to be super bright)
  - Some defined boundary or edge (even if irregular or somewhat soft)
  - Structure appears biological (rounded, irregular, or oblong shape)
  - Structure is not cut off at the edge of the frame
  - You can distinguish the embryo from background even if contrast is moderate

✗ NO USABLE EMBRYO:
  - Absolutely no structure visible, only uniform background
  - Frame is completely empty or saturated
  - No contrast at all between any structures and background
  - Any visible structure is completely cut off at frame edge

RESPOND FORMAT:
Line 1: "yes" if ANY embryo structure is visible that we can work with, "no" only if truly absent
Line 2: Brief description of what you see (1-2 sentences)

Example response:
yes
An irregularly-shaped embryo structure is visible in the left-center region with moderate brightness and defined boundaries against the dark background."""

EMBRYO_EDGE_PROMPT = """You are an expert microscopist specializing in diSPIM light sheet microscopy of embryos.

This image shows ONE camera view of an embryo captured with light sheet illumination. We are trying to determine if the embryo is still visible at this Z position.

CONTEXT:
- We are sweeping through Z positions to find where the embryo appears/disappears
- This may be at the edge of the embryo where it starts to fade out
- We need to detect even faint/sparse embryo signal

YOUR TASK:
Determine if there is ANY embryo structure visible in this image, even if faint or sparse.

WHAT COUNTS AS VISIBLE:
✓ YES (embryo visible):
  - Any distinct embryo structure, even if faint
  - Partial embryo at edge (sparse but present)
  - Moderate contrast showing biological structure
  - Even if only a small portion is visible

✗ NO (embryo not visible):
  - Completely empty/uniform background
  - Only noise and artifacts, no structure
  - Embryo has completely disappeared

RESPOND FORMAT:
Line 1: "yes" if embryo is visible (even faintly), "no" if completely absent
Line 2: Brief description

Example:
yes
Faint embryo structure visible in center, appears to be at edge of sample."""


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
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5-20251101",
        max_tokens: int = 100,
        timeout: float = 30.0
    ):
        """Initialize async Claude client."""
        if api_key is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
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
        with open(image_path, 'rb') as f:
            image_data = f.read()
        return base64.standard_b64encode(image_data).decode('utf-8')

    @staticmethod
    def parse_yes_no_response(response_text: str) -> Tuple[bool, str]:
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
        lines = response_text.strip().split('\n', 1)

        if len(lines) == 0:
            return False, "Empty response"

        # Parse first line for yes/no
        first_line = lines[0].strip().lower()
        is_yes = 'yes' in first_line

        # Get description from remaining lines
        description = lines[1].strip() if len(lines) > 1 else "No description provided"

        return is_yes, description

    async def check_embryo_centered(
        self,
        image_path: Path,
        custom_prompt: Optional[str] = None
    ) -> Tuple[bool, str]:
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
        ext = image_path.suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/png')

        # Prepare prompt
        prompt = custom_prompt if custom_prompt else EMBRYO_CENTERING_PROMPT

        try:
            # Make async API call with timeout
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                ),
                timeout=self.timeout
            )

            # Extract text response
            response_text = response.content[0].text

            # Parse yes/no response
            is_visible, description = self.parse_yes_no_response(response_text)

            return is_visible, description

        except asyncio.TimeoutError:
            return False, f"Claude API timeout after {self.timeout}s"
        except Exception as e:
            return False, f"Claude API error: {str(e)}"

    async def detect_embryo_presence(
        self,
        image_path: Path,
        custom_prompt: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Detect if embryo is present at current Z position (for edge detection).

        Uses EMBRYO_EDGE_PROMPT to determine if any embryo structure is
        visible, even if faint or at the edge of the sample.

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
        str
            Description of what Claude sees

        Examples
        --------
        >>> client = AsyncClaudeClient()
        >>> present, desc = await client.detect_embryo_presence(Path("edge.png"))
        >>> if not present:
        ...     print("Reached embryo edge")
        """
        image_path = Path(image_path)

        if not image_path.exists():
            return False, f"Image file not found: {image_path}"

        # Encode image
        image_data = self.encode_image(image_path)

        # Get media type
        ext = image_path.suffix.lower()
        media_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/png')

        # Prepare prompt
        prompt = custom_prompt if custom_prompt else EMBRYO_EDGE_PROMPT

        try:
            # Make async API call with timeout
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                ),
                timeout=self.timeout
            )

            # Extract text response
            response_text = response.content[0].text

            # Parse yes/no response
            is_present, description = self.parse_yes_no_response(response_text)

            return is_present, description

        except asyncio.TimeoutError:
            return False, f"Claude API timeout after {self.timeout}s"
        except Exception as e:
            return False, f"Claude API error: {str(e)}"

    async def validate_focus_montage(
        self,
        montage_path: Path,
        selected_position_um: float,
        prompt_template: Optional[str] = None
    ) -> Tuple[str, str]:
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
            prompt_template = """You are an expert microscopist reviewing focus quality in embryo images.

This montage shows a focus sweep through an embryo sample. Each panel is labeled with its Z position in micrometers.

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
        media_type = 'image/png'

        try:
            # Make async API call
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=150,  # Slightly longer for reasoning
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }]
                ),
                timeout=self.timeout
            )

            # Extract response
            response_text = response.content[0].text
            lines = response_text.strip().split('\n', 1)

            decision = lines[0].strip().upper()
            reasoning = lines[1].strip() if len(lines) > 1 else "No reasoning provided"

            # Normalize decision
            if 'CONFIRM' in decision:
                decision = 'CONFIRM'
            elif 'REJECT' in decision:
                decision = 'REJECT'
            else:
                decision = 'REJECT'  # Default to reject if unclear
                reasoning = f"Unclear response: {decision}. {reasoning}"

            return decision, reasoning

        except asyncio.TimeoutError:
            return "REJECT", f"Claude API timeout after {self.timeout}s"
        except Exception as e:
            return "REJECT", f"Claude API error: {str(e)}"


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

    def check_embryo_centered(self, image_path: Path) -> Tuple[bool, str]:
        """Sync version of check_embryo_centered."""
        return asyncio.run(self.async_client.check_embryo_centered(image_path))

    def detect_embryo_presence(self, image_path: Path) -> Tuple[bool, str]:
        """Sync version of detect_embryo_presence."""
        return asyncio.run(self.async_client.detect_embryo_presence(image_path))

    def validate_focus_montage(
        self,
        montage_path: Path,
        selected_position_um: float
    ) -> Tuple[str, str]:
        """Sync version of validate_focus_montage."""
        return asyncio.run(
            self.async_client.validate_focus_montage(montage_path, selected_position_um)
        )
