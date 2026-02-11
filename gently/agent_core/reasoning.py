"""
Agent reasoning — the LLM-based thinking core.

This module contains the Agent class that:
- Constructs prompts from context and world state
- Calls the LLM API
- Parses responses into structured results
- Handles escalation-aware responses
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..context import Context
from .types import ThinkTrigger, ThinkingMode, model_name_for_mode, ThinkResult, WorldState
from .prompt import build_agent_prompt, build_system_prompt
from .parsing import parse_agent_response

logger = logging.getLogger(__name__)


class Agent:
    """
    The reasoning core of the agent.

    Handles LLM calls and response parsing.
    """

    def __init__(self, claude_client: Optional[Any] = None):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic, optional
            Anthropic client for API calls. If None, thinking is simulated.
        """
        self.client = claude_client
        self._system_prompt = build_system_prompt()

    async def think(
        self,
        context: Context,
        world: WorldState,
        trigger: ThinkTrigger,
        mode: ThinkingMode,
        trigger_data: Optional[Dict] = None,
    ) -> ThinkResult:
        """
        Execute one thinking cycle.

        Parameters
        ----------
        context : Context
            Agent's current context
        world : WorldState
            Current world state
        trigger : ThinkTrigger
            What triggered this think
        mode : ThinkingMode
            Thinking depth
        trigger_data : dict, optional
            Additional data about the trigger

        Returns
        -------
        ThinkResult
            Parsed result with actions and context updates
        """
        start_time = time.time()

        # Build prompt
        prompt = build_agent_prompt(context, world, trigger, mode, trigger_data)
        model = model_name_for_mode(mode)

        logger.debug(f"Thinking with {model} (mode={mode.value})")

        if self.client is None:
            # Simulated response for testing
            response_text = self._simulate_response(trigger, mode, context, trigger_data)
        else:
            # Real LLM call
            response_text = await self._call_llm(prompt, model, mode)

        # Parse response
        result = parse_agent_response(response_text, model)
        result.duration_ms = (time.time() - start_time) * 1000

        logger.debug(
            f"Think complete: {result.duration_ms:.0f}ms, "
            f"{len(result.actions)} actions"
        )

        return result

    async def _call_llm(
        self,
        prompt: str,
        model: str,
        mode: ThinkingMode,
    ) -> str:
        """Call the LLM API."""
        # Configure based on mode
        max_tokens = {
            ThinkingMode.FAST: 500,
            ThinkingMode.MODERATE: 1500,
            ThinkingMode.DEEP: 3000,
        }[mode]

        try:
            # Run synchronous Anthropic client in a thread so we don't
            # block the event loop (the daemon shares it with the CLI).
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                max_tokens=max_tokens,
                system=[{
                    "type": "text",
                    "text": self._system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            # Extract text from response
            text_content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text_content += block.text

            return text_content

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"<reasoning>Error calling LLM: {e}</reasoning>"

    def _simulate_response(
        self,
        trigger: ThinkTrigger,
        mode: ThinkingMode,
        context: Context,
        trigger_data: Optional[Dict] = None,
    ) -> str:
        """Generate a simulated response for testing."""
        # Escalation trigger: deeper investigation
        if trigger == ThinkTrigger.ESCALATION:
            reason = (trigger_data or {}).get("reason", "unknown")
            return f"""
<reasoning>
Escalated from quick scan. Reason: {reason}. Investigating more thoroughly.
Looking at the current state and what the quick scan flagged.
This warrants updating our understanding and possibly setting new watchpoints.
</reasoning>

<observations>
- observation: "Escalation investigation complete - {reason}"
  significance: medium
</observations>

<actions>
</actions>

<context_updates>
- update: learning
  content: {{content: "Escalation from fast scan provided useful signal", confidence: "low"}}
</context_updates>
"""

        # Expectation trigger
        if trigger == ThinkTrigger.EXPECTATION:
            exp_type = (trigger_data or {}).get("type", "unknown")
            target = (trigger_data or {}).get("target", "unknown")
            if exp_type == "expired":
                return f"""
<reasoning>
Expectation for {target} has expired. Need to investigate whether the prediction was wrong
or we missed the event. Should observe the target and update our understanding.
</reasoning>

<observations>
- observation: "Expectation expired for {target}, investigating"
  significance: high
</observations>

<actions>
- action: observe
  params: {{target: "{target}"}}
  reason: "check status after expired expectation"
</actions>

<context_updates>
- update: resolve_expectation
  content: {{id: "{(trigger_data or {}).get('expectation_id', '')}", status: "expired"}}
</context_updates>
"""
            else:  # approaching
                return f"""
<reasoning>
Expectation for {target} is approaching. Should check if things are on track.
</reasoning>

<observations>
- observation: "Checking {target} - expectation approaching deadline"
  significance: medium
</observations>

<actions>
- action: observe
  params: {{target: "{target}"}}
  reason: "verify expectation is on track"
</actions>

<context_updates>
</context_updates>
"""

        # Interval - routine scan
        if trigger == ThinkTrigger.INTERVAL:
            return """
<reasoning>
Routine check. No urgent items. All embryos progressing normally.
</reasoning>

<observations>
- observation: "Routine scan complete, all systems nominal"
  significance: low
</observations>

<actions>
</actions>

<context_updates>
</context_updates>
"""

        # Surprise
        if trigger == ThinkTrigger.SURPRISE:
            expected = (trigger_data or {}).get("expected", "?")
            actual = (trigger_data or {}).get("actual", "?")
            embryo = (trigger_data or {}).get("embryo_id", "unknown")
            return f"""
<reasoning>
Something unexpected happened with {embryo}. Expected {expected} but got {actual}.
Need to investigate and update our understanding. This is significant and may require
re-evaluating our predictions for other embryos too.
</reasoning>

<observations>
- observation: "Surprising result for {embryo}: expected {expected}, got {actual}"
  significance: high
  relates_to: [{embryo}]
</observations>

<actions>
- action: observe
  params: {{target: "{embryo}"}}
  reason: "investigate surprise"
</actions>

<context_updates>
- update: question
  content: {{content: "Why did {embryo} show {actual} instead of expected {expected}?"}}
- update: understanding
  content: {{embryo_id: "{embryo}", stage: "{actual}", needs_attention: true, attention_reason: "unexpected stage"}}
</context_updates>
"""

        # User interaction
        if trigger == ThinkTrigger.USER:
            user_msg = (trigger_data or {}).get("message", "")
            return f"""
<reasoning>
User is present and interacting. Should acknowledge and be ready to help.
{f'User said: "{user_msg}"' if user_msg else ''}
</reasoning>

<observations>
- observation: "User interaction detected"
  significance: medium
</observations>

<actions>
- action: speak
  params: {{message: "I noticed your input. How can I help?"}}
  reason: "acknowledge user"
</actions>

<context_updates>
</context_updates>
"""

        # Event
        if trigger == ThinkTrigger.EVENT:
            return """
<reasoning>
An event occurred. Processing and updating context accordingly.
</reasoning>

<observations>
- observation: "Event processed"
  significance: medium
</observations>

<actions>
</actions>

<context_updates>
</context_updates>
"""

        # Watchpoint
        return """
<reasoning>
A watchpoint was triggered. This needs attention.
</reasoning>

<observations>
- observation: "Watchpoint condition met"
  significance: high
</observations>

<actions>
- action: notify
  params: {message: "Watched condition detected"}
  reason: "alert researcher"
</actions>

<context_updates>
</context_updates>
"""


async def create_think_function(claude_client: Optional[Any] = None):
    """
    Factory function to create a think function for the daemon.

    Parameters
    ----------
    claude_client : anthropic.Anthropic, optional
        Anthropic client for API calls

    Returns
    -------
    callable
        Async function that performs thinking
    """
    agent = Agent(claude_client)

    async def think_fn(
        context: Context,
        world: WorldState,
        trigger: ThinkTrigger,
        mode: ThinkingMode,
        trigger_data: Optional[Dict] = None,
    ) -> ThinkResult:
        return await agent.think(context, world, trigger, mode, trigger_data)

    return think_fn
