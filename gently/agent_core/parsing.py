"""
Response parsing for agent output.

Parses the structured response from the LLM into actions and context updates.
"""

import re
import uuid
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ..context import (
    ContextUpdates,
    Observation,
    Expectation,
    Watchpoint,
    Learning,
    Question,
    Significance,
    Confidence,
    ExpectationStatus,
)
from .types import ThinkResult

logger = logging.getLogger(__name__)


def _gen_id() -> str:
    return str(uuid.uuid4())[:8]


def _parse_section(text: str, tag: str) -> Optional[str]:
    """Extract content between XML-like tags."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_yaml_like_list(text: str) -> List[Dict[str, Any]]:
    """
    Parse a YAML-like list from the response.

    Example input:
    - observation: "something happened"
      significance: high
      relates_to: [goal1, embryo_1]

    Returns list of dicts.
    """
    items = []
    current_item = {}
    indent_level = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # New item starts with "-"
        if stripped.startswith("-"):
            if current_item:
                items.append(current_item)
            current_item = {}
            # Parse first key-value on same line
            rest = stripped[1:].strip()
            if ":" in rest:
                key, value = rest.split(":", 1)
                current_item[key.strip()] = _parse_value(value.strip())

        elif ":" in stripped and current_item is not None:
            # Continuation key-value
            key, value = stripped.split(":", 1)
            current_item[key.strip()] = _parse_value(value.strip())

    if current_item:
        items.append(current_item)

    return items


def _parse_value(value: str) -> Any:
    """Parse a value from YAML-like format."""
    value = value.strip()

    # Remove quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # List
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        if not inner:
            return []
        return [v.strip().strip('"').strip("'") for v in inner.split(",")]

    # Dict-like (simple)
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1]
        if not inner:
            return {}
        result = {}
        for pair in inner.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                result[k.strip().strip('"')] = v.strip().strip('"')
        return result

    # Boolean
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def _parse_significance(value: str) -> Significance:
    """Parse significance value."""
    value = value.lower()
    if value == "high":
        return Significance.HIGH
    if value == "low":
        return Significance.LOW
    return Significance.MEDIUM


def _parse_confidence(value: str) -> Confidence:
    """Parse confidence value."""
    value = value.lower()
    if value == "high":
        return Confidence.HIGH
    if value == "low":
        return Confidence.LOW
    return Confidence.MEDIUM


def _parse_time(value: str) -> datetime:
    """Parse a time reference into datetime."""
    now = datetime.now()

    # Relative time (e.g., "in 30 minutes", "+30m")
    if "minute" in value.lower() or value.startswith("+"):
        try:
            # Extract number
            nums = re.findall(r"\d+", value)
            if nums:
                minutes = int(nums[0])
                return now + timedelta(minutes=minutes)
        except Exception:
            pass

    # Absolute time (e.g., "14:30")
    try:
        time_match = re.search(r"(\d{1,2}):(\d{2})", value)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            result = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            # If time is in the past, assume tomorrow
            if result < now:
                result += timedelta(days=1)
            return result
    except Exception:
        pass

    # Default: 30 minutes from now
    return now + timedelta(minutes=30)


def parse_observations(text: str) -> List[Observation]:
    """Parse observations section."""
    section = _parse_section(text, "observations")
    if not section:
        return []

    items = _parse_yaml_like_list(section)
    observations = []

    for item in items:
        obs_text = item.get("observation", "")
        if not obs_text:
            continue

        observations.append(Observation(
            id=_gen_id(),
            timestamp=datetime.now(),
            type="agent_observation",
            content=obs_text,
            embryo_id=item.get("embryo_id"),
            significance=_parse_significance(item.get("significance", "medium")),
            relates_to=item.get("relates_to") if isinstance(item.get("relates_to"), list) else None,
        ))

    return observations


def parse_actions(text: str) -> List[Dict[str, Any]]:
    """Parse actions section."""
    section = _parse_section(text, "actions")
    if not section:
        return []

    items = _parse_yaml_like_list(section)
    actions = []

    for item in items:
        action_type = item.get("action")
        if not action_type:
            continue

        actions.append({
            "type": action_type,
            "params": item.get("params", {}),
            "reason": item.get("reason", ""),
        })

    return actions


def parse_context_updates(text: str) -> ContextUpdates:
    """Parse context_updates section into ContextUpdates."""
    section = _parse_section(text, "context_updates")
    if not section:
        return ContextUpdates()

    updates = ContextUpdates()
    items = _parse_yaml_like_list(section)

    for item in items:
        update_type = item.get("update")
        content = item.get("content", {})
        if isinstance(content, str):
            # Parser returned a string instead of dict — wrap it
            content = {"content": content}

        if update_type == "expectation":
            exp = Expectation(
                id=_gen_id(),
                target=content.get("target", "unknown"),
                prediction=content.get("prediction", ""),
                expected_time=_parse_time(content.get("expected_time", "")),
                uncertainty=content.get("uncertainty"),
                basis=content.get("basis"),
            )
            updates.new_expectations.append(exp)

        elif update_type == "watchpoint":
            wp = Watchpoint(
                id=_gen_id(),
                target=content.get("target", "unknown"),
                condition=content.get("condition", ""),
                priority=_parse_significance(content.get("priority", "medium")),
            )
            updates.new_watchpoints.append(wp)

        elif update_type == "learning":
            learning = Learning(
                id=_gen_id(),
                content=content.get("content", ""),
                confidence=_parse_confidence(content.get("confidence", "medium")),
                basis=content.get("basis"),
            )
            updates.new_learnings.append(learning)

        elif update_type == "understanding":
            embryo_id = content.get("embryo_id")
            if embryo_id:
                update_dict = {}
                if "stage" in content:
                    update_dict["current_stage"] = content["stage"]
                if "confidence" in content:
                    update_dict["stage_confidence"] = _parse_confidence(content["confidence"])
                if "note" in content:
                    update_dict["note"] = content["note"]
                if "needs_attention" in content:
                    update_dict["needs_attention"] = content["needs_attention"]
                if "attention_reason" in content:
                    update_dict["attention_reason"] = content["attention_reason"]
                if update_dict:
                    updates.embryo_updates[embryo_id] = update_dict

        elif update_type == "question":
            q = Question(
                id=_gen_id(),
                content=content.get("content", ""),
            )
            updates.new_questions.append(q)

        elif update_type == "resolve_expectation":
            exp_id = content.get("id")
            status = content.get("status", "confirmed")
            if exp_id:
                status_enum = ExpectationStatus.CONFIRMED if status == "confirmed" else ExpectationStatus.SURPRISED
                updates.resolved_expectations[exp_id] = status_enum

        elif update_type == "focus":
            updates.new_focus = content.get("focus")

    return updates


def parse_agent_response(response_text: str, model_used: str = "") -> ThinkResult:
    """
    Parse full agent response into ThinkResult.

    Parameters
    ----------
    response_text : str
        Raw response from the LLM
    model_used : str
        Model that generated the response

    Returns
    -------
    ThinkResult
        Parsed result with actions and context updates
    """
    # Extract reasoning
    reasoning = _parse_section(response_text, "reasoning") or ""

    # Parse sections
    observations = parse_observations(response_text)
    actions = parse_actions(response_text)
    context_updates = parse_context_updates(response_text)

    # Add observations to context updates
    context_updates.new_observations.extend(observations)

    return ThinkResult(
        actions=actions,
        context_updates=context_updates,
        reasoning=reasoning,
        observations_noted=[obs.content for obs in observations],
        model_used=model_used,
    )
