"""
Shared helper utilities for tool implementations

This module provides common validation and extraction patterns
used across multiple tools to reduce code duplication.
"""

from typing import Any, Dict, Optional, Tuple
from datetime import datetime


def require_agent(context: Dict) -> Tuple[Optional[Any], Optional[str]]:
    """
    Extract agent from context or return error message

    Parameters
    ----------
    context : dict
        Tool execution context

    Returns
    -------
    tuple
        (agent, None) if found, (None, error_message) if not
    """
    agent = context.get('agent')
    if not agent:
        return None, "Error: No agent context"
    return agent, None


def get_embryo_or_error(agent, embryo_id: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Get embryo by any name or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance
    embryo_id : str
        Embryo ID, nickname, or label

    Returns
    -------
    tuple
        (embryo, None) if found, (None, error_message) if not
    """
    embryo = agent.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return None, f"Embryo '{embryo_id}' not found"
    return embryo, None


def require_microscope(context: Dict) -> Tuple[Optional[Any], Optional[str]]:
    """
    Get microscope client from context or return error message

    Parameters
    ----------
    context : dict
        Tool execution context

    Returns
    -------
    tuple
        (client, None) if connected, (None, error_message) if not
    """
    client = context.get('client')
    if not client:
        return None, "Not connected to microscope. Use connect_microscope first."
    return client, None


def require_interaction_logger(agent) -> Tuple[Optional[Any], Optional[str]]:
    """
    Get interaction logger or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (logger, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, 'interaction_logger') or not agent.interaction_logger:
        return None, "Interaction logging not enabled."
    return agent.interaction_logger, None


def require_developmental_tracker(agent) -> Tuple[Optional[Any], Optional[str]]:
    """
    Get developmental tracker or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (tracker, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, 'developmental_tracker') or not agent.developmental_tracker:
        return None, "No stage classifications recorded yet. Use classify_embryo_stage first."
    return agent.developmental_tracker, None


def require_timelapse_orchestrator(agent) -> Tuple[Optional[Any], Optional[str]]:
    """
    Get timelapse orchestrator or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (orchestrator, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, 'timelapse_orchestrator') or agent.timelapse_orchestrator is None:
        return None, "Timelapse orchestrator not initialized."
    return agent.timelapse_orchestrator, None


def require_databroker(agent) -> Tuple[Optional[Any], Optional[str]]:
    """
    Get databroker connection or return error message

    Parameters
    ----------
    agent : MicroscopyAgent
        Agent instance

    Returns
    -------
    tuple
        (databroker, None) if available, (None, error_message) if not
    """
    if not hasattr(agent, 'databroker') or agent.databroker is None:
        return None, "No databroker connection. Data persistence not available."
    return agent.databroker, None


def get_timestamp_string() -> str:
    """
    Get standard timestamp string for filenames

    Returns
    -------
    str
        Timestamp in YYYYMMDD_HHMMSS format
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable string

    Parameters
    ----------
    seconds : float
        Duration in seconds

    Returns
    -------
    str
        Human-readable duration (e.g., "2h 30m", "45s")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
