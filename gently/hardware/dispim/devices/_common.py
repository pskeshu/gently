"""
Common utilities shared across device modules.
"""

import logging

logger = logging.getLogger(__name__)


def _safe_obtain(obj):
    """Pass-through for MMCore data.

    With direct MMCore (in-process), arrays are already local numpy arrays.
    This function exists for API compatibility - it simply returns the object.
    """
    return obj
