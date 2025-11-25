"""
Session management for Microscopy Copilot

Provides persistence and resume capabilities for copilot sessions.
"""

from .state import SessionState, ConversationMessage
from .manager import SessionManager

__all__ = ['SessionState', 'ConversationMessage', 'SessionManager']
