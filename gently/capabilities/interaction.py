"""
Interaction capability — how the agent communicates with users.

Provides speaking, asking questions, and notifications.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message from the agent."""
    content: str
    priority: str = "normal"  # low, normal, high, urgent
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PendingQuestion:
    """A question waiting for user response."""
    question: str
    options: Optional[List[str]] = None
    callback: Optional[Callable] = None
    asked_at: datetime = field(default_factory=datetime.now)
    response: Optional[str] = None


class InteractionCapability:
    """
    Handles agent communication with users.

    Supports:
    - Speaking (output messages)
    - Asking questions (with optional choices)
    - Notifications (for when user is away)
    """

    def __init__(
        self,
        message_handler: Optional[Callable] = None,
        notifier: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        message_handler : callable, optional
            Function to handle messages: def handler(message: Message)
        notifier : Any, optional
            Notification system for alerts
        """
        self.message_handler = message_handler
        self.notifier = notifier

        # State
        self.user_present = False
        self.pending_messages: List[Message] = []
        self.pending_questions: List[PendingQuestion] = []
        self.message_history: List[Message] = []

    async def speak(
        self,
        message: str,
        priority: str = "normal",
    ):
        """
        Say something to the user.

        If user is not present, message is queued and/or notified.

        Parameters
        ----------
        message : str
            What to say
        priority : str
            Message priority (low, normal, high, urgent)
        """
        msg = Message(content=message, priority=priority)
        self.message_history.append(msg)

        if self.user_present:
            # Deliver immediately
            if self.message_handler:
                await self._call_handler(self.message_handler, msg)
            else:
                logger.info(f"Agent says: {message}")
        else:
            # Queue for later
            self.pending_messages.append(msg)

            # High priority messages get notifications
            if priority in ("high", "urgent"):
                await self.notify(message)

    async def ask(
        self,
        question: str,
        options: Optional[List[str]] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Ask the user a question.

        Parameters
        ----------
        question : str
            The question to ask
        options : List[str], optional
            Predefined options (if any)
        timeout : float
            Timeout in seconds

        Returns
        -------
        str or None
            User's response, or None if timeout/not present
        """
        if not self.user_present:
            logger.info(f"User not present, can't ask: {question}")
            return None

        pending = PendingQuestion(question=question, options=options)
        self.pending_questions.append(pending)

        # Signal question to handler
        if self.message_handler:
            question_msg = Message(
                content=f"Question: {question}"
                + (f" [{', '.join(options)}]" if options else ""),
                priority="high",
            )
            await self._call_handler(self.message_handler, question_msg)
        else:
            logger.info(f"Agent asks: {question}")

        # Wait for response (simplified - actual implementation would use events)
        try:
            start = asyncio.get_event_loop().time()
            while pending.response is None:
                if asyncio.get_event_loop().time() - start > timeout:
                    break
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

        self.pending_questions.remove(pending)
        return pending.response

    async def notify(self, message: str):
        """
        Send a notification (for when user is away).

        Parameters
        ----------
        message : str
            Notification content
        """
        logger.info(f"Notification: {message}")

        if self.notifier:
            try:
                await self._call_async_or_sync(
                    self.notifier.send,
                    message=message,
                )
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def on_user_arrives(self):
        """Handle user arrival."""
        self.user_present = True
        logger.info("User arrived")

        # Deliver pending messages
        if self.pending_messages and self.message_handler:
            for msg in self.pending_messages:
                asyncio.create_task(self._call_handler(self.message_handler, msg))
            self.pending_messages.clear()

    def on_user_leaves(self):
        """Handle user departure."""
        self.user_present = False
        logger.info("User left")

    def provide_answer(self, answer: str):
        """
        Provide an answer to a pending question.

        Parameters
        ----------
        answer : str
            The user's answer
        """
        if self.pending_questions:
            self.pending_questions[0].response = answer

    async def _call_handler(self, handler: Callable, message: Message):
        """Call message handler, handling both sync and async."""
        await self._call_async_or_sync(handler, message)

    async def _call_async_or_sync(self, func: Callable, *args, **kwargs):
        """Call a function whether it's sync or async."""
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            await result
