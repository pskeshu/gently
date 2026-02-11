"""
Combined capabilities — all agent capabilities in one interface.
"""

import logging
from typing import Any, Dict, Optional

from .hardware import HardwareCapability, Result
from .perception import PerceptionCapability
from .interaction import InteractionCapability

logger = logging.getLogger(__name__)


class Capabilities:
    """
    All agent capabilities combined.

    Provides a unified interface for the daemon to execute actions.
    """

    def __init__(
        self,
        device_client: Optional[Any] = None,
        perception_manager: Optional[Any] = None,
        message_handler: Optional[Any] = None,
        notifier: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        device_client : Any, optional
            HTTP client for device layer
        perception_manager : Any, optional
            Perception system manager
        message_handler : callable, optional
            Function to handle agent messages
        notifier : Any, optional
            Notification system
        """
        self.hardware = HardwareCapability(device_client)
        self.perception = PerceptionCapability(perception_manager)
        self.interaction = InteractionCapability(message_handler, notifier)

    async def execute(self, action_type: str, params: Dict[str, Any]) -> Result:
        """
        Execute an action.

        Parameters
        ----------
        action_type : str
            Type of action (move, image, perceive, speak, ask, notify, observe)
        params : dict
            Action parameters

        Returns
        -------
        Result
            Result of the action
        """
        logger.debug(f"Executing action: {action_type} with {params}")

        try:
            match action_type:
                case "move":
                    target = params.get("target", params.get("embryo_id"))
                    return await self.hardware.move_to_embryo(target)

                case "image" | "acquire":
                    return await self.hardware.acquire_volume(**params)

                case "perceive" | "classify":
                    result = await self.perception.classify_stage(**params)
                    return Result(
                        success=True,
                        data={
                            "stage": result.stage,
                            "confidence": result.confidence,
                            "reasoning": result.reasoning,
                        },
                    )

                case "observe":
                    # Observation is a combination of image + perceive
                    target = params.get("target", params.get("embryo_id"))
                    if target:
                        move_result = await self.hardware.move_to_embryo(target)
                        if not move_result.success:
                            return move_result

                    # Get status instead of full acquisition for quick observation
                    status = await self.hardware.get_status()
                    return Result(
                        success=True,
                        data={"status": status.status, "embryo": status.current_embryo},
                    )

                case "speak":
                    message = params.get("message", "")
                    priority = params.get("priority", "normal")
                    await self.interaction.speak(message, priority)
                    return Result(success=True)

                case "ask":
                    question = params.get("question", "")
                    options = params.get("options")
                    response = await self.interaction.ask(question, options)
                    return Result(success=True, data={"response": response})

                case "notify":
                    message = params.get("message", "")
                    await self.interaction.notify(message)
                    return Result(success=True)

                case "configure":
                    return await self.hardware.configure(**params)

                case _:
                    logger.warning(f"Unknown action type: {action_type}")
                    return Result(success=False, error=f"Unknown action: {action_type}")

        except Exception as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            return Result(success=False, error=str(e))

    @property
    def user_present(self) -> bool:
        """Check if user is present."""
        return self.interaction.user_present

    def set_user_present(self, present: bool):
        """Update user presence."""
        if present:
            self.interaction.on_user_arrives()
        else:
            self.interaction.on_user_leaves()
