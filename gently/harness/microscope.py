"""
Microscope — the abstract interface between tools and hardware.

A Microscope exposes named **plans** (move, acquire, snap, calibrate, detect)
and a single execute() method. Each hardware module implements the same plan
names with different internal devices and coordination.

Plans ARE capabilities — if a microscope lists "acquire" in its plans,
tools that need volume acquisition will work. No separate CAPABILITIES set.

Usage:
    result = await microscope.execute("acquire", num_slices=50, exposure_ms=10)
    result = await microscope.acquire(num_slices=50, exposure_ms=10)  # convenience
"""

import logging
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class Microscope:
    """
    Base class for all microscope implementations.

    Subclasses implement plans by defining ``_plan_<name>`` async methods.
    The ``plans`` property auto-discovers which plans are available.

    Example
    -------
    >>> class MyMicroscope(Microscope):
    ...     async def _plan_move(self, x, y, **kw):
    ...         # move the stage
    ...         return {"success": True}
    ...
    ...     async def _plan_acquire(self, **params):
    ...         # acquire a volume
    ...         return {"success": True, "volume": array}
    ...
    >>> m = MyMicroscope()
    >>> m.plans  # {'move', 'acquire'}
    >>> await m.execute("acquire", num_slices=50)
    """

    # Hardware description for LLM context (override in subclass or hardware module)
    DESCRIPTION: str = ""

    @property
    def plans(self) -> Set[str]:
        """Discover available plans by inspecting _plan_* methods."""
        return {
            name[6:]  # strip "_plan_" prefix
            for name in dir(self)
            if name.startswith("_plan_") and callable(getattr(self, name))
        }

    @property
    def is_connected(self) -> bool:
        """Whether the microscope is connected and ready."""
        return False

    async def connect(self) -> bool:
        """Connect to the microscope. Returns True on success."""
        return True

    async def disconnect(self) -> None:
        """Disconnect from the microscope."""
        pass

    async def execute(self, plan: str, **params) -> dict:
        """
        Execute a named plan with the given parameters.

        Parameters
        ----------
        plan : str
            Plan name (e.g., "move", "acquire", "snap", "calibrate", "detect").
        **params
            Hardware-specific parameters for the plan.

        Returns
        -------
        dict
            Result with at least ``{"success": bool}``.

        Raises
        ------
        ValueError
            If the plan is not supported.
        """
        handler = getattr(self, f"_plan_{plan}", None)
        if handler is None:
            raise ValueError(
                f"Plan '{plan}' not supported by {type(self).__name__}. "
                f"Available plans: {', '.join(sorted(self.plans))}"
            )
        return await handler(**params)

    # =========================================================================
    # Convenience methods — delegate to execute()
    # These provide readable call sites and type hints for common plans.
    # =========================================================================

    async def move(self, x: float, y: float) -> dict:
        """Move stage to (x, y) in µm."""
        return await self.execute("move", x=x, y=y)

    async def acquire(self, **params) -> dict:
        """Acquire a 3D volume. Params are hardware-specific."""
        return await self.execute("acquire", **params)

    async def snap(self, **params) -> dict:
        """Acquire a single 2D image. Params are hardware-specific."""
        return await self.execute("snap", **params)

    async def calibrate(self, **params) -> dict:
        """Run calibration. Params and results are hardware-specific."""
        return await self.execute("calibrate", **params)

    async def detect(self, **params) -> dict:
        """Detect samples in the field of view."""
        return await self.execute("detect", **params)

    async def get_position(self) -> dict:
        """Get current stage position."""
        return await self.execute("get_position")

    async def get_status(self) -> dict:
        """Get microscope status."""
        return await self.execute("status")
