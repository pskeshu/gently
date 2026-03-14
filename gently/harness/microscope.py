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

HTTPMicroscope is the generic client for device layers that expose the
/api/microscope protocol. It discovers plans on connect and delegates
all execution to the device layer — no hardware-specific code needed.
"""

import logging
from pathlib import Path
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


class HTTPMicroscope(Microscope):
    """
    Generic HTTP client for any device layer that speaks /api/microscope.

    Discovers available plans on connect. Delegates all execution to the
    device layer — no hardware-specific code in the gently process.

    Usage
    -----
    >>> microscope = HTTPMicroscope("http://localhost:60610")
    >>> await microscope.connect()
    >>> microscope.plans  # discovered from device layer
    >>> result = await microscope.execute("acquire", num_slices=50)
    """

    def __init__(self, http_url: str):
        self.http_url = http_url
        self._session = None
        self._connected = False
        self._available_plans: Set[str] = set()

    @property
    def plans(self) -> Set[str]:
        return self._available_plans

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        """Connect and discover available plans from the device layer."""
        import aiohttp

        self._connected = False
        self._session = aiohttp.ClientSession()

        try:
            # Handshake: discover plans and description
            async with self._session.get(f"{self.http_url}/api/microscope") as resp:
                if resp.status != 200:
                    logger.warning("Microscope handshake failed: HTTP %d", resp.status)
                    return False
                info = await resp.json()

            self._available_plans = set(info.get("plans", []))
            self.DESCRIPTION = info.get("description", "")
            self._connected = True

            logger.info(
                "Connected to %s (%s) — plans: %s",
                info.get("display_name", "microscope"),
                self.http_url,
                ", ".join(sorted(self._available_plans)),
            )
            return True

        except Exception as e:
            logger.warning("Failed to connect to %s: %s", self.http_url, e)
            return False

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    async def execute(self, plan: str, **params) -> dict:
        """Execute a plan on the remote device layer."""
        if plan not in self._available_plans:
            raise ValueError(
                f"Plan '{plan}' not available. "
                f"Available: {', '.join(sorted(self._available_plans))}"
            )

        if not self._connected or not self._session:
            raise ConnectionError("Not connected. Call connect() first.")

        import aiohttp

        try:
            async with self._session.post(
                f"{self.http_url}/api/microscope/execute",
                json={"plan": plan, "params": params},
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                result = await resp.json()

            # Resolve file references (large arrays written to disk)
            if isinstance(result, dict):
                self._resolve_file_refs(result)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _is_file_ref(obj) -> bool:
        return isinstance(obj, dict) and obj.get("__file_ref__") is True

    def _resolve_file_refs(self, data: dict) -> None:
        """Resolve file references in-place (load TIFFs from staging dir)."""
        for key, val in list(data.items()):
            if self._is_file_ref(val):
                import tifffile
                path = Path(val["path"])
                data[key] = tifffile.imread(str(path))
                data[f"{key}_path"] = str(path)

    async def configure_session(self, volume_dir: str) -> dict:
        """Tell the device layer where to write staging TIFFs."""
        if not self._session:
            raise ConnectionError("Not connected.")
        async with self._session.post(
            f"{self.http_url}/session/configure",
            json={"volume_dir": volume_dir},
        ) as resp:
            return await resp.json()
