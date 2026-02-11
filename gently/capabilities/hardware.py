"""
Hardware capability — wraps the device layer.

Provides a clean interface for the agent to control microscope hardware.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Result:
    """Result of a hardware operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class MicroscopeStatus:
    """Current microscope status."""
    status: str  # "idle", "acquiring", "moving", "error"
    current_embryo: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    position_z: Optional[float] = None
    last_acquisition: Optional[str] = None


class HardwareCapability:
    """
    Wraps the existing device layer for agent use.

    Provides high-level operations like "move to embryo" and "acquire volume"
    rather than low-level motor commands.
    """

    def __init__(self, device_client: Optional[Any] = None):
        """
        Parameters
        ----------
        device_client : Any, optional
            HTTP client for device layer API. If None, operations are simulated.
        """
        self.client = device_client
        self._status = MicroscopeStatus(status="idle")

    async def move_to_embryo(self, embryo_id: str) -> Result:
        """
        Move to an embryo's position.

        Parameters
        ----------
        embryo_id : str
            ID of the embryo to move to

        Returns
        -------
        Result
            Success status and any error
        """
        logger.info(f"Moving to embryo: {embryo_id}")

        if self.client is None:
            # Simulated
            self._status.current_embryo = embryo_id
            return Result(success=True, data={"embryo_id": embryo_id})

        try:
            # Call device layer API
            response = await self.client.post(
                "/api/queue/item/add",
                json={
                    "plan": "move_to_embryo",
                    "kwargs": {"embryo_id": embryo_id},
                },
            )
            response.raise_for_status()
            self._status.current_embryo = embryo_id
            return Result(success=True, data=response.json())
        except Exception as e:
            logger.error(f"Move to embryo failed: {e}")
            return Result(success=False, error=str(e))

    async def acquire_volume(
        self,
        embryo_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timepoint: Optional[int] = None,
        **params,
    ) -> Result:
        """
        Acquire a volume.

        Parameters
        ----------
        embryo_id : str, optional
            Embryo to image
        session_id : str, optional
            Session ID for data storage
        timepoint : int, optional
            Timepoint number
        **params
            Additional acquisition parameters

        Returns
        -------
        Result
            Success status with volume path/data
        """
        logger.info(f"Acquiring volume: embryo={embryo_id}, tp={timepoint}")

        if self.client is None:
            # Simulated
            return Result(
                success=True,
                data={
                    "embryo_id": embryo_id,
                    "timepoint": timepoint,
                    "volume_path": f"/simulated/{embryo_id}_t{timepoint:04d}.tif",
                },
            )

        try:
            # Call device layer API
            plan_kwargs = {
                "embryo_id": embryo_id,
                "session_id": session_id,
                "timepoint": timepoint,
                **params,
            }
            response = await self.client.post(
                "/api/queue/item/add",
                json={
                    "plan": "acquire_volume",
                    "kwargs": plan_kwargs,
                },
            )
            response.raise_for_status()
            return Result(success=True, data=response.json())
        except Exception as e:
            logger.error(f"Acquire volume failed: {e}")
            return Result(success=False, error=str(e))

    async def get_status(self) -> MicroscopeStatus:
        """
        Get current microscope status.

        Returns
        -------
        MicroscopeStatus
            Current status
        """
        if self.client is None:
            return self._status

        try:
            response = await self.client.get("/api/status")
            response.raise_for_status()
            data = response.json()
            return MicroscopeStatus(
                status=data.get("status", "unknown"),
                current_embryo=data.get("current_embryo"),
                position_x=data.get("position", {}).get("x"),
                position_y=data.get("position", {}).get("y"),
                position_z=data.get("position", {}).get("z"),
            )
        except Exception as e:
            logger.error(f"Get status failed: {e}")
            return MicroscopeStatus(status="error")

    async def configure(self, **settings) -> Result:
        """
        Configure microscope settings.

        Parameters
        ----------
        **settings
            Settings to configure (exposure, laser_power, etc.)

        Returns
        -------
        Result
            Success status
        """
        logger.info(f"Configuring microscope: {settings}")

        if self.client is None:
            return Result(success=True, data=settings)

        try:
            # Different endpoints for different settings
            for key, value in settings.items():
                if key == "exposure":
                    await self.client.post(
                        "/api/camera/exposure",
                        json={"exposure_ms": value},
                    )
                elif key == "led":
                    await self.client.post(
                        "/api/led/set",
                        json={"state": value},
                    )
            return Result(success=True, data=settings)
        except Exception as e:
            logger.error(f"Configure failed: {e}")
            return Result(success=False, error=str(e))
