"""
CV Subagent Client

Async HTTP client for communicating with the CV Subagent service.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CVSubagentClient:
    """
    Async client for CV Subagent service

    Provides methods for:
    - Submitting analysis requests
    - Checking task status
    - Polling for results
    - Service health checks

    Usage
    -----
    ```python
    async with CVSubagentClient() as client:
        result = await client.analyze(
            intent="classify embryo stage",
            embryo_id="embryo_1",
        )
        print(f"Task ID: {result['task_id']}")

        # Poll for result
        status = await client.wait_for_result(result['task_id'])
        print(f"Result: {status['result']}")
    ```
    """

    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 8100
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """
        Initialize client

        Parameters
        ----------
        host : str
            CV subagent service host
        port : int
            CV subagent service port
        timeout : float
            Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"
        self._client = None

    async def __aenter__(self):
        """Enter async context"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context"""
        await self.close()

    async def connect(self):
        """Connect to service"""
        try:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
            logger.info(f"Connected to CV subagent at {self.base_url}")
        except ImportError:
            raise RuntimeError("httpx package required: pip install httpx")

    async def close(self):
        """Close connection"""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("CV subagent client closed")

    def _ensure_connected(self):
        """Ensure client is connected"""
        if self._client is None:
            raise RuntimeError("Client not connected. Use 'async with' or call connect()")

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    async def analyze(
        self,
        intent: str,
        embryo_id: str,
        timepoints: Optional[List[int]] = None,
        volume_uids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: str = "normal",
    ) -> Dict[str, Any]:
        """
        Submit an analysis request

        Parameters
        ----------
        intent : str
            High-level intent describing what to analyze
        embryo_id : str
            ID of embryo to analyze
        timepoints : list, optional
            Specific timepoints to analyze
        volume_uids : list, optional
            Specific volume UIDs to analyze
        context : dict, optional
            Additional context for the agent
        priority : str
            Task priority (low, normal, high, urgent)

        Returns
        -------
        dict
            Response containing task_id, status, and plan
        """
        self._ensure_connected()

        response = await self._client.post(
            "/api/v1/analyze",
            json={
                "intent": intent,
                "embryo_id": embryo_id,
                "timepoints": timepoints,
                "volume_uids": volume_uids,
                "context": context or {},
                "priority": priority,
            },
        )
        response.raise_for_status()
        return response.json()

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """
        Get task status

        Parameters
        ----------
        task_id : str
            Task ID to check

        Returns
        -------
        dict
            Task status including result if completed
        """
        self._ensure_connected()

        response = await self._client.get(f"/api/v1/tasks/{task_id}")
        response.raise_for_status()
        return response.json()

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task

        Parameters
        ----------
        task_id : str
            Task ID to cancel

        Returns
        -------
        bool
            True if cancelled successfully
        """
        self._ensure_connected()

        response = await self._client.delete(f"/api/v1/tasks/{task_id}")
        if response.status_code == 200:
            return True
        elif response.status_code == 400:
            return False
        response.raise_for_status()
        return False

    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List tasks

        Parameters
        ----------
        status : str, optional
            Filter by status
        limit : int
            Maximum tasks to return
        offset : int
            Pagination offset

        Returns
        -------
        dict
            Task list with pagination info
        """
        self._ensure_connected()

        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = await self._client.get("/api/v1/tasks", params=params)
        response.raise_for_status()
        return response.json()

    # =========================================================================
    # Polling and Waiting
    # =========================================================================

    async def wait_for_result(
        self,
        task_id: str,
        poll_interval: float = 1.0,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Wait for task to complete

        Parameters
        ----------
        task_id : str
            Task ID to wait for
        poll_interval : float
            Seconds between status checks
        timeout : float, optional
            Maximum time to wait (None for no timeout)

        Returns
        -------
        dict
            Final task status with result

        Raises
        ------
        TimeoutError
            If timeout exceeded
        RuntimeError
            If task failed
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            status = await self.get_task(task_id)

            if status["status"] == "completed":
                return status
            elif status["status"] == "failed":
                raise RuntimeError(f"Task failed: {status.get('error', 'Unknown error')}")
            elif status["status"] == "cancelled":
                raise RuntimeError("Task was cancelled")

            # Check timeout
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Timeout waiting for task {task_id}")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    # =========================================================================
    # Service Methods
    # =========================================================================

    async def get_status(self) -> Dict[str, Any]:
        """
        Get service status

        Returns
        -------
        dict
            Service status including GPU info and capabilities
        """
        self._ensure_connected()

        response = await self._client.get("/api/v1/status")
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> Dict[str, Any]:
        """
        Check service health

        Returns
        -------
        dict
            Health status
        """
        self._ensure_connected()

        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def is_healthy(self) -> bool:
        """
        Check if service is healthy

        Returns
        -------
        bool
            True if service is running and healthy
        """
        try:
            health = await self.health_check()
            return health.get("healthy", False)
        except Exception:
            return False


# Global client instance (for convenience)
_global_client: Optional[CVSubagentClient] = None


def get_cv_client() -> CVSubagentClient:
    """Get or create global CV client"""
    global _global_client
    if _global_client is None:
        _global_client = CVSubagentClient()
    return _global_client


def set_cv_client(client: CVSubagentClient):
    """Set global CV client"""
    global _global_client
    _global_client = client
