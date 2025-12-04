"""
CV Subagent Service

An intelligent computer vision service that receives high-level intent
and autonomously determines which CV tools to use for C. elegans embryo analysis.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from gently.core.service import HTTPService, get_service_registry
from gently.core.event_bus import get_event_bus, EventType, Event

from .tasks.task_queue import TaskQueue, TaskPriority
from .api.routes import create_router
from .config import CVSubagentConfig, AutoAnalysisConfig

logger = logging.getLogger(__name__)


class CVSubagentService(HTTPService):
    """
    Computer Vision Subagent Service

    An AI-powered service that:
    - Receives high-level intent (e.g., "classify embryo 1")
    - Uses Claude to plan which CV tools to use
    - Orchestrates classical CV, Cellpose, StarDist, and Claude Vision
    - Returns enriched analysis results

    Features:
    - Intent-based API (/api/v1/analyze)
    - Async task queue with GPU awareness
    - Event bus integration for pub/sub results
    - Data store access for historical volumes
    """

    DEFAULT_PORT = 8100

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        anthropic_api_key: Optional[str] = None,
        data_store_url: Optional[str] = None,
        gpu_device: int = 0,
        config: Optional[CVSubagentConfig] = None,
    ):
        super().__init__(name="cv-subagent", host=host, port=port)

        # Configuration
        self.config = config or CVSubagentConfig()
        self.anthropic_api_key = anthropic_api_key or self.config.anthropic_api_key
        self.data_store_url = data_store_url or self.config.data_store_url
        self.gpu_device = gpu_device

        # Task queue for async processing
        self._task_queue: Optional[TaskQueue] = None

        # CV Agent (initialized on start)
        self._agent = None

        # Event bus for publishing results
        self._event_bus = get_event_bus()

        # Auto-analysis tracking
        self._last_auto_analysis: Dict[str, float] = {}  # embryo_id -> timestamp
        self._auto_analysis_enabled = self.config.auto_analysis.enabled

        # Event subscriptions (stored for cleanup)
        self._event_subscriptions = []

        # Metadata
        self._metadata = {
            'description': 'Computer Vision Subagent for C. elegans analysis',
            'capabilities': [
                'segmentation',
                'stage_classification',
                'cell_tracking',
                'anomaly_detection',
            ],
        }

    @property
    def task_queue(self) -> TaskQueue:
        """Get the task queue"""
        if self._task_queue is None:
            raise RuntimeError("Service not started - task queue not initialized")
        return self._task_queue

    @property
    def agent(self):
        """Get the CV agent"""
        if self._agent is None:
            raise RuntimeError("Service not started - agent not initialized")
        return self._agent

    async def on_start(self):
        """Initialize and start the service"""
        logger.info(f"Starting CV Subagent service on {self.host}:{self.port}")

        # Initialize task queue
        self._task_queue = TaskQueue(
            max_concurrent=2,
            event_bus=self._event_bus,
        )
        await self._task_queue.start()

        # Initialize CV Agent (lazy import to avoid circular deps)
        from .agent import CVAgent
        self._agent = CVAgent(
            anthropic_api_key=self.anthropic_api_key,
            data_store_url=self.data_store_url,
            task_queue=self._task_queue,
        )

        # Register with service registry
        try:
            registry = get_service_registry()
            registry.register(self)
        except Exception as e:
            logger.warning(f"Could not register with service registry: {e}")

        # Subscribe to volume acquisition events (for auto-processing)
        self._subscribe_to_events()

        # Call parent to start HTTP server
        await super().on_start()

        logger.info(f"CV Subagent service started on http://{self.host}:{self.port}")

    async def on_stop(self):
        """Stop the service"""
        logger.info("Stopping CV Subagent service")

        # Unsubscribe from events
        self._unsubscribe_from_events()

        # Stop task queue
        if self._task_queue:
            await self._task_queue.stop()

        # Unregister from service registry
        try:
            registry = get_service_registry()
            registry.unregister(self.name)
        except Exception:
            pass

        # Call parent to stop HTTP server
        await super().on_stop()

        logger.info("CV Subagent service stopped")

    def setup_routes(self, app):
        """Setup FastAPI routes"""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Create router with service reference
        router = create_router(self)
        app.include_router(router, prefix="/api/v1")

        # Health check endpoint
        @app.get("/health")
        async def health():
            return await self.health_check()

        # Root endpoint
        @app.get("/")
        async def root():
            return {
                "service": self.name,
                "version": "0.1.0",
                "status": self._state.name,
                "docs": f"http://{self.host}:{self.port}/docs",
            }

    async def health_check(self) -> Dict:
        """Extended health check with CV-specific info"""
        base_health = await super().health_check()

        # Add CV-specific health info
        base_health.update({
            'task_queue': {
                'active_tasks': len(self._task_queue._active) if self._task_queue else 0,
                'queued_tasks': len(self._task_queue._queue) if self._task_queue else 0,
            },
            'gpu_available': self._check_gpu_available(),
            'auto_analysis': {
                'enabled': self._auto_analysis_enabled,
                'embryos_tracked': len(self._last_auto_analysis),
            },
        })

        return base_health

    def _check_gpu_available(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _subscribe_to_events(self):
        """Subscribe to relevant events for auto-processing"""
        # Subscribe to volume acquisition for potential auto-analysis
        # This allows the CV agent to automatically process new volumes
        try:
            # Use subscribe_async for async handler
            unsubscribe = self._event_bus.subscribe_async(
                EventType.VOLUME_ACQUIRED,
                self._on_volume_acquired,
            )
            self._event_subscriptions.append(unsubscribe)
            logger.info("Subscribed to VOLUME_ACQUIRED events")
        except Exception as e:
            logger.warning(f"Could not subscribe to events: {e}")

    def _unsubscribe_from_events(self):
        """Unsubscribe from all events"""
        for unsubscribe in self._event_subscriptions:
            try:
                unsubscribe()
            except Exception:
                pass
        self._event_subscriptions.clear()
        logger.info("Unsubscribed from all events")

    async def _on_volume_acquired(self, event: Event):
        """
        Handle new volume acquisition events

        If auto-processing is enabled, queue the volume for analysis.
        """
        if not self._auto_analysis_enabled:
            logger.debug(f"Auto-analysis disabled, ignoring volume: {event.data}")
            return

        data = event.data
        volume_uid = data.get("uid") or data.get("volume_uid")
        embryo_id = data.get("embryo_id")
        timepoint = data.get("timepoint")

        if not volume_uid:
            logger.debug("No volume UID in event, skipping auto-analysis")
            return

        # Check embryo filter
        auto_config = self.config.auto_analysis
        if auto_config.embryo_filter and embryo_id:
            if embryo_id not in auto_config.embryo_filter:
                logger.debug(f"Embryo {embryo_id} not in filter, skipping")
                return

        # Rate limiting per embryo
        if embryo_id:
            last_time = self._last_auto_analysis.get(embryo_id, 0)
            elapsed = time.time() - last_time
            if elapsed < auto_config.min_interval_seconds:
                logger.debug(
                    f"Rate limit: {elapsed:.1f}s < {auto_config.min_interval_seconds}s "
                    f"for embryo {embryo_id}, skipping"
                )
                return
            self._last_auto_analysis[embryo_id] = time.time()

        logger.info(
            f"Auto-analyzing volume {volume_uid} "
            f"(embryo={embryo_id}, timepoint={timepoint})"
        )

        # Build auto-analysis intent based on config
        intent_parts = []
        if auto_config.auto_segment:
            intent_parts.append("segment cells/nuclei")
        if auto_config.auto_stage:
            intent_parts.append("classify developmental stage")

        if not intent_parts:
            return

        intent = f"For volume {volume_uid}: " + " and ".join(intent_parts)

        try:
            # Queue the analysis task
            result = await self._agent.submit_task(
                intent=intent,
                embryo_id=embryo_id or "unknown",
                volume_uids=[volume_uid],
                timepoints=[timepoint] if timepoint is not None else None,
                context={
                    "auto_triggered": True,
                    "source_event": event.event_id,
                    "segmentation_model": auto_config.segmentation_model,
                },
            )
            logger.info(f"Auto-analysis task queued: {result.get('task_id')}")

        except Exception as e:
            logger.error(f"Failed to queue auto-analysis: {e}")

    def set_auto_analysis_enabled(self, enabled: bool):
        """Enable or disable auto-analysis at runtime"""
        self._auto_analysis_enabled = enabled
        logger.info(f"Auto-analysis {'enabled' if enabled else 'disabled'}")

    def get_auto_analysis_config(self) -> Dict:
        """Get current auto-analysis configuration"""
        return {
            "enabled": self._auto_analysis_enabled,
            "auto_segment": self.config.auto_analysis.auto_segment,
            "auto_stage": self.config.auto_analysis.auto_stage,
            "auto_track": self.config.auto_analysis.auto_track,
            "segmentation_model": self.config.auto_analysis.segmentation_model,
            "min_interval_seconds": self.config.auto_analysis.min_interval_seconds,
            "embryo_filter": self.config.auto_analysis.embryo_filter,
        }

    async def analyze(
        self,
        intent: str,
        embryo_id: str,
        timepoints: Optional[list] = None,
        volume_uids: Optional[list] = None,
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        Submit an analysis request

        Parameters
        ----------
        intent : str
            High-level intent (e.g., "classify embryo stage")
        embryo_id : str
            ID of the embryo to analyze
        timepoints : list, optional
            Specific timepoints to analyze
        volume_uids : list, optional
            Specific volume UIDs to analyze
        context : dict, optional
            Additional context for the agent

        Returns
        -------
        dict
            Task submission result with task_id and plan
        """
        return await self._agent.submit_task(
            intent=intent,
            embryo_id=embryo_id,
            timepoints=timepoints,
            volume_uids=volume_uids,
            context=context or {},
        )

    async def get_task_status(self, task_id: str) -> Dict:
        """Get status of a task"""
        return await self._task_queue.get_status(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        return await self._task_queue.cancel(task_id)

    def get_service_status(self) -> Dict:
        """Get detailed service status"""
        return {
            'name': self.name,
            'version': '0.1.0',
            'state': self._state.name,
            'host': self.host,
            'port': self.port,
            'gpu_available': self._check_gpu_available(),
            'task_queue': {
                'active_tasks': len(self._task_queue._active) if self._task_queue else 0,
                'queued_tasks': len(self._task_queue._queue) if self._task_queue else 0,
                'max_concurrent': self._task_queue.max_concurrent if self._task_queue else 0,
            },
            'auto_analysis': self.get_auto_analysis_config(),
            'capabilities': self._metadata.get('capabilities', []),
        }
