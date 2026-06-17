"""
Gently - Main Entry Point

This is the unified entry point that integrates all system components:
- Core Infrastructure (DataStore, EventBus, Service)
- Session Management
- Analysis Pipelines
- Tool Registry
- Agent

Usage:
    from gently import Gently

    # Create system instance
    gently = Gently()

    # Access components
    gently.data_store      # UID-based data storage
    gently.event_bus       # Async event messaging
    gently.services        # Service registry
    gently.sessions        # Session management
    gently.tools           # Tool registry
    gently.pipelines       # Analysis pipeline builder

    # Start a session
    await gently.start_session(name="My Experiment")

    # Connect to microscope
    await gently.connect_microscope(host="localhost", port=18861)

    # Run analysis pipeline
    result = await gently.analyze(volume, pipeline="embryo_detection")
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ui.web.server import VisualizationServer

from .analysis import (
    Pipeline,
    PipelineBuilder,
    create_embryo_detection_pipeline,
    create_hatching_detection_pipeline,
    create_morphology_analysis_pipeline,
)
from .core import (
    EventBus,
    EventType,
    ServiceClient,
    ServiceInfo,
    ServiceRegistry,
    get_event_bus,
    get_service_registry,
)
from .core.file_store import FileStore
from .harness.tools.registry import ToolRegistry, get_tool_registry
from .log_config import configure_logging
from .settings import settings

logger = logging.getLogger(__name__)


class Gently:
    """
    Main entry point for the Gently microscope control system

    Integrates all components into a unified interface:
    - Data storage with UID-based lineage tracking
    - Event-driven architecture
    - Service discovery and communication
    - Session persistence
    - Composable analysis pipelines
    - Plugin-based tool system
    """

    def __init__(
        self,
        storage_path: Path = settings.storage.base_path,
    ):
        """
        Initialize the Gently system

        Parameters
        ----------
        storage_path : Path
            Base path for all data storage (default: D:/Gently)
        """
        configure_logging(level="INFO")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._event_bus = get_event_bus()
        self._services = get_service_registry()
        self._client = ServiceClient(self._services)

        # Initialize FileStore (unified storage)
        self._store = FileStore(self.storage_path)

        # Current session ID (set by start_session or resume_session)
        self._current_session_id: str | None = None

        # Initialize tool registry
        self._tools = get_tool_registry()

        # Pre-built pipelines
        self._pipelines: dict[str, Pipeline] = {
            "embryo_detection": create_embryo_detection_pipeline(),
            "hatching_detection": create_hatching_detection_pipeline(),
            "morphology_analysis": create_morphology_analysis_pipeline(),
        }

        # Agent instance (lazy loaded)
        self._agent = None

        # Visualization server (lazy loaded)
        self._viz_server: VisualizationServer | None = None

        # Register standard services
        self._register_standard_services()

        logger.info(f"Gently initialized with storage at {self.storage_path}")

    def _register_standard_services(self):
        """Register info for standard services"""
        # These are typically external services that may or may not be running
        standard_services = [
            ServiceInfo(
                name="microscope_server",
                service_type="rpc",
                host="localhost",
                port=18861,
                metadata={"description": "Main microscope control server"},
            ),
            ServiceInfo(
                name="sam_server",
                service_type="rpc",
                host="localhost",
                port=18862,
                metadata={"description": "SAM segmentation server"},
            ),
            ServiceInfo(
                name="queue_server",
                service_type="http",
                host="localhost",
                port=settings.network.device_port,
                metadata={"description": "Bluesky queue server"},
            ),
        ]

        for info in standard_services:
            self._services.register_info(info)

    # =========================================================================
    # Properties for accessing components
    # =========================================================================

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus"""
        return self._event_bus

    @property
    def services(self) -> ServiceRegistry:
        """Access the service registry"""
        return self._services

    @property
    def client(self) -> ServiceClient:
        """Access the service client"""
        return self._client

    @property
    def store(self) -> FileStore:
        """Access the unified data store"""
        return self._store

    @property
    def tools(self) -> ToolRegistry:
        """Access the tool registry"""
        return self._tools

    @property
    def pipelines(self) -> dict[str, Pipeline]:
        """Access pre-built pipelines"""
        return self._pipelines

    # =========================================================================
    # Session Management
    # =========================================================================

    async def start_session(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        """
        Start a new session

        Parameters
        ----------
        name : str, optional
            Human-readable session name
        description : str, optional
            Session description (stored in metadata)

        Returns
        -------
        str
            Session ID
        """
        import uuid

        session_id = str(uuid.uuid4())[:8]
        self._store.create_session(session_id, name=name)
        self._current_session_id = session_id

        self._event_bus.publish(
            EventType.SESSION_STARTED,
            {"session_id": session_id, "name": name},
            source="gently",
        )

        logger.info(f"Started session: {session_id}")
        return session_id

    async def resume_session(self, session_id: str) -> bool:
        """
        Resume an existing session

        Parameters
        ----------
        session_id : str
            Session ID to resume

        Returns
        -------
        bool
            True if resumed successfully
        """
        session = self._store.get_session(session_id)
        if session:
            self._current_session_id = session_id
            self._event_bus.publish(
                EventType.SESSION_RESTORED,
                {"session_id": session_id},
                source="gently",
            )
            logger.info(f"Resumed session: {session_id}")
            return True
        return False

    def list_sessions(self) -> list[Any]:
        """List available sessions"""
        return self._store.list_sessions()

    # =========================================================================
    # Service Connection
    # =========================================================================

    async def connect_microscope(
        self,
        host: str = "localhost",
        port: int = 18861,
    ) -> bool:
        """
        Connect to the microscope server

        Parameters
        ----------
        host : str
            Server hostname
        port : int
            Server port

        Returns
        -------
        bool
            True if connected successfully
        """
        # Update service info
        info = self._services.get_info("microscope_server")
        if info:
            info.host = host
            info.port = port

        try:
            await self._client.connect("microscope_server")
            logger.info(f"Connected to microscope server at {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to microscope: {e}")
            return False

    async def connect_sam_server(
        self,
        host: str = "localhost",
        port: int = 18862,
    ) -> bool:
        """
        Connect to the SAM segmentation server

        Parameters
        ----------
        host : str
            Server hostname
        port : int
            Server port

        Returns
        -------
        bool
            True if connected successfully
        """
        info = self._services.get_info("sam_server")
        if info:
            info.host = host
            info.port = port

        try:
            await self._client.connect("sam_server")
            logger.info(f"Connected to SAM server at {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to SAM server: {e}")
            return False

    # =========================================================================
    # Analysis
    # =========================================================================

    async def analyze(
        self,
        data: Any,
        pipeline: str = "embryo_detection",
        context: dict | None = None,
    ) -> Any:
        """
        Run analysis pipeline on data

        Parameters
        ----------
        data : any
            Input data (volume, image, etc.)
        pipeline : str
            Name of pipeline to run
        context : dict, optional
            Additional context (embryo_id, timepoint, etc.)

        Returns
        -------
        AnalysisResult
            Pipeline result with lineage tracking
        """
        if pipeline not in self._pipelines:
            raise ValueError(
                f"Unknown pipeline: {pipeline}. Available: {list(self._pipelines.keys())}"
            )

        pipe = self._pipelines[pipeline]

        result = await pipe.execute(data, context=context)

        self._event_bus.publish(
            EventType.ANALYSIS_COMPLETED,
            {
                "pipeline": pipeline,
                "result_uid": result.uid,
                "success": result.success,
            },
            source="gently",
        )

        return result

    def create_pipeline(self, name: str) -> PipelineBuilder:
        """
        Create a new analysis pipeline

        Parameters
        ----------
        name : str
            Pipeline name

        Returns
        -------
        PipelineBuilder
            Builder for constructing pipeline
        """
        return PipelineBuilder(name)

    def register_pipeline(self, name: str, pipeline: Pipeline):
        """Register a custom pipeline"""
        self._pipelines[name] = pipeline

    # =========================================================================
    # Event Handling
    # =========================================================================

    def on(self, event_type: EventType, handler):
        """
        Subscribe to events

        Parameters
        ----------
        event_type : EventType
            Event type to subscribe to
        handler : callable
            Handler function

        Returns
        -------
        callable
            Unsubscribe function
        """
        return self._event_bus.subscribe(event_type, handler)

    def emit(self, event_type: EventType, data: dict):
        """Emit an event"""
        self._event_bus.publish(event_type, data, source="gently")

    # =========================================================================
    # Agent Access
    # =========================================================================

    def get_agent(self, **kwargs):
        """
        Get or create the agent instance

        Parameters
        ----------
        **kwargs
            Arguments passed to MicroscopyAgent constructor

        Returns
        -------
        MicroscopyAgent
            The agent instance
        """
        if self._agent is None:
            from .app.agent import MicroscopyAgent

            self._agent = MicroscopyAgent(
                storage_path=self.storage_path, store=self._store, **kwargs
            )
        return self._agent

    # =========================================================================
    # Visualization Server
    # =========================================================================

    async def start_visualization_server(self, port: int = settings.network.viz_port):
        """
        Start the web-based visualization server

        Parameters
        ----------
        port : int
            Server port (default: 8080)

        Returns
        -------
        VisualizationServer
            Running server instance
        """
        if self._viz_server is None:
            from .ui.web.server import VisualizationServer

            self._viz_server = VisualizationServer(
                port=port,
                data_store=None,  # Legacy DataStore removed; viz uses event bus
                event_bus=self._event_bus,
            )
            await self._viz_server.start()
            logger.info(f"Visualization server started on port {port}")
        return self._viz_server

    async def push_image(
        self,
        array,
        uid: str,
        data_type: str = "image",
        metadata: dict | None = None,
    ):
        """
        Push an image to the visualization server

        Parameters
        ----------
        array : np.ndarray
            Image array
        uid : str
            Unique identifier
        data_type : str
            Type of image
        metadata : dict, optional
            Additional metadata
        """
        if self._viz_server:
            await self._viz_server.push_image(array, uid, data_type, metadata)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def shutdown(self):
        """Shutdown the system cleanly"""
        logger.info("Shutting down Gently...")

        # Stop visualization server
        if self._viz_server:
            await self._viz_server.stop()
            self._viz_server = None

        # Disconnect services
        await self._client.disconnect_all()

        # Stop any running services
        await self._services.stop_all()

        if self._current_session_id:
            self._event_bus.publish(
                EventType.SESSION_ENDED,
                {"session_id": self._current_session_id},
                source="gently",
            )

        # Close store
        self._store.close()

        logger.info("Gently shutdown complete")


# Convenience function
def create_gently(**kwargs) -> Gently:
    """Create a Gently instance with default settings"""
    return Gently(**kwargs)
