"""
Tool Registry for CV Agent

Provides registration and management of CV tools available to the agent.
"""

import functools
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, get_type_hints

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of CV tools"""
    DATA_ACCESS = auto()
    PREPARATION = auto()
    SEGMENTATION = auto()
    ANALYSIS = auto()
    VISION = auto()
    CLASSICAL_CV = auto()
    IO = auto()


@dataclass
class ToolDefinition:
    """Definition of a registered tool"""
    name: str
    description: str
    category: ToolCategory
    function: Callable
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    returns: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    requires_gpu: bool = False


def cv_tool(
    name: str,
    description: str,
    category: ToolCategory,
    requires_gpu: bool = False,
    examples: Optional[List[str]] = None,
):
    """
    Decorator to register a function as a CV tool

    Parameters
    ----------
    name : str
        Tool name
    description : str
        Tool description
    category : ToolCategory
        Tool category
    requires_gpu : bool
        Whether tool requires GPU
    examples : list, optional
        Example usage strings

    Example
    -------
    @cv_tool(
        name="detect_embryo_roi",
        description="Find embryo bounding box for proper framing",
        category=ToolCategory.PREPARATION,
    )
    def detect_embryo_roi(volume: np.ndarray) -> dict:
        ...
    """
    def decorator(func: Callable) -> Callable:
        # Extract parameter info from type hints
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        sig = inspect.signature(func)

        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_info = {
                "type": str(hints.get(param_name, "any")),
                "required": param.default == inspect.Parameter.empty,
            }
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            parameters[param_name] = param_info

        # Store tool info on function
        func._cv_tool = ToolDefinition(
            name=name,
            description=description,
            category=category,
            function=func,
            parameters=parameters,
            returns=str(hints.get('return', '')),
            examples=examples or [],
            requires_gpu=requires_gpu,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._cv_tool = func._cv_tool
        return wrapper

    return decorator


class CVToolRegistry:
    """
    Registry of CV tools available to the agent

    Manages tool registration, lookup, and schema generation
    for Claude tool calling.
    """

    def __init__(
        self,
        data_store_url: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize tool registry

        Parameters
        ----------
        data_store_url : str, optional
            URL for data store access
        config : CVSubagentConfig, optional
            Configuration
        """
        self.data_store_url = data_store_url
        self.config = config

        self._tools: Dict[str, ToolDefinition] = {}

        # Register built-in tools
        self._register_builtin_tools()

    def register(self, tool_def: ToolDefinition):
        """Register a tool"""
        self._tools[tool_def.name] = tool_def
        logger.debug(f"Registered tool: {tool_def.name}")

    def register_function(self, func: Callable):
        """Register a decorated function"""
        if hasattr(func, '_cv_tool'):
            self.register(func._cv_tool)
        else:
            raise ValueError(f"Function {func.__name__} is not decorated with @cv_tool")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get tool by name"""
        return self._tools.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """List all tools, optionally filtered by category"""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools

    def get_claude_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get tools schema for Claude API

        Returns list of tool definitions in Claude's format.
        """
        schemas = []
        for tool in self._tools.values():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }

            for param_name, param_info in tool.parameters.items():
                prop = {
                    "type": self._python_type_to_json(param_info["type"]),
                    "description": f"Parameter: {param_name}",
                }
                schema["input_schema"]["properties"][param_name] = prop

                if param_info.get("required", False):
                    schema["input_schema"]["required"].append(param_name)

            schemas.append(schema)

        return schemas

    def _python_type_to_json(self, type_str: str) -> str:
        """Convert Python type string to JSON schema type"""
        type_map = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "None": "null",
        }
        # Handle generic types like List[int], Optional[str]
        for py_type, json_type in type_map.items():
            if py_type in type_str:
                return json_type
        return "string"

    async def execute(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name

        Parameters
        ----------
        name : str
            Tool name
        **kwargs
            Tool arguments

        Returns
        -------
        any
            Tool result
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        logger.info(f"Executing tool: {name}")

        # Execute the tool function
        result = tool.function(**kwargs)

        # Handle async functions
        if inspect.iscoroutine(result):
            result = await result

        return result

    def _register_builtin_tools(self):
        """Register tools from implemented tool modules"""
        # Import tool modules to discover decorated functions
        try:
            from . import data_access
            from . import preparation
            from . import vision
            from . import segmentation
            from . import morphology
            from . import tracking

            modules = [data_access, preparation, vision, segmentation, morphology, tracking]

            for module in modules:
                self._register_module_tools(module)

            logger.info(f"Registered {len(self._tools)} tools from modules")

        except ImportError as e:
            logger.warning(f"Could not import tool modules: {e}")
            # Register minimal placeholders for testing
            self._register_placeholder_tools()

    def _register_module_tools(self, module):
        """Register all @cv_tool decorated functions from a module"""
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and hasattr(obj, '_cv_tool'):
                self.register(obj._cv_tool)
                logger.debug(f"Registered {obj._cv_tool.name} from {module.__name__}")

    def _register_placeholder_tools(self):
        """Register placeholder tools for testing when modules unavailable"""
        @cv_tool(
            name="cellpose_segment_3d",
            description="Segment cells/nuclei in 3D volume using Cellpose",
            category=ToolCategory.SEGMENTATION,
            requires_gpu=True,
        )
        def cellpose_segment_3d(
            volume_uid: str,
            model_type: str = "cyto2",
            diameter: Optional[float] = None,
        ) -> dict:
            """Run Cellpose 3D segmentation (placeholder)"""
            return {
                "num_cells": 0,
                "mask_uid": None,
                "message": "Cellpose not yet implemented - will use GPU",
            }

        @cv_tool(
            name="stardist_segment_3d",
            description="Segment nuclei in 3D volume using StarDist",
            category=ToolCategory.SEGMENTATION,
            requires_gpu=True,
        )
        def stardist_segment_3d(volume_uid: str) -> dict:
            """Run StarDist 3D segmentation (placeholder)"""
            return {
                "num_nuclei": 0,
                "mask_uid": None,
                "message": "StarDist not yet implemented - will use GPU",
            }

        @cv_tool(
            name="measure_morphology",
            description="Measure shape metrics from segmentation masks",
            category=ToolCategory.ANALYSIS,
        )
        def measure_morphology(masks_uid: str) -> dict:
            """Measure morphology from masks (placeholder)"""
            return {
                "elongation": 1.0,
                "circularity": 1.0,
                "solidity": 1.0,
                "message": "Morphology measurement not yet implemented",
            }

        @cv_tool(
            name="track_objects",
            description="Track objects across multiple timepoints",
            category=ToolCategory.ANALYSIS,
        )
        def track_objects(masks_uids: list) -> dict:
            """Track objects across timepoints (placeholder)"""
            return {
                "num_tracks": 0,
                "division_events": [],
                "message": "Object tracking not yet implemented",
            }

        # Register placeholders
        for func in [cellpose_segment_3d, stardist_segment_3d, measure_morphology, track_objects]:
            self.register_function(func)

        logger.info(f"Registered {len(self._tools)} placeholder tools")
