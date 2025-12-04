"""
Tool Registry for CV Agent

Provides registration and management of CV tools available to the agent.
Follows the same patterns as the copilot's tool_registry.py for consistent
Claude API schema generation.
"""

import asyncio
import functools
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Union,
    get_type_hints, get_origin, get_args
)

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
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # JSON schema type: "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolExample:
    """Example of when to use a tool"""
    user_query: str
    tool_input: Dict = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Definition of a registered tool"""
    name: str
    description: str
    category: ToolCategory
    function: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    returns: Optional[str] = None
    examples: List[ToolExample] = field(default_factory=list)
    requires_gpu: bool = False
    is_async: bool = False


def _python_type_to_json_schema(python_type) -> str:
    """Convert Python type hint to JSON schema type"""
    origin = get_origin(python_type)

    # Handle Optional[X] -> X (Union[X, None])
    if origin is Union:
        args = get_args(python_type)
        # Filter out NoneType
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _python_type_to_json_schema(non_none_args[0])

    # Handle List[X]
    if origin is list:
        return "array"

    # Handle Dict[K, V]
    if origin is dict:
        return "object"

    # Handle Tuple
    if origin is tuple:
        return "array"

    # Basic types
    if python_type is str:
        return "string"
    elif python_type is int:
        return "integer"
    elif python_type is float:
        return "number"
    elif python_type is bool:
        return "boolean"
    elif python_type is list:
        return "array"
    elif python_type is dict:
        return "object"

    # Default to string for complex types
    return "string"


def _parse_docstring_params(doc: str) -> Dict[str, str]:
    """
    Parse parameter descriptions from NumPy-style docstring.

    Parameters
    ----------
    doc : str
        The docstring to parse

    Returns
    -------
    dict
        Mapping of parameter names to their descriptions
    """
    param_docs = {}
    in_params = False
    current_param = None

    for line in doc.split('\n'):
        line_stripped = line.strip()

        if line_stripped.lower().startswith('parameters'):
            in_params = True
            continue

        if in_params:
            # Skip separator lines
            if line_stripped.startswith('---'):
                continue

            # Stop at Returns or other sections
            if line_stripped.lower().startswith(('returns', 'raises', 'examples', 'notes', 'see also')):
                in_params = False
                continue

            # New parameter definition: "param_name : type"
            if ' : ' in line_stripped:
                parts = line_stripped.split(' : ')
                current_param = parts[0].strip()
                param_docs[current_param] = ""
            elif current_param and line_stripped:
                # Continuation of description
                if param_docs[current_param]:
                    param_docs[current_param] += " " + line_stripped
                else:
                    param_docs[current_param] = line_stripped

    return param_docs


def _extract_parameters_from_function(func: Callable) -> List[ToolParameter]:
    """
    Extract parameter definitions from function signature and docstring.

    This mirrors the copilot's _extract_parameters_from_function() to ensure
    consistent parameter documentation in Claude tool schemas.

    Parameters
    ----------
    func : callable
        The function to extract parameters from

    Returns
    -------
    list of ToolParameter
        Extracted parameter definitions
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    doc = inspect.getdoc(func) or ""

    # Parse docstring for parameter descriptions
    param_docs = _parse_docstring_params(doc)

    parameters = []
    for param_name, param in sig.parameters.items():
        # Skip 'self' and 'context' (injected at runtime)
        if param_name in ('self', 'context'):
            continue

        python_type = hints.get(param_name, str)
        json_type = _python_type_to_json_schema(python_type)

        # Check if optional (has default or is Optional type)
        required = param.default is inspect.Parameter.empty
        default = None if param.default is inspect.Parameter.empty else param.default

        # Get description from docstring
        description = param_docs.get(param_name, f"The {param_name} parameter").strip()

        parameters.append(ToolParameter(
            name=param_name,
            type=json_type,
            description=description,
            required=required,
            default=default,
        ))

    return parameters


def cv_tool(
    name: str,
    description: str,
    category: ToolCategory,
    requires_gpu: bool = False,
    examples: Optional[List[ToolExample]] = None,
    parameters: Optional[List[ToolParameter]] = None,
):
    """
    Decorator to register a function as a CV tool.

    Parameters are automatically extracted from the function's docstring
    (NumPy-style) unless explicitly provided via the `parameters` argument.

    Parameters
    ----------
    name : str
        Tool name (used in Claude API)
    description : str
        Tool description shown to Claude
    category : ToolCategory
        Tool category for organization
    requires_gpu : bool
        Whether tool requires GPU acceleration
    examples : list of ToolExample, optional
        Usage examples showing when to call this tool
    parameters : list of ToolParameter, optional
        Explicit parameter definitions (overrides docstring extraction).
        Use this when you need enum constraints or custom descriptions.

    Example
    -------
    @cv_tool(
        name="detect_embryo_roi",
        description="Find embryo bounding box for proper framing",
        category=ToolCategory.PREPARATION,
        examples=[
            ToolExample("Find the embryo in volume xyz", {"volume_uid": "xyz"}),
        ],
    )
    def detect_embryo_roi(volume_uid: str, method: str = "threshold") -> dict:
        '''
        Find embryo bounding box in a volume

        Parameters
        ----------
        volume_uid : str
            UID of the volume to analyze
        method : str
            Detection method: "threshold", "otsu", or "adaptive"
        '''
        ...
    """
    def decorator(func: Callable) -> Callable:
        # Auto-extract parameters from docstring, or use explicit if provided
        tool_params = parameters if parameters is not None else _extract_parameters_from_function(func)

        # Get return type hint
        hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        returns = str(hints.get('return', ''))

        # Store tool info on function
        func._cv_tool = ToolDefinition(
            name=name,
            description=description,
            category=category,
            function=func,
            parameters=tool_params,
            returns=returns,
            examples=examples or [],
            requires_gpu=requires_gpu,
            is_async=asyncio.iscoroutinefunction(func),
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._cv_tool = func._cv_tool
        return wrapper

    return decorator


class CVToolRegistry:
    """
    Registry of CV tools available to the agent.

    Manages tool registration, lookup, and schema generation
    for Claude tool calling.
    """

    def __init__(
        self,
        data_store_url: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize tool registry.

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
        Get tools schema for Claude API.

        Returns list of tool definitions in Claude's format with:
        - Proper parameter descriptions (from docstrings)
        - JSON schema types
        - Enum constraints where specified
        - Default values
        - Examples embedded in description
        """
        schemas = []
        for tool in self._tools.values():
            properties = {}
            required = []

            for param in tool.parameters:
                prop = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.enum:
                    prop["enum"] = param.enum
                if param.default is not None:
                    prop["default"] = param.default

                properties[param.name] = prop

                if param.required:
                    required.append(param.name)

            # Build description with examples embedded
            desc = tool.description
            if tool.examples:
                example_inputs = [ex.tool_input for ex in tool.examples if ex.tool_input]
                if example_inputs:
                    desc += f"\n\nExample inputs: {example_inputs}"

            schemas.append({
                "name": tool.name,
                "description": desc,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            })

        return schemas

    async def execute(self, name: str, **kwargs) -> Any:
        """
        Execute a tool by name.

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
            examples=[
                ToolExample("Segment nuclei in volume abc", {"volume_uid": "abc", "model_type": "nuclei"}),
            ],
            parameters=[
                ToolParameter(name="volume_uid", type="string", description="UID of volume to segment", required=True),
                ToolParameter(name="model_type", type="string", description="Model type for segmentation",
                              required=False, default="cyto2", enum=["nuclei", "cyto2", "cyto"]),
                ToolParameter(name="diameter", type="number", description="Expected cell diameter in pixels",
                              required=False, default=None),
            ],
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
            examples=[
                ToolExample("Segment nuclei with StarDist", {"volume_uid": "abc"}),
            ],
        )
        def stardist_segment_3d(volume_uid: str) -> dict:
            """
            Run StarDist 3D segmentation (placeholder)

            Parameters
            ----------
            volume_uid : str
                UID of volume to segment
            """
            return {
                "num_nuclei": 0,
                "mask_uid": None,
                "message": "StarDist not yet implemented - will use GPU",
            }

        @cv_tool(
            name="measure_morphology",
            description="Measure shape metrics from segmentation masks",
            category=ToolCategory.ANALYSIS,
            examples=[
                ToolExample("Measure elongation of segmented cells", {"masks_uid": "mask_abc"}),
            ],
        )
        def measure_morphology(masks_uid: str) -> dict:
            """
            Measure morphology from masks (placeholder)

            Parameters
            ----------
            masks_uid : str
                UID of segmentation masks to analyze
            """
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
            examples=[
                ToolExample("Track cells across 5 timepoints", {"masks_uids": ["m1", "m2", "m3", "m4", "m5"]}),
            ],
        )
        def track_objects(masks_uids: list) -> dict:
            """
            Track objects across timepoints (placeholder)

            Parameters
            ----------
            masks_uids : list
                List of mask UIDs for consecutive timepoints
            """
            return {
                "num_tracks": 0,
                "division_events": [],
                "message": "Object tracking not yet implemented",
            }

        # Register placeholders
        for func in [cellpose_segment_3d, stardist_segment_3d, measure_morphology, track_objects]:
            self.register_function(func)

        logger.info(f"Registered {len(self._tools)} placeholder tools")
