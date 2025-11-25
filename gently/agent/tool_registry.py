"""
Tool Plugin System for Microscopy Copilot

Provides a decorator-based tool registration system that:
- Eliminates the rigid if/elif chain
- Auto-generates tool schemas from type hints
- Supports tool categories and filtering
- Enables runtime tool discovery and registration
- Integrates with the event bus for tool execution events
"""

import asyncio
import functools
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, List, Optional, Type, Union,
    get_type_hints, get_origin, get_args
)
import time

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for organizing tools"""
    ACQUISITION = auto()      # Volume/image acquisition
    MOVEMENT = auto()         # Stage movement, positioning
    CALIBRATION = auto()      # Calibration procedures
    ANALYSIS = auto()         # Image/volume analysis
    DETECTION = auto()        # Detector management
    EXPERIMENT = auto()       # Experiment state management
    EMBRYO = auto()           # Embryo-specific operations
    HARDWARE = auto()         # Direct hardware control
    DATA = auto()             # Data/Databroker operations
    UTILITY = auto()          # Utility functions


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # JSON schema type
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class ToolDefinition:
    """
    Complete definition of a tool

    Contains all metadata needed for:
    - Claude API tool schema generation
    - Validation
    - Documentation
    - Filtering/discovery
    """
    name: str
    description: str
    handler: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.UTILITY
    requires_microscope: bool = False
    is_async: bool = False
    tags: List[str] = field(default_factory=list)

    def to_claude_schema(self) -> Dict:
        """Generate Claude API tool schema"""
        properties = {}
        required = []

        for param in self.parameters:
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

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }


def _python_type_to_json_schema(python_type) -> str:
    """Convert Python type hint to JSON schema type"""
    origin = get_origin(python_type)

    # Handle Optional[X] -> X
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


def _extract_parameters_from_function(func: Callable) -> List[ToolParameter]:
    """Extract parameter definitions from function signature and type hints"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
    doc = inspect.getdoc(func) or ""

    # Parse docstring for parameter descriptions
    param_docs = {}
    in_params = False
    current_param = None
    for line in doc.split('\n'):
        line = line.strip()
        if line.lower().startswith('parameters'):
            in_params = True
            continue
        if in_params:
            if line.startswith('---'):
                continue
            if line.lower().startswith('returns'):
                in_params = False
                continue
            if ' : ' in line:
                parts = line.split(' : ')
                current_param = parts[0].strip()
                param_docs[current_param] = ""
            elif current_param and line:
                param_docs[current_param] += " " + line

    parameters = []
    for param_name, param in sig.parameters.items():
        # Skip 'self' and 'tool_input' (legacy pattern)
        if param_name in ('self', 'tool_input'):
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


class ToolRegistry:
    """
    Central registry for all copilot tools

    Features:
    - Decorator-based registration
    - Category filtering
    - Automatic schema generation
    - Tool execution with timing and events
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._context: Dict[str, Any] = {}  # Shared context (copilot, client, etc.)

    def set_context(self, key: str, value: Any):
        """Set shared context available to all tools"""
        self._context[key] = value

    def get_context(self, key: str) -> Any:
        """Get shared context"""
        return self._context.get(key)

    def register(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: ToolCategory = ToolCategory.UTILITY,
        requires_microscope: bool = False,
        tags: Optional[List[str]] = None,
        parameters: Optional[List[ToolParameter]] = None,
    ) -> Callable:
        """
        Decorator to register a function as a tool

        Usage:
            @registry.register(
                name="acquire_volume",
                description="Acquire a 3D volume for an embryo",
                category=ToolCategory.ACQUISITION,
                requires_microscope=True,
            )
            async def acquire_volume(embryo_id: str, num_slices: int = 50) -> str:
                ...

        Parameters
        ----------
        name : str, optional
            Tool name (defaults to function name)
        description : str, optional
            Tool description (defaults to function docstring)
        category : ToolCategory
            Tool category for filtering
        requires_microscope : bool
            Whether tool requires microscope connection
        tags : list of str, optional
            Additional tags for filtering
        parameters : list of ToolParameter, optional
            Explicit parameter definitions (auto-extracted if not provided)
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (inspect.getdoc(func) or "").split('\n')[0]

            # Extract or use provided parameters
            tool_params = parameters or _extract_parameters_from_function(func)

            # Create tool definition
            tool_def = ToolDefinition(
                name=tool_name,
                description=tool_desc,
                handler=func,
                parameters=tool_params,
                category=category,
                requires_microscope=requires_microscope,
                is_async=asyncio.iscoroutinefunction(func),
                tags=tags or [],
            )

            self._tools[tool_name] = tool_def
            logger.debug(f"Registered tool: {tool_name} ({category.name})")

            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                return await self.execute(tool_name, kwargs)

            return wrapper

        return decorator

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: ToolCategory = ToolCategory.UTILITY,
        requires_microscope: bool = False,
        tags: Optional[List[str]] = None,
    ):
        """
        Register an existing function as a tool (non-decorator form)

        Parameters
        ----------
        func : callable
            Function to register
        name : str, optional
            Tool name
        description : str, optional
            Tool description
        category : ToolCategory
            Tool category
        requires_microscope : bool
            Whether microscope is required
        tags : list of str, optional
            Additional tags
        """
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "").split('\n')[0]
        tool_params = _extract_parameters_from_function(func)

        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            handler=func,
            parameters=tool_params,
            category=category,
            requires_microscope=requires_microscope,
            is_async=asyncio.iscoroutinefunction(func),
            tags=tags or [],
        )

        self._tools[tool_name] = tool_def
        logger.debug(f"Registered tool: {tool_name} ({category.name})")

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name"""
        return self._tools.get(name)

    def list_all(self) -> List[ToolDefinition]:
        """List all registered tools"""
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """List tools in a category"""
        return [t for t in self._tools.values() if t.category == category]

    def list_by_tag(self, tag: str) -> List[ToolDefinition]:
        """List tools with a specific tag"""
        return [t for t in self._tools.values() if tag in t.tags]

    def list_available(self, has_microscope: bool = False) -> List[ToolDefinition]:
        """List tools available given current context"""
        tools = []
        for tool in self._tools.values():
            if tool.requires_microscope and not has_microscope:
                continue
            tools.append(tool)
        return tools

    def get_claude_schemas(self, has_microscope: bool = False) -> List[Dict]:
        """Get Claude API tool schemas for available tools"""
        return [
            tool.to_claude_schema()
            for tool in self.list_available(has_microscope)
        ]

    async def execute(self, tool_name: str, tool_input: Dict) -> str:
        """
        Execute a tool by name

        Parameters
        ----------
        tool_name : str
            Name of tool to execute
        tool_input : dict
            Tool input parameters

        Returns
        -------
        str
            Tool result
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Check microscope requirement
        if tool.requires_microscope:
            client = self._context.get('client')
            if client is None:
                return "Error: Not connected to microscope server. Start the server and reconnect."

        start_time = time.time()

        try:
            # Prepare arguments
            kwargs = dict(tool_input)

            # Inject context if handler expects it
            sig = inspect.signature(tool.handler)
            if 'context' in sig.parameters:
                kwargs['context'] = self._context

            # Execute handler
            if tool.is_async:
                result = await tool.handler(**kwargs)
            else:
                result = await asyncio.to_thread(tool.handler, **kwargs)

            duration = time.time() - start_time
            logger.debug(f"Tool {tool_name} executed in {duration:.2f}s")

            return result

        except Exception as e:
            import traceback
            logger.error(f"Tool {tool_name} failed: {e}")
            return f"Error executing {tool_name}: {str(e)}\n{traceback.format_exc()}"

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def set_tool_registry(registry: ToolRegistry):
    """Set the global tool registry"""
    global _global_registry
    _global_registry = registry


# Convenience decorator using global registry
def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.UTILITY,
    requires_microscope: bool = False,
    tags: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to register a tool with the global registry

    Usage:
        @tool(
            name="get_status",
            description="Get experiment status",
            category=ToolCategory.EXPERIMENT,
        )
        def get_status() -> str:
            return "Status: running"
    """
    return get_tool_registry().register(
        name=name,
        description=description,
        category=category,
        requires_microscope=requires_microscope,
        tags=tags,
    )
