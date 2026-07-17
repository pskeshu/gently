"""
Tool Plugin System for Microscopy Agent

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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for organizing tools"""

    ACQUISITION = auto()  # Volume/image acquisition
    MOVEMENT = auto()  # Stage movement, positioning
    CALIBRATION = auto()  # Calibration procedures
    ANALYSIS = auto()  # Image/volume analysis
    DETECTION = auto()  # Detector management
    EXPERIMENT = auto()  # Experiment state management
    EMBRYO = auto()  # Embryo-specific operations
    HARDWARE = auto()  # Direct hardware control
    DATA = auto()  # Data/Databroker operations
    UTILITY = auto()  # Utility functions
    ML = auto()  # Machine learning training
    TRANSFER = auto()  # Bulk data transfer


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""

    name: str
    type: str  # JSON schema type
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolExample:
    """Example of when to use a tool"""

    user_query: str
    tool_input: dict = field(default_factory=dict)


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
    parameters: list[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.UTILITY
    requires_microscope: bool = False
    is_async: bool = False
    tags: list[str] = field(default_factory=list)
    examples: list[ToolExample] = field(default_factory=list)

    def to_claude_schema(self) -> dict:
        """Generate Claude API tool schema with examples embedded in description"""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
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
        # (input_examples field not supported in current API)
        desc = self.description
        if self.examples:
            example_inputs = [ex.tool_input for ex in self.examples if ex.tool_input]
            if example_inputs:
                desc += f"\n\nExample inputs: {example_inputs}"

        return {
            "name": self.name,
            "description": desc,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
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


def _unwrap_optional(tp: Any) -> Any:
    """Reduce ``float | None`` / ``Optional[int]`` to the underlying scalar type.

    Returns the single non-None member of a union, or the type unchanged for a
    plain annotation. Returns None when there's no unambiguous scalar (so callers
    skip coercion).
    """
    args = get_args(tp)
    if args:
        non_none = [a for a in args if a is not type(None)]
        return non_none[0] if len(non_none) == 1 else None
    return tp


def _coerce_kwargs(handler: Callable, kwargs: dict) -> dict:
    """Best-effort coercion of string tool args to their annotated scalar types.

    Tool inputs arrive as JSON from the model (and sometimes as UI form strings),
    so a param annotated ``float``/``int`` can show up as e.g. ``"120"``. Without
    this, a downstream numeric comparison raises
    ``'<' not supported between instances of 'str' and 'int'``. Coercion is
    conservative: only string values whose annotation resolves to int/float/bool
    are touched; anything that fails to parse is left as-is for the tool to report.
    """
    try:
        hints = get_type_hints(handler)
    except Exception:
        return kwargs
    for name, value in list(kwargs.items()):
        if name == "context" or not isinstance(value, str):
            continue
        target = _unwrap_optional(hints.get(name))
        if target in (int, float):
            s = value.strip()
            if not s:
                continue
            try:
                kwargs[name] = target(s)
            except (ValueError, TypeError):
                pass
        elif target is bool:
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                kwargs[name] = True
            elif low in ("false", "0", "no", "off"):
                kwargs[name] = False
    return kwargs


def _extract_parameters_from_function(func: Callable) -> list[ToolParameter]:
    """Extract parameter definitions from function signature and type hints"""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    doc = inspect.getdoc(func) or ""

    # Parse docstring for parameter descriptions
    param_docs = {}
    in_params = False
    current_param = None
    for line in doc.split("\n"):
        line = line.strip()
        if line.lower().startswith("parameters"):
            in_params = True
            continue
        if in_params:
            if line.startswith("---"):
                continue
            if line.lower().startswith("returns"):
                in_params = False
                continue
            if " : " in line:
                parts = line.split(" : ")
                current_param = parts[0].strip()
                param_docs[current_param] = ""
            elif current_param and line:
                param_docs[current_param] += " " + line

    parameters = []
    for param_name, param in sig.parameters.items():
        # Skip 'self', 'tool_input' (legacy pattern), and 'context' (injected at runtime)
        if param_name in ("self", "tool_input", "context"):
            continue

        python_type = hints.get(param_name, str)
        json_type = _python_type_to_json_schema(python_type)

        # Check if optional (has default or is Optional type)
        required = param.default is inspect.Parameter.empty
        default = None if param.default is inspect.Parameter.empty else param.default

        # Get description from docstring
        description = param_docs.get(param_name, f"The {param_name} parameter").strip()

        parameters.append(
            ToolParameter(
                name=param_name,
                type=json_type,
                description=description,
                required=required,
                default=default,
            )
        )

    return parameters


class ToolRegistry:
    """
    Central registry for all agent tools

    Features:
    - Decorator-based registration
    - Category filtering
    - Automatic schema generation
    - Tool execution with timing and events
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._context: dict[str, Any] = {}  # Shared context (agent, client, etc.)

    def set_context(self, key: str, value: Any):
        """Set shared context available to all tools"""
        self._context[key] = value

    def get_context(self, key: str) -> Any:
        """Get shared context"""
        return self._context.get(key)

    def register(
        self,
        name: str | None = None,
        description: str | None = None,
        category: ToolCategory = ToolCategory.UTILITY,
        requires_microscope: bool = False,
        tags: list[str] | None = None,
        parameters: list[ToolParameter] | None = None,
        examples: list[ToolExample] | None = None,
    ) -> Callable:
        """
        Decorator to register a function as a tool

        Usage:
            @registry.register(
                name="acquire_volume",
                description="Acquire a 3D volume for an embryo",
                category=ToolCategory.ACQUISITION,
                requires_microscope=True,
                examples=[
                    ToolExample("Acquire a volume of embryo 1", {"embryo_id": "embryo_1"}),
                ],
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
        examples : list of ToolExample, optional
            Usage examples showing when to call this tool
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (inspect.getdoc(func) or "").split("\n")[0]

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
                examples=examples or [],
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
        name: str | None = None,
        description: str | None = None,
        category: ToolCategory = ToolCategory.UTILITY,
        requires_microscope: bool = False,
        tags: list[str] | None = None,
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
        tool_desc = description or (inspect.getdoc(func) or "").split("\n")[0]
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

    def get(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name"""
        return self._tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        """List all registered tools"""
        return list(self._tools.values())

    def list_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """List tools in a category"""
        return [t for t in self._tools.values() if t.category == category]

    def list_by_tag(self, tag: str) -> list[ToolDefinition]:
        """List tools with a specific tag"""
        return [t for t in self._tools.values() if tag in t.tags]

    def list_available(self, has_microscope: bool = False) -> list[ToolDefinition]:
        """List tools available given current context"""
        tools = []
        for tool in self._tools.values():
            if tool.requires_microscope and not has_microscope:
                continue
            tools.append(tool)
        return tools

    def get_claude_schemas(self, has_microscope: bool = False) -> list[dict]:
        """Get Claude API tool schemas for available tools"""
        return [tool.to_claude_schema() for tool in self.list_available(has_microscope)]

    async def execute(self, tool_name: str, tool_input: dict, context: dict | None = None) -> str:
        """
        Execute a tool by name

        Parameters
        ----------
        tool_name : str
            Name of tool to execute
        tool_input : dict
            Tool input parameters
        context : dict, optional
            Execution context (agent, client, etc.)

        Returns
        -------
        str
            Tool result
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_name}")

        # Determine execution context:
        # 1. Use explicit context parameter if provided (from agent.execute_tool)
        # 2. Check if context was passed in tool_input (from nested tool calls via wrapper)
        # 3. Fall back to stored registry context
        if context is not None:
            exec_context = context
        elif "context" in tool_input and tool_input["context"] is not None:
            exec_context = tool_input["context"]
        else:
            exec_context = self._context

        # Hybrid-autonomy backstop: during an autonomous (wake) turn, a small set
        # of irreversible tools (laser-on, embryo termination, stopping the run)
        # must NEVER execute without a human — even if the model tries to call
        # them directly. The agent sets these flags around its autonomous turns;
        # user-driven turns are unaffected. The blocked set is supplied by the
        # agent so this layer stays free of app-specific tool names.
        _agent = exec_context.get("agent") if isinstance(exec_context, dict) else None
        if _agent is not None and getattr(_agent, "_autonomous_active", False):
            blocked = getattr(_agent, "_autonomous_blocked_tools", None) or ()
            if tool_name in blocked:
                logger.info("Autonomy backstop blocked '%s' (irreversible)", tool_name)
                return (
                    f"'{tool_name}' is an irreversible action and cannot run "
                    f"autonomously. Ask the operator to confirm it."
                )

        # Check microscope requirement
        if tool.requires_microscope:
            client = exec_context.get("client")
            if client is None:
                return "Error: Not connected to microscope server. Start the server and reconnect."

        start_time = time.time()

        try:
            # Prepare arguments
            kwargs = dict(tool_input)

            # Coerce string args to their annotated scalar types. JSON/UI inputs
            # can deliver e.g. new_interval_seconds="120", which would otherwise
            # crash on a numeric comparison inside the tool.
            kwargs = _coerce_kwargs(tool.handler, kwargs)

            # Inject context if handler expects it (but don't overwrite if already provided)
            sig = inspect.signature(tool.handler)
            if "context" in sig.parameters and "context" not in kwargs:
                kwargs["context"] = exec_context

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
_global_registry: ToolRegistry | None = None


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
    name: str | None = None,
    description: str | None = None,
    category: ToolCategory = ToolCategory.UTILITY,
    requires_microscope: bool = False,
    tags: list[str] | None = None,
    examples: list[ToolExample] | None = None,
) -> Callable:
    """
    Decorator to register a tool with the global registry

    Usage:
        @tool(
            name="get_status",
            description="Get experiment status",
            category=ToolCategory.EXPERIMENT,
            examples=[
                ToolExample("What's the status?"),
                ToolExample("Show me experiment info"),
            ],
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
        examples=examples,
    )
