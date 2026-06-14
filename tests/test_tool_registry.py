"""
Tests for ToolRegistry — decorator-based tool plugin system.

Tests cover:
- Register and retrieve tools
- Decorator-based registration
- Unregister
- Schema generation from type hints
- Required vs optional parameter detection
- Category filtering
- Microscope availability filtering
- Sync and async tool execution
- Execution of non-existent tool
"""

import pytest

from gently.harness.tools.registry import (
    ToolCategory,
    ToolRegistry,
)


@pytest.fixture
def registry():
    """Fresh ToolRegistry for each test."""
    return ToolRegistry()


# =========================================================================
# Registration
# =========================================================================


class TestRegistration:
    def test_register_and_get(self, registry):
        def my_tool(x: int) -> str:
            return str(x)

        registry.register_function(my_tool, name="my_tool", description="test tool")

        tool = registry.get("my_tool")
        assert tool is not None
        assert tool.name == "my_tool"
        assert tool.description == "test tool"

    def test_register_with_decorator(self, registry):
        @registry.register(
            name="deco_tool", description="decorated", category=ToolCategory.ANALYSIS
        )
        async def deco_tool(value: float) -> str:
            return f"result: {value}"

        tool = registry.get("deco_tool")
        assert tool is not None
        assert tool.name == "deco_tool"
        assert tool.category == ToolCategory.ANALYSIS
        assert tool.is_async is True

    def test_unregister(self, registry):
        def temp_tool():
            pass

        registry.register_function(temp_tool, name="temp_tool")
        assert "temp_tool" in registry

        result = registry.unregister("temp_tool")
        assert result is True
        assert "temp_tool" not in registry

    def test_unregister_nonexistent(self, registry):
        assert registry.unregister("no_such_tool") is False


# =========================================================================
# Schema generation
# =========================================================================


class TestSchemaGeneration:
    def test_schema_generation_from_type_hints(self, registry):
        def typed_tool(name: str, count: int, ratio: float, flag: bool) -> str:
            """A typed tool.

            Parameters
            ----------
            name : str
                The name
            count : int
                How many
            ratio : float
                The ratio
            flag : bool
                A flag
            """
            return "ok"

        registry.register_function(typed_tool, name="typed_tool")
        tool = registry.get("typed_tool")
        schema = tool.to_claude_schema()

        props = schema["input_schema"]["properties"]
        assert props["name"]["type"] == "string"
        assert props["count"]["type"] == "integer"
        assert props["ratio"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"

    def test_schema_required_vs_optional_params(self, registry):
        def mixed_tool(required_param: str, optional_param: int = 10) -> str:
            return "ok"

        registry.register_function(mixed_tool, name="mixed_tool")
        tool = registry.get("mixed_tool")
        schema = tool.to_claude_schema()

        assert "required_param" in schema["input_schema"]["required"]
        assert "optional_param" not in schema["input_schema"]["required"]


# =========================================================================
# Filtering
# =========================================================================


class TestFiltering:
    def test_list_by_category(self, registry):
        def tool_a():
            pass

        def tool_b():
            pass

        def tool_c():
            pass

        registry.register_function(tool_a, name="a", category=ToolCategory.ACQUISITION)
        registry.register_function(tool_b, name="b", category=ToolCategory.ANALYSIS)
        registry.register_function(tool_c, name="c", category=ToolCategory.ACQUISITION)

        acq_tools = registry.list_by_category(ToolCategory.ACQUISITION)
        assert len(acq_tools) == 2
        assert all(t.category == ToolCategory.ACQUISITION for t in acq_tools)

    def test_list_available_with_microscope_filter(self, registry):
        def offline_tool():
            pass

        def hw_tool():
            pass

        registry.register_function(offline_tool, name="offline", requires_microscope=False)
        registry.register_function(hw_tool, name="hw", requires_microscope=True)

        # Without microscope
        available = registry.list_available(has_microscope=False)
        names = [t.name for t in available]
        assert "offline" in names
        assert "hw" not in names

        # With microscope
        available = registry.list_available(has_microscope=True)
        names = [t.name for t in available]
        assert "offline" in names
        assert "hw" in names


# =========================================================================
# Execution
# =========================================================================


class TestExecution:
    @pytest.mark.asyncio
    async def test_execute_sync_tool(self, registry):
        def adder(a: int, b: int) -> str:
            """Add two numbers.

            Parameters
            ----------
            a : int
                First number
            b : int
                Second number
            """
            return str(a + b)

        registry.register_function(adder, name="adder")
        result = await registry.execute("adder", {"a": 3, "b": 4})
        assert result == "7"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self, registry):
        async def async_greeter(name: str) -> str:
            """Greet someone.

            Parameters
            ----------
            name : str
                Who to greet
            """
            return f"Hello, {name}!"

        registry.register_function(async_greeter, name="greeter")
        result = await registry.execute("greeter", {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, registry):
        with pytest.raises(ValueError, match="Unknown tool"):
            await registry.execute("nonexistent", {})
