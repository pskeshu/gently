# Tool Development Guide

Best practices for creating tools for the Microscopy Copilot, based on Anthropic's tool use documentation.

## Key Principle

> "Detailed descriptions are by far the most important factor in tool performance."
> — Anthropic Docs

## Tool Definition Structure

```python
from ..tool_registry import tool, ToolCategory, ToolExample

@tool(
    name="tool_name",
    description="""...""",  # 3-4 sentences - SEE BELOW
    category=ToolCategory.EXPERIMENT,  # Choose appropriate category
    requires_microscope=True,  # Set True if needs hardware connection
    examples=[
        ToolExample("User says this", {"param": "value"}),
    ],
)
async def tool_name(param: str, context: Dict) -> str:
    """Docstring for internal use"""
    ...
```

## Writing Good Descriptions (Critical!)

Every description should include 3-4 sentences covering:

1. **What it does** - Clear explanation of the tool's function
2. **When to use it** - Trigger phrases that should invoke this tool
3. **Important details** - Parameters, limitations, side effects
4. **Related context** - What happens after, related tools

### Example - Good Description

```python
description="""Get a comprehensive summary of the current experiment including all embryos, their XY stage positions, calibration status, and imaging history.
Use this tool when the user asks about embryo locations, experiment status, how many embryos exist, or wants an overview.
This is the primary tool for answering questions like "where are the embryos?" or "what's the current status?"
Returns all embryo IDs with their coordinates - no parameters needed."""
```

### Example - Bad Description (Too Short)

```python
description="Get experiment status"  # DON'T DO THIS
```

## Adding Examples

Examples help Claude match user queries to tools. Use the `input_examples` field:

```python
examples=[
    ToolExample("Where are all the embryos?", {}),  # No params needed
    ToolExample("Check embryo 3", {"embryo_id": "embryo_3"}),  # With params
]
```

- First argument: Example user query that should trigger this tool
- Second argument: The tool_input dict to use (empty `{}` if no params)

## Categories

Choose the appropriate category from `ToolCategory`:

| Category | Use For |
|----------|---------|
| `EXPERIMENT` | Experiment-wide operations, status |
| `EMBRYO` | Single embryo operations |
| `MOVEMENT` | Stage movement |
| `HARDWARE` | Direct hardware control (LED, etc.) |
| `DETECTION` | Embryo detection, marking |
| `CALIBRATION` | Calibration procedures |
| `ACQUISITION` | Image/volume acquisition |
| `DATA` | Databroker, file operations |
| `UTILITY` | General utilities |

## Async vs Sync

- Use `async def` for tools that call hardware or do I/O
- Use regular `def` for pure computation tools
- The registry handles both automatically

## Context Parameter

Always include `context: Dict` as the last parameter:

```python
async def my_tool(embryo_id: str, context: Dict) -> str:
    copilot = context.get('copilot')
    client = context.get('client')  # Microscope client
```

## Return Values

- Return strings (displayed to user via Claude)
- Start success messages with `✓` for clarity
- Include relevant details in the response
- On error, return descriptive error message

## Common Patterns

### Getting Embryo with Validation

```python
from ..tool_helpers import require_copilot, get_embryo_or_error

copilot, err = require_copilot(context)
if err:
    return err

embryo, err = get_embryo_or_error(copilot, embryo_id)
if err:
    return err
```

### Hardware Operations

```python
client = context.get('client')
if not client:
    return "Error: Not connected to microscope"

try:
    result = await client.some_operation()
    return f"✓ Success: {result}"
except Exception as e:
    return f"Error: {str(e)}"
```

## Checklist for New Tools

- [ ] Description is 3-4 sentences
- [ ] Description includes "when to use" trigger phrases
- [ ] Added 2-3 examples with realistic user queries
- [ ] Correct category selected
- [ ] `requires_microscope=True` if needs hardware
- [ ] Returns helpful success/error messages
- [ ] Uses `context` parameter for copilot/client access

## References

- [Anthropic Tool Use Docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Advanced Tool Use Patterns](https://www.anthropic.com/engineering/advanced-tool-use)
