# Rich CLI for Microscopy Copilot

Enhanced terminal interface with colors, streaming responses, and interactive features.

## Features

### 🎨 Semantic Color Coding
- **User input**: Green (bold)
- **Copilot responses**: Blue
- **System messages**: Yellow
- **Tool calls**: Magenta
- **Errors**: Red (bold)
- **Success**: Bright green

### ⚡ Streaming Responses
- See copilot responses appear word-by-word as Claude generates them
- Real-time progress indicators during long operations
- Smoother, more responsive interaction

### 📝 Command Autocomplete
Press `Tab` for intelligent completions:
- **Slash commands**: `/detectors`, `/status`, `/embryos`, `/help`, etc.
- **Embryo IDs**: `embryo_001`, `embryo_002`, etc.
- **Detector names**: `hatching`, `comma`, `pretzel`, etc.
- **Common phrases**: "What detectors do we have?", "Show me the status", etc.

### 📜 Command History
- Press `↑/↓` to browse previous commands
- Press `Ctrl+R` for reverse search through history
- Auto-suggest from history as you type
- History persisted across sessions

### 📊 Rich Formatting
- **Panels**: Messages displayed in bordered panels with timestamps
- **Tables**: Detectors and embryos shown in formatted tables
- **Markdown**: Copilot responses rendered with markdown formatting
- **Syntax highlighting**: Code blocks with proper highlighting

## Usage

### Running the Rich CLI

```python
import asyncio
from pathlib import Path
from gently.agent import MicroscopyCopilot, run_rich_cli

async def main():
    # Initialize copilot
    copilot = MicroscopyCopilot(
        storage_path=Path("./experiment_data"),
        model="claude-sonnet-4-5-20250929"
    )

    # Load embryos
    copilot.load_embryos_from_database(database)

    # Run Rich CLI
    await run_rich_cli(copilot)

asyncio.run(main())
```

### Demo Scripts

```bash
# Test copilot with Rich CLI
python test_copilot.py

# Detector system demo with Rich CLI
python demo_detector_conversation.py
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Tab` | Autocomplete command/ID/name |
| `↑` / `↓` | Browse command history |
| `Ctrl+R` | Reverse search history |
| `Ctrl+C` | Confirm exit |
| `Ctrl+D` | Exit immediately |
| `Ctrl+L` | Clear screen |

## Slash Commands

Quick access to common operations:

- `/detectors` - List all detectors with statistics
- `/embryos` - List all embryos with status
- `/status` - Show experiment status dashboard
- `/help` - Show help message
- `/clear` - Clear screen
- `/quit` - Exit

## Example Session

```
┌─────────────────────────────────────────────────────────────┐
│ Microscopy Copilot v2.0                                     │
│ AI-powered adaptive microscopy control                      │
│                                                              │
│ Commands:                                                    │
│   • Type naturally to interact with copilot                 │
│   • Use /detectors, /status, /embryos for quick info        │
│   • Press Tab for autocomplete                              │
│   • Press ↑/↓ for command history                           │
│   • Press Ctrl+C to exit                                    │
└─────────────────────────────────────────────────────────────┘

> What detectors do we have?_
```

After entering, you'll see:

```
┌─ 12:34:56 You ──────────────────────────────────────────────┐
│ What detectors do we have?                                   │
└──────────────────────────────────────────────────────────────┘

⠋ Thinking...

┌─ Tool Call ──────────────────────────────────────────────────┐
│ list_detectors                                               │
│   filter: all                                                │
│                                                              │
│ ⏱  0.12s                                                     │
└──────────────────────────────────────────────────────────────┘

┌─ 12:34:58 Copilot ───────────────────────────────────────────┐
│ We have 3 detectors configured:                             │
│                                                              │
│ ┌────────────┬────────┬──────────┬──────┬────────────┐      │
│ │ Name       │ Status │ Mode     │ Runs │ Detections │      │
│ ├────────────┼────────┼──────────┼──────┼────────────┤      │
│ │ hatching   │   ✓    │ recommend│  45  │     3      │      │
│ │ comma      │   ✓    │ auto     │  30  │     5      │      │
│ │ pretzel    │   ✓    │ passive  │  28  │     2      │      │
│ └────────────┴────────┴──────────┴──────┴────────────┘      │
│                                                              │
│ All detectors are currently enabled!                        │
└──────────────────────────────────────────────────────────────┘

>
```

## Customization

### Custom Color Scheme

Edit `gently/agent/rich_cli.py`:

```python
class ColorScheme:
    USER = "green"          # Your color here
    COPILOT = "blue"        # Your color here
    SYSTEM = "yellow"       # Your color here
    TOOL = "magenta"        # Your color here
    ERROR = "bold red"      # Your color here
    SUCCESS = "bright_green"# Your color here
```

### Custom History File

```python
await run_rich_cli(
    copilot,
    history_file=Path("/custom/path/.copilot_history")
)
```

## Architecture

```
User Input (prompt-toolkit with autocomplete)
    ↓
RichCopilotCLI
    ↓
MicroscopyCopilot.handle_message_stream()
    ↓
Claude API (streaming)
    ↓
Rich Progress Spinner + Live Updates
    ↓
Rich Formatted Output (panels, tables, markdown)
```

## Dependencies

```
rich>=13.7.0           # Terminal UI framework
prompt-toolkit>=3.0.43 # Advanced input with autocomplete
```

## Files

- `gently/agent/rich_cli.py` - Main Rich CLI implementation
- `gently/agent/autocomplete.py` - Autocomplete logic
- `gently/agent/copilot.py` - Streaming support added
- `test_copilot.py` - Updated to use Rich CLI
- `demo_detector_conversation.py` - Updated to use Rich CLI

## Troubleshoats

### Import Error

```bash
# Make sure dependencies are installed
pip install rich prompt-toolkit
```

### Streaming Not Working

The streaming implementation uses `anthropic.messages.stream()`. Make sure you have:
```
anthropic>=0.39.0
```

### Autocomplete Not Showing

- Make sure you're pressing `Tab`, not `Enter`
- Autocomplete works for slash commands (starting with `/`), embryo IDs, and detector names
- Try typing a few more characters if no matches appear

## Future Enhancements

Potential additions:
- Live status dashboard (right sidebar)
- Real-time detector monitoring
- Visual progress bars for acquisitions
- Image preview in terminal
- Keyboard shortcuts for common operations

---

Enjoy the enhanced microscopy copilot experience! 🔬✨
