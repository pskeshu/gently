# DiSPIM Microscope Control with Claude Code

This repository contains a DiSPIM (Dual-View Selective Plane Illumination Microscopy) control system integrated with Claude Code for AI-powered microscope operation.

## MCP Server Configuration

The microscope control tools are available through an MCP server. To use them:

```bash
# Start Claude Code with microscope tools
claude --mcp-config "C:\Users\dispim\Documents\GitHub\gently\microscope_mcp_config.json"

# Or use the interactive mode
cd "C:\Users\dispim\Documents\GitHub\gently"
claude --mcp-config microscope_mcp_config.json
```

## Available Microscope Tools

1. **connect_microscope** - Connect to the DiSPIM microscope hardware
2. **move_z_stage** - Move the microscope Z stage to a specific position (50-250 μm range)
3. **capture_image** - Capture an image from the microscope camera at current Z position
4. **get_microscope_status** - Get current status including position and settings
5. **get_focus_history** - Get recent focus sweep history with images
6. **clear_focus_history** - Clear the focus history buffer

## Focus Workflow

The system supports Claude-guided focusing:

1. Connect to the microscope using `connect_microscope`
2. Check current status with `get_microscope_status`
3. Perform focus sweeps by moving the Z stage and capturing images
4. Claude analyzes images for focus quality based on embryo boundary sharpness
5. Move to the optimal focus position

## Files

- `microscope_mcp_server.py` - Main MCP server with microscope tools
- `run_microscope_mcp.py` - Wrapper script for MCP server
- `microscope_mcp_config.json` - MCP server configuration
- `claude_focus_tools.py` - Original microscope control functions
- `claude_focus_script.py` - Original SDK-based implementation

## Usage Notes

- The microscope server must be running (default: localhost:18861)
- Focus evaluation is optimized for bottom camera view showing embryo outline
- Image artifacts from room lighting are expected and should be ignored during focus evaluation

## Accessing Copilot Session History

Copilot conversations are stored as JSON files in `D:\Gently\sessions\`. Each session contains the full conversation history, experiment state, and embryo configurations.

**To find and read the latest session:**

1. List sessions by modification time:
   ```bash
   ls -la "D:/Gently/sessions/"
   ```

2. The most recent session is the one with the latest timestamp. Look for larger files (>1KB) which contain actual conversations.

3. Read the session JSON file to access:
   - `conversation` array: Full chat history with user messages, assistant responses, and tool calls
   - `experiment_data`: Current experiment state and embryo configurations
   - `session_id`, `created_at`, `last_active`: Session metadata

**Session file structure:**
```
D:\Gently\
├── sessions\
│   ├── <session_id>.json    # Session state and conversation
│   └── ...
├── logs\                     # Copilot logs
├── lightsheet_captures\      # Captured images
└── .copilot_history          # CLI command history
```