#!/usr/bin/env python3
"""
DiSPIM Microscope MCP Server
============================

MCP (Model Context Protocol) server for integrating DiSPIM microscope controls
with Claude Code CLI. This server provides microscope control tools that Claude
can use through the standard Claude Code interface.

Run this server and register it in Claude Code to get the full CLI experience
with your microscope tools.
"""

import asyncio
import base64
import io
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image
from mcp import ClientSession, StdioServerParameters
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource

# Import existing microscope functionality
from claude_focus_tools import connect_microscope, _microscope_state

# Initialize MCP server
server = Server("dispim-microscope")

def image_to_base64(image_array: np.ndarray) -> str:
    """Convert numpy image array to base64 encoded PNG"""
    # Normalize image to 8-bit
    if image_array.dtype != np.uint8:
        if image_array.max() > 255:
            # Normalize to 0-255 range
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    # Convert to PIL Image and save as PNG
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode as base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available microscope tools"""
    return [
        Tool(
            name="move_z_stage",
            description="Move the microscope Z stage to a specific position in micrometers",
            inputSchema={
                "type": "object",
                "properties": {
                    "position": {
                        "type": "number",
                        "description": "Z position in micrometers (50-250 μm range)",
                        "minimum": 50,
                        "maximum": 250
                    }
                },
                "required": ["position"],
                "additionalProperties": False
            }
        ),
        Tool(
            name="capture_image",
            description="Capture an image from the microscope camera at current Z position",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_microscope_status",
            description="Get current status of the microscope including position and settings",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        ),
        Tool(
            name="get_focus_history",
            description="Get recent focus sweep history with images",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of recent entries to return (default: 5)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        ),
        Tool(
            name="clear_focus_history",
            description="Clear the focus history buffer",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False
            }
        ),
        Tool(
            name="connect_microscope",
            description="Connect to the DiSPIM microscope hardware",
            inputSchema={
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string",
                        "description": "Microscope server hostname (default: localhost)",
                        "default": "localhost"
                    },
                    "port": {
                        "type": "integer",
                        "description": "Microscope server port (default: 18861)",
                        "default": 18861
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[TextContent | ImageContent]:
    """Handle tool calls from Claude"""

    if name == "connect_microscope":
        hostname = arguments.get("hostname", "localhost")
        port = arguments.get("port", 18861)

        try:
            success = connect_microscope(hostname, port)
            if success:
                return [TextContent(type="text", text=f"Successfully connected to microscope at {hostname}:{port}")]
            else:
                return [TextContent(type="text", text=f"Failed to connect to microscope at {hostname}:{port}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error connecting to microscope: {str(e)}")]

    elif name == "move_z_stage":
        if not _microscope_state['connected']:
            return [TextContent(type="text", text="Error: Microscope not connected. Use connect_microscope tool first.")]

        try:
            position = float(arguments['position'])
            z_stage = _microscope_state['z_stage']

            # Check limits
            if not (z_stage.limits[0] <= position <= z_stage.limits[1]):
                return [TextContent(type="text", text=f"Error: Position {position} outside limits {z_stage.limits}")]

            # Move stage
            status = z_stage.set(position)
            while not status.done:
                await asyncio.sleep(0.1)

            if status.success:
                current_pos = z_stage.read()[z_stage.name]['value']
                return [TextContent(type="text", text=f"Z stage moved to {current_pos:.2f} μm")]
            else:
                return [TextContent(type="text", text="Error: Failed to move Z stage")]

        except Exception as e:
            return [TextContent(type="text", text=f"Error moving Z stage: {str(e)}")]

    elif name == "capture_image":
        if not _microscope_state['connected']:
            return [TextContent(type="text", text="Error: Microscope not connected. Use connect_microscope tool first.")]

        try:
            camera = _microscope_state['camera']
            z_stage = _microscope_state['z_stage']

            # Capture image
            status = camera.trigger()
            while not status.done:
                await asyncio.sleep(0.1)

            if not status.success:
                return [TextContent(type="text", text="Error: Failed to capture image")]

            # Get image data
            image_data = camera.read()[camera.name]['value']
            current_z = z_stage.read()[z_stage.name]['value']

            # Convert to base64 for Claude
            image_b64 = image_to_base64(image_data)

            # Save image to disk for Napari viewer
            from pathlib import Path
            from datetime import datetime

            timestamp = time.time()
            image_dir = Path("microscope_images")
            image_dir.mkdir(exist_ok=True)

            # Create filename with timestamp and Z position
            dt = datetime.fromtimestamp(timestamp)
            filename = f"microscope_{dt.strftime('%Y%m%d_%H%M%S')}_Z{current_z:.2f}um.png"
            image_path = image_dir / filename

            # Save as PNG
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(image_data)
            pil_image.save(image_path)

            # Store in focus history
            _microscope_state['focus_history'].append({
                'z_position': current_z,
                'image': image_data,
                'timestamp': timestamp,
                'saved_path': str(image_path)
            })

            return [
                TextContent(type="text", text=f"Image captured at Z={current_z:.2f} μm"),
                ImageContent(
                    type="image",
                    data=image_b64,
                    mimeType="image/png"
                )
            ]

        except Exception as e:
            return [TextContent(type="text", text=f"Error capturing image: {str(e)}")]

    elif name == "get_microscope_status":
        if not _microscope_state['connected']:
            return [TextContent(type="text", text="Microscope: Disconnected")]

        try:
            z_stage = _microscope_state['z_stage']

            current_z = z_stage.read()[z_stage.name]['value']
            z_limits = z_stage.limits

            status_text = f"""Microscope Status:
- Connected: Yes
- Current Z position: {current_z:.2f} μm
- Z limits: {z_limits[0]:.1f} - {z_limits[1]:.1f} μm
- Focus history: {len(_microscope_state['focus_history'])} positions"""

            return [TextContent(type="text", text=status_text)]

        except Exception as e:
            return [TextContent(type="text", text=f"Error getting status: {str(e)}")]

    elif name == "get_focus_history":
        history = _microscope_state['focus_history']

        if not history:
            return [TextContent(type="text", text="No focus history available")]

        # Get recent history (last N images)
        limit = arguments.get('limit', 5)
        recent_history = history[-limit:]

        result = [TextContent(type="text", text=f"Focus History ({len(recent_history)} most recent):")]

        for i, entry in enumerate(recent_history):
            z_pos = entry['z_position']
            image_b64 = image_to_base64(entry['image'])

            result.extend([
                TextContent(type="text", text=f"\nPosition {i+1}: Z = {z_pos:.2f} μm"),
                ImageContent(
                    type="image",
                    data=image_b64,
                    mimeType="image/png"
                )
            ])

        return result

    elif name == "clear_focus_history":
        count = len(_microscope_state['focus_history'])
        _microscope_state['focus_history'].clear()

        return [TextContent(type="text", text=f"Cleared {count} entries from focus history")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main entry point for the MCP server"""
    # Start the stdio server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="dispim-microscope",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP server shutting down...")
        sys.exit(0)