#!/usr/bin/env python3
"""
Claude-powered Microscope Focus Tools
====================================

MCP server tools for Claude Code SDK integration with DiSPIM microscope.
Provides tool functions for microscope control and Claude-based focus evaluation.
"""

import asyncio
import base64
import io
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import rpyc

# Import existing gently devices
from gently.devices import DiSPIMCamera, DiSPIMZstage
from client import get_mmc

# Global microscope state
_microscope_state = {
    'core': None,
    'camera': None,
    'z_stage': None,
    'connected': False,
    'focus_history': []
}

def connect_microscope(hostname: str = "localhost", port: int = 18861) -> bool:
    """Connect to the DiSPIM microscope"""
    try:
        core = get_mmc(hostname, port)
        camera = DiSPIMCamera("Bottom PCO", core, name="bottom_camera")
        z_stage = DiSPIMZstage("ZStage:Z:32", core, name="focus_bottom_z")

        _microscope_state.update({
            'core': core,
            'camera': camera,
            'z_stage': z_stage,
            'connected': True
        })
        return True
    except Exception as e:
        print(f"Failed to connect to microscope: {e}")
        return False

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

# MCP Tool Functions for Claude Code SDK

async def move_z_stage(args: Dict[str, Any]) -> Dict[str, Any]:
    """Move Z stage to specified position"""
    if not _microscope_state['connected']:
        return {
            "content": [{"type": "text", "text": "Error: Microscope not connected"}],
            "is_error": True
        }

    try:
        position = float(args['position'])
        z_stage = _microscope_state['z_stage']

        # Check limits
        if not (z_stage.limits[0] <= position <= z_stage.limits[1]):
            return {
                "content": [{"type": "text", "text": f"Error: Position {position} outside limits {z_stage.limits}"}],
                "is_error": True
            }

        # Move stage
        status = z_stage.set(position)
        while not status.done:
            await asyncio.sleep(0.1)

        if status.success:
            current_pos = z_stage.read()[z_stage.name]['value']
            return {
                "content": [{"type": "text", "text": f"Z stage moved to {current_pos:.2f} μm"}]
            }
        else:
            return {
                "content": [{"type": "text", "text": "Error: Failed to move Z stage"}],
                "is_error": True
            }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error moving Z stage: {str(e)}"}],
            "is_error": True
        }

async def capture_image(args: Dict[str, Any]) -> Dict[str, Any]:
    """Capture image from microscope camera"""
    if not _microscope_state['connected']:
        return {
            "content": [{"type": "text", "text": "Error: Microscope not connected"}],
            "is_error": True
        }

    try:
        camera = _microscope_state['camera']
        z_stage = _microscope_state['z_stage']

        # Capture image
        status = camera.trigger()
        while not status.done:
            await asyncio.sleep(0.1)

        if not status.success:
            return {
                "content": [{"type": "text", "text": "Error: Failed to capture image"}],
                "is_error": True
            }

        # Get image data
        image_data = camera.read()[camera.name]['value']
        current_z = z_stage.read()[z_stage.name]['value']

        # Convert to base64 for Claude
        image_b64 = image_to_base64(image_data)

        # Store in focus history
        _microscope_state['focus_history'].append({
            'z_position': current_z,
            'image': image_data,
            'timestamp': time.time()
        })

        return {
            "content": [
                {"type": "text", "text": f"Image captured at Z={current_z:.2f} μm"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
            ]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error capturing image: {str(e)}"}],
            "is_error": True
        }

async def get_microscope_status(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get current microscope status"""
    if not _microscope_state['connected']:
        return {
            "content": [{"type": "text", "text": "Microscope: Disconnected"}]
        }

    try:
        z_stage = _microscope_state['z_stage']

        current_z = z_stage.read()[z_stage.name]['value']
        z_limits = z_stage.limits

        status_text = f"""Microscope Status:
- Connected: Yes
- Current Z position: {current_z:.2f} μm
- Z limits: {z_limits[0]:.1f} - {z_limits[1]:.1f} μm
- Focus history: {len(_microscope_state['focus_history'])} positions"""

        return {
            "content": [{"type": "text", "text": status_text}]
        }

    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error getting status: {str(e)}"}],
            "is_error": True
        }

async def get_focus_history(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get focus sweep history with images"""
    history = _microscope_state['focus_history']

    if not history:
        return {
            "content": [{"type": "text", "text": "No focus history available"}]
        }

    # Get recent history (last N images)
    limit = args.get('limit', 5)
    recent_history = history[-limit:]

    content = [{"type": "text", "text": f"Focus History ({len(recent_history)} most recent):"}]

    for i, entry in enumerate(recent_history):
        z_pos = entry['z_position']
        image_b64 = image_to_base64(entry['image'])

        content.extend([
            {"type": "text", "text": f"\nPosition {i+1}: Z = {z_pos:.2f} μm"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}}
        ])

    return {"content": content}

async def clear_focus_history(args: Dict[str, Any]) -> Dict[str, Any]:
    """Clear focus history"""
    count = len(_microscope_state['focus_history'])
    _microscope_state['focus_history'].clear()

    return {
        "content": [{"type": "text", "text": f"Cleared {count} entries from focus history"}]
    }

# Tool definitions for MCP server
MICROSCOPE_TOOLS = [
    {
        "name": "move_z_stage",
        "description": "Move the microscope Z stage to a specific position in micrometers",
        "inputSchema": {
            "type": "object",
            "properties": {
                "position": {"type": "number", "description": "Z position in micrometers (50-250 μm range)"}
            },
            "required": ["position"]
        },
        "handler": move_z_stage
    },
    {
        "name": "capture_image",
        "description": "Capture an image from the microscope camera at current Z position",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "handler": capture_image
    },
    {
        "name": "get_microscope_status",
        "description": "Get current status of the microscope including position and settings",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "handler": get_microscope_status
    },
    {
        "name": "get_focus_history",
        "description": "Get recent focus sweep history with images",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "number", "description": "Number of recent entries to return (default: 5)"}
            },
            "required": []
        },
        "handler": get_focus_history
    },
    {
        "name": "clear_focus_history",
        "description": "Clear the focus history buffer",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        },
        "handler": clear_focus_history
    }
]

if __name__ == "__main__":
    # Test connection
    if connect_microscope():
        print("Microscope tools ready for Claude Code SDK")
    else:
        print("Failed to connect to microscope")