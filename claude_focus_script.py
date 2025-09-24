#!/usr/bin/env python3
"""
Claude-Powered Focus Script
===========================

Demonstrates using Claude model for microscope focusing decisions instead of traditional algorithms.
Uses Claude Code SDK to let Claude directly control the microscope through tool calls.
"""

import asyncio
import os
import json
from typing import List, Dict, Any
from claude_focus_tools import connect_microscope, MICROSCOPE_TOOLS

# Import Claude Code SDK - required for operation
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions

class ClaudeFocusController:
    """Claude-powered focus controller using Claude Code SDK"""

    def __init__(self, config: Dict[str, Any] = None):
        self.client = None
        self.focus_session_active = False
        self.config = config or {}

        # Set default configuration
        self.config.setdefault('max_retries', 3)
        self.config.setdefault('timeout', 30)

    async def initialize_claude(self):
        """Initialize Claude Code client with microscope tools"""
        # Configure Claude Code client
        try:
            options = ClaudeCodeOptions()
            self.client = ClaudeSDKClient(options=options)

            # Register microscope tools
            for tool in MICROSCOPE_TOOLS:
                self.client.register_tool(tool["name"], tool["handler"], tool["input_schema"], tool["description"])

            await self.client.connect()
        except Exception as e:
            print(f"Failed to initialize Claude Code client: {e}")
            raise

    async def start_focus_session(self, focus_range: tuple = (100, 200), num_steps: int = 10):
        """Start a Claude-guided focus session"""

        # Connect to microscope
        if not connect_microscope():
            print("Failed to connect to microscope")
            return False

        await self.initialize_claude()
        self.focus_session_active = True

        # Initial prompt to Claude about the focusing task
        focus_prompt = f"""
I need your help to focus a DiSPIM microscope on an embryo sample using the bottom camera view.

IMPORTANT CONTEXT:
- This is a bottom camera view showing a zoomed-out perspective of the entire embryo
- You will see the overall shape and outline of the embryo, not internal cellular structures
- The embryo appears as a roughly oval/spherical object in the field of view
- Focus quality should be judged by the sharpness of the embryo's outer boundary/edge
- When in focus: the embryo boundary will be crisp and well-defined
- When out of focus: the boundary will appear soft, blurry, or have a halo effect

VISUAL ARTIFACTS TO EXPECT:
- You will see lens flare-like artifacts due to stray light from room illumination
- This same room light is what illuminates the sample, so these artifacts are normal
- The lens flares may appear as bright spots, halos, or light streaks in the image
- Focus on the embryo boundary sharpness despite these optical artifacts
- Do not mistake lens flares for focus quality - they are separate phenomena

Your task is to:
1. Check the current microscope status
2. Perform a focus sweep from {focus_range[0]} to {focus_range[1]} μm in {num_steps} steps
3. At each position, capture an image and evaluate the focus quality of the embryo boundary
4. Determine the best focus position based on edge sharpness
5. Move to the optimal focus position

Focus evaluation criteria for this bottom camera view:
- Sharp, well-defined embryo boundary/outline (ignore lens flares)
- Good contrast between embryo and background
- Minimal blur or soft edges around the embryo perimeter
- Clear distinction of the embryo shape

Please start by checking the microscope status, then begin the focus sweep.
Remember to focus on analyzing the sharpness of the embryo's outer boundary, not internal details or optical artifacts.
"""

        if not self.client:
            raise RuntimeError("Claude client not initialized. Call initialize_claude() first.")

        # Send query and receive response
        try:
            await self.client.query(focus_prompt)

            print("Claude response:")
            async for message in self.client.receive_response():
                print(message)

            return True
        except Exception as e:
            print(f"Error communicating with Claude: {e}")
            return False
        finally:
            if self.client:
                await self.client.disconnect()


    async def interactive_focus_session(self):
        """Interactive session where user can chat with Claude about focusing"""

        if not connect_microscope():
            print("Failed to connect to microscope")
            return

        await self.initialize_claude()

        print("\n=== Interactive Claude Focus Session ===")
        print("You can now chat with Claude about focusing the microscope.")
        print("Claude has access to microscope control tools.")
        print("Remember: This is a bottom camera view showing the entire embryo outline.")
        print("Type 'quit' to exit.\n")

        try:
            while True:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if not user_input:
                    continue

                if not self.client:
                    print("Error: Claude client not initialized")
                    break

                try:
                    await self.client.query(user_input)

                    print("Claude:")
                    async for message in self.client.receive_response():
                        print(message)
                except Exception as e:
                    print(f"Error: {e}")
        finally:
            if self.client:
                await self.client.disconnect()

async def main():
    """Main function to run Claude focus demonstration"""

    print("Claude-Powered Microscope Focus Demonstration")
    print("Bottom Camera View - Embryo Boundary Focus")
    print("=" * 50)

    try:
        controller = ClaudeFocusController()

        print("\nChoose a mode:")
        print("1. Automatic focus session (Claude performs full focus sweep)")
        print("2. Interactive session (Chat with Claude about focusing)")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == "1":
            print("\nStarting automatic focus session...")
            print("Focus target: Embryo boundary sharpness in bottom camera view")
            focus_range = (120, 180)  # Focus range in μm
            num_steps = 8
            await controller.start_focus_session(focus_range, num_steps)

        elif choice == "2":
            await controller.interactive_focus_session()

        elif choice == "3":
            print("Goodbye!")

        else:
            print("Invalid choice")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())