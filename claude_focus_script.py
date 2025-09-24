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

# Import Claude Code SDK (would need to be installed)
try:
    from claude_code_sdk import ClaudeCodeClient, ClaudeCodeOptions
    SDK_AVAILABLE = True
except ImportError:
    print("Claude Code SDK not available - this is a demonstration script")
    SDK_AVAILABLE = False

class ClaudeFocusController:
    """Claude-powered focus controller using Claude Code SDK"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        self.focus_session_active = False

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required")

    async def initialize_claude(self):
        """Initialize Claude Code client with microscope tools"""
        if not SDK_AVAILABLE:
            print("Claude Code SDK not available - simulating responses")
            return

        # Configure Claude Code with microscope tools
        options = ClaudeCodeOptions(
            tools=MICROSCOPE_TOOLS,
            allow_tool_execution=True
        )

        self.client = ClaudeCodeClient(
            api_key=self.api_key,
            options=options
        )

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

Your task is to:
1. Check the current microscope status
2. Perform a focus sweep from {focus_range[0]} to {focus_range[1]} μm in {num_steps} steps
3. At each position, capture an image and evaluate the focus quality of the embryo boundary
4. Determine the best focus position based on edge sharpness
5. Move to the optimal focus position

Focus evaluation criteria for this bottom camera view:
- Sharp, well-defined embryo boundary/outline
- Good contrast between embryo and background
- Minimal blur or soft edges around the embryo perimeter
- Clear distinction of the embryo shape

Please start by checking the microscope status, then begin the focus sweep.
Remember to focus on analyzing the sharpness of the embryo's outer boundary, not internal details.
"""

        if SDK_AVAILABLE:
            try:
                response = await self.client.query(focus_prompt)
                print("Claude response:", response)
                return True
            except Exception as e:
                print(f"Error communicating with Claude: {e}")
                return False
        else:
            print("Would send to Claude:", focus_prompt)
            return await self.simulate_focus_session(focus_range, num_steps)

    async def simulate_focus_session(self, focus_range: tuple, num_steps: int):
        """Simulate Claude-guided focusing when SDK not available"""
        print("\n=== Simulating Claude Focus Session ===")

        from claude_focus_tools import _microscope_state

        # Simulate Claude checking status
        print("\nClaude: Let me check the microscope status first...")
        await asyncio.sleep(1)

        if _microscope_state['connected']:
            z_stage = _microscope_state['z_stage']
            current_z = z_stage.read()[z_stage.name]['value']
            print(f"Claude: I can see the microscope is connected. Current Z position is {current_z:.2f} μm.")
            print(f"Claude: I'll now perform a focus sweep from {focus_range[0]} to {focus_range[1]} μm to find the sharpest embryo boundary.")

        # Simulate focus sweep
        import numpy as np
        positions = np.linspace(focus_range[0], focus_range[1], num_steps)

        focus_scores = []
        for i, pos in enumerate(positions):
            print(f"\nClaude: Moving to position {pos:.2f} μm ({i+1}/{num_steps})...")

            # Move stage (this would be real)
            z_stage = _microscope_state['z_stage']
            status = z_stage.set(pos)
            while not status.done:
                await asyncio.sleep(0.1)

            # Capture image (this would be real)
            camera = _microscope_state['camera']
            img_status = camera.trigger()
            while not img_status.done:
                await asyncio.sleep(0.1)

            # Simulate Claude's visual analysis
            await asyncio.sleep(0.5)  # Simulate analysis time

            # Generate simulated focus score (peak around middle of range)
            optimal_pos = (focus_range[0] + focus_range[1]) / 2
            score = 100 - abs(pos - optimal_pos) * 2 + np.random.normal(0, 5)
            focus_scores.append((pos, score))

            print(f"Claude: Analyzing embryo boundary sharpness at {pos:.2f} μm...")
            if score > 85:
                print("Claude: Excellent focus! The embryo boundary is very sharp and well-defined.")
            elif score > 70:
                print("Claude: Good focus. The embryo outline is clear with minor softness.")
            elif score > 50:
                print("Claude: Moderate focus. The embryo boundary shows some blur.")
            else:
                print("Claude: Poor focus. The embryo appears very blurry with soft edges.")

        # Find best position
        best_pos, best_score = max(focus_scores, key=lambda x: x[1])

        print(f"\nClaude: Analysis complete! Best focus found at {best_pos:.2f} μm (score: {best_score:.1f})")
        print("Claude: The embryo boundary appears sharpest at this position.")
        print("Claude: Moving to optimal focus position...")

        # Move to best position
        status = z_stage.set(best_pos)
        while not status.done:
            await asyncio.sleep(0.1)

        print(f"Claude: Focus complete! Microscope positioned at {best_pos:.2f} μm for optimal embryo boundary sharpness.")

        return True

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

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            if SDK_AVAILABLE:
                try:
                    response = await self.client.query(user_input)
                    print(f"Claude: {response}")
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print("Claude: [SDK not available - would respond to:", user_input, "]")

async def main():
    """Main function to run Claude focus demonstration"""

    print("Claude-Powered Microscope Focus Demonstration")
    print("Bottom Camera View - Embryo Boundary Focus")
    print("=" * 50)

    # Check if API key is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key and SDK_AVAILABLE:
        print("Warning: ANTHROPIC_API_KEY not found. Some features may not work.")

    try:
        controller = ClaudeFocusController(api_key)

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