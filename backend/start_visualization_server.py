"""
Visualization Server Launcher
Starts the Gently visualization server on port 8080
"""
import asyncio
import signal
import sys
sys.path.insert(0, '.')

from gently.visualization.server import VisualizationServer


async def run_server():
    """Run the visualization server with proper signal handling."""
    server = VisualizationServer(host="127.0.0.1", port=8080)

    # Use the server's run_forever method which handles signals properly
    await server.run_forever()


def main():
    print("Starting Visualization Server on http://127.0.0.1:8080")
    print("Press Ctrl+C to stop.")

    # On Windows, we need special handling for Ctrl+C
    if sys.platform == 'win32':
        # Use ProactorEventLoop on Windows for better signal handling
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()
