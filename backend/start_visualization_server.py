"""
Visualization Server Launcher
Starts the Gently visualization server on port 8080
"""
import asyncio
import sys
sys.path.insert(0, '.')

from gently.visualization.server import VisualizationServer


def main():
    print("Starting Visualization Server on http://127.0.0.1:8080")
    server = VisualizationServer(host="127.0.0.1", port=8080)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(server.start())
        print("Visualization Server running. Press Ctrl+C to stop.")
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        loop.run_until_complete(server.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
