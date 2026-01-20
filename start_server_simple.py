"""Simple MMCore server for testing - exposes MMCore directly via rpyc"""
import os
import yaml
import pymmcore
import rpyc
from rpyc.utils.server import ThreadedServer


def load_config(config_path="config.yml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['mmdirectory'], config['mmconfig']


def initialize_mmcore(mm_dir: str, config_file: str) -> pymmcore.CMMCore:
    """Initialize MMCore"""
    core = pymmcore.CMMCore()
    core.enableStderrLog(True)

    # Setup MM environment
    os.environ["PATH"] += os.pathsep.join(["", mm_dir])
    core.setDeviceAdapterSearchPaths([mm_dir])

    # Load configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    core.loadSystemConfiguration(config_file)
    return core


class MMService(rpyc.Service):
    """Simple RPC service for MMCore access"""

    _core = None  # Class-level shared core

    def on_connect(self, conn):
        print("Client connected")

    def on_disconnect(self, conn):
        print("Client disconnected")

    def exposed_get_core(self):
        """Expose the MMCore instance"""
        return MMService._core


def start_server(port: int = 18861, hostname: str = "localhost"):
    """Start the server"""
    print("=" * 60)
    print("SIMPLE MMCORE SERVER")
    print("=" * 60)

    # Load config
    mm_dir, config_file = load_config()
    print(f"\nMM Directory: {mm_dir}")
    print(f"Config file: {config_file}")

    print("\nInitializing MMCore...")
    core = initialize_mmcore(mm_dir, config_file)
    MMService._core = core

    # Configure rpyc
    config = {
        'allow_all_attrs': True,
        'allow_pickle': True,
        'sync_request_timeout': 300,
    }

    print(f"\nStarting server on {hostname}:{port}")
    server = ThreadedServer(
        MMService,
        hostname=hostname,
        port=port,
        protocol_config=config
    )

    print("\n" + "=" * 60)
    print(f"Server ready at {hostname}:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.close()
        core.reset()
        print("Server stopped.")


if __name__ == "__main__":
    start_server()
