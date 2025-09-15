"""Simple MMCore server for gently DiSPIM"""
import os
import pymmcore
import rpyc
from rpyc.utils.server import ThreadedServer

def initialize_mmcore(mm_dir: str, config_file: str) -> pymmcore.CMMCore:
    """Initialize MMCore using gently's approach"""
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
    
    def __init__(self, core):
        super().__init__()
        self.core = core
    
    def on_connect(self, conn):
        print("Client connected")
    
    def on_disconnect(self, conn):
        print("Client disconnected")
    
    def exposed_get_core(self):
        """Expose the MMCore instance"""
        return self.core

def start_server(mm_dir: str, config_file: str, port: int = 18861, hostname: str = "localhost"):
    """Start the server"""
    print(f"Initializing MMCore with {config_file}")
    core = initialize_mmcore(mm_dir, config_file)
    
    print(f"Starting server on {hostname}:{port}")
    service = MMService(core)
    
    # Configure rpyc to allow all attributes
    from rpyc.core import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    config['allow_all_attrs'] = True
    
    server = ThreadedServer(service, hostname=hostname, port=port, protocol_config=config)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.close()
        core.reset()

if __name__ == "__main__":
    # Default config from gently
    mm_dir = "C:/Program Files/Micro-Manager-1.4"
    config_file = "MMConfig_tracking_screening.cfg"
    start_server(mm_dir, config_file)