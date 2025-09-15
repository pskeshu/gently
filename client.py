"""Client for connecting to MMCore server"""
import rpyc

def get_mmc(hostname="localhost", port=18861):
    """Get MMCore instance from server"""
    conn = rpyc.connect(hostname, port)
    return conn.root.get_core()