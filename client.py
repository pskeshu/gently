"""Client for connecting to MMCore server"""
import rpyc

def get_mmc(hostname="10.103.30.131", port=18861):
    """Get MMCore instance from server"""
    conn = rpyc.connect(hostname, port)
    return conn.root.get_core()