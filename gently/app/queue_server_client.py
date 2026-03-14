"""Backward-compatibility shim — client moved to gently.hardware.dispim.client."""
from gently.hardware.dispim.client import QueueServerClient, create_queue_server_client

__all__ = ['QueueServerClient', 'create_queue_server_client']
