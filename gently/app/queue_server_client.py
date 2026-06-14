"""Backward-compatibility shim — client moved to gently.hardware.dispim.client."""

from gently.hardware.dispim.client import (
    DiSPIMMicroscope,
    QueueServerClient,
    create_queue_server_client,
)

__all__ = ["DiSPIMMicroscope", "QueueServerClient", "create_queue_server_client"]
