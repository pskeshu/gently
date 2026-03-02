"""
Bulk Transfer Protocol — TCP-based transfer for large files between peers.

Resumable, authenticated transfers for datasets and model weights.
"""

from .models import TransferFile, TransferJob, TransferManifest, TransferStatus, TransferType
from .client import TransferClient
from .server import TransferService
from .tracker import TransferTracker

__all__ = [
    "TransferClient",
    "TransferFile",
    "TransferJob",
    "TransferManifest",
    "TransferService",
    "TransferStatus",
    "TransferTracker",
    "TransferType",
]
