"""
TransferClient — Initiates outbound bulk transfers.
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path

from ...core.event_bus import EventType, get_event_bus
from .models import TransferJob, TransferStatus, TransferType
from .protocol import send_file

logger = logging.getLogger(__name__)


class TransferClient:
    """Initiates outbound transfers to remote peers.

    Parameters
    ----------
    pairing_manager : optional
        For getting auth tokens.
    """

    def __init__(self, pairing_manager=None):
        self._pairing_manager = pairing_manager
        self._active_jobs: dict = {}

    def _get_token(self, peer_instance_id: str) -> str:
        """Get auth token for a peer."""
        if self._pairing_manager:
            return self._pairing_manager.get_token_for_peer(peer_instance_id) or ""
        return ""

    async def send_dataset(
        self,
        peer_ip: str,
        peer_port: int,
        peer_instance_id: str,
        file_paths: list[Path],
        session_id: str = "",
    ) -> TransferJob:
        """Send dataset files to a peer.

        Parameters
        ----------
        peer_ip : str
            Remote peer IP address.
        peer_port : int
            Remote peer transfer port.
        peer_instance_id : str
            Remote peer instance ID (for auth).
        file_paths : list of Path
            Files to send.
        session_id : str
            Session ID being transferred.

        Returns
        -------
        TransferJob
        """
        job = TransferJob(
            id=str(uuid.uuid4())[:8],
            transfer_type=TransferType.DATASET.value,
            direction="send",
            peer_instance_id=peer_instance_id,
            session_id=session_id,
            status=TransferStatus.TRANSFERRING.value,
            started_at=time.time(),
        )

        bus = get_event_bus()
        token = self._get_token(peer_instance_id)

        try:
            for file_path in file_paths:
                if not file_path.exists():
                    continue

                job.total_bytes += file_path.stat().st_size

                reader, writer = await asyncio.open_connection(peer_ip, peer_port)
                try:
                    success, sha256 = await send_file(
                        writer,
                        file_path,
                        job.id,
                        auth_token=token,
                    )
                    if success:
                        job.bytes_transferred += file_path.stat().st_size
                        bus.publish(
                            EventType.TRANSFER_PROGRESS,
                            {"transfer_id": job.id, "progress_pct": job.progress_pct},
                            source="transfer_client",
                        )
                finally:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except Exception:
                        pass

            job.status = TransferStatus.COMPLETED.value
            job.completed_at = time.time()
            bus.publish(
                EventType.TRANSFER_COMPLETED,
                job.to_dict(),
                source="transfer_client",
            )

        except Exception as e:
            job.status = TransferStatus.FAILED.value
            job.error = str(e)
            bus.publish(
                EventType.TRANSFER_FAILED,
                {"transfer_id": job.id, "error": str(e)},
                source="transfer_client",
            )

        return job

    async def send_model_weights(
        self,
        peer_ip: str,
        peer_port: int,
        peer_instance_id: str,
        weights_path: Path,
        pipeline_id: str = "",
    ) -> TransferJob:
        """Send model weights to a peer."""
        return await self.send_dataset(
            peer_ip=peer_ip,
            peer_port=peer_port,
            peer_instance_id=peer_instance_id,
            file_paths=[weights_path],
            session_id="",
        )

    async def request_dataset(
        self,
        peer_ip: str,
        peer_port: int,
        peer_instance_id: str,
        session_id: str,
    ) -> TransferJob:
        """Request a dataset from a peer (pull mode).

        This sends a pull request via the HTTP control plane,
        then the remote peer initiates a push transfer.
        """
        # Pull mode is orchestrated via the HTTP API, not raw TCP
        # The caller should POST /api/mesh/transfers/request first
        job = TransferJob(
            id=str(uuid.uuid4())[:8],
            transfer_type=TransferType.DATASET.value,
            direction="receive",
            peer_instance_id=peer_instance_id,
            session_id=session_id,
            status=TransferStatus.PENDING.value,
        )
        return job
