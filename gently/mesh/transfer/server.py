"""
TransferService — TCP server for receiving bulk transfers.
"""

import asyncio
import logging
import uuid
from pathlib import Path

from ...core.event_bus import EventType, get_event_bus
from ...core.service import Service
from ...settings import settings
from .protocol import receive_file

logger = logging.getLogger(__name__)


class TransferService(Service):
    """TCP server accepting incoming bulk transfers.

    Parameters
    ----------
    dest_dir : Path
        Directory to write received files.
    pairing_manager : optional
        For auth token verification.
    port : int
        TCP port to listen on.
    """

    def __init__(
        self,
        dest_dir: Path,
        pairing_manager=None,
        port: int = settings.transfer.transfer_port,
    ):
        super().__init__(
            name="transfer",
            service_type="transfer",
            host="0.0.0.0",
            port=port,
        )
        self._dest_dir = dest_dir
        self._pairing_manager = pairing_manager
        self._port = port
        self._server: asyncio.AbstractServer | None = None
        self._active_transfers: dict = {}

    async def on_start(self):
        self._server = await asyncio.start_server(
            self._handle_connection,
            "0.0.0.0",
            self._port,
        )
        logger.info(f"TransferService listening on port {self._port}")

    async def on_stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
            logger.info("TransferService stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle an incoming transfer connection."""
        addr = writer.get_extra_info("peername")
        transfer_id = str(uuid.uuid4())[:8]
        bus = get_event_bus()

        try:
            bus.publish(
                EventType.TRANSFER_STARTED,
                {"transfer_id": transfer_id, "direction": "receive", "peer": str(addr)},
                source="transfer_service",
            )

            header, file_path, sha256 = await receive_file(reader, self._dest_dir)

            if header is None:
                bus.publish(
                    EventType.TRANSFER_FAILED,
                    {"transfer_id": transfer_id, "error": "Incomplete transfer"},
                    source="transfer_service",
                )
                return

            # Verify auth
            auth_token = header.get("auth_token", "")
            if self._pairing_manager and auth_token:
                peer_id = self._pairing_manager.verify_token(auth_token)
                if not peer_id:
                    logger.warning(f"Transfer auth failed from {addr}")
                    bus.publish(
                        EventType.TRANSFER_FAILED,
                        {"transfer_id": transfer_id, "error": "Authentication failed"},
                        source="transfer_service",
                    )
                    return

            bus.publish(
                EventType.TRANSFER_COMPLETED,
                {
                    "transfer_id": transfer_id,
                    "file_path": str(file_path),
                    "sha256": sha256,
                    "total_size": header.get("total_size", 0),
                },
                source="transfer_service",
            )
            logger.info(
                f"Transfer {transfer_id} complete: {file_path} "
                f"({header.get('total_size', 0)} bytes, sha256={sha256[:12]}...)"
            )

        except Exception as e:
            logger.error(f"Transfer error: {e}")
            bus.publish(
                EventType.TRANSFER_FAILED,
                {"transfer_id": transfer_id, "error": str(e)},
                source="transfer_service",
            )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    @property
    def active_transfer_count(self) -> int:
        return len(self._active_transfers)
