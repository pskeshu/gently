"""
Binary wire protocol for file transfer over TCP.

Format:
    [4-byte header length][JSON header][raw chunks]

Header JSON:
    {
        "transfer_id": str,
        "file_path": str,
        "offset": int,       # for resume
        "chunk_size": int,
        "total_size": int,
        "sha256": str,
        "auth_token": str,
    }
"""

import asyncio
import hashlib
import json
import logging
import struct
from pathlib import Path

from ...settings import settings

logger = logging.getLogger(__name__)

HEADER_LENGTH_FORMAT = "!I"  # 4-byte unsigned int, network byte order
HEADER_LENGTH_SIZE = struct.calcsize(HEADER_LENGTH_FORMAT)


async def send_file(
    writer: asyncio.StreamWriter,
    file_path: Path,
    transfer_id: str,
    auth_token: str = "",
    offset: int = 0,
    chunk_size: int = 0,
) -> tuple[bool, str]:
    """Send a file over a TCP connection.

    Parameters
    ----------
    writer : asyncio.StreamWriter
        TCP writer.
    file_path : Path
        File to send.
    transfer_id : str
        Transfer job ID.
    auth_token : str
        Authentication token.
    offset : int
        Byte offset to resume from.
    chunk_size : int
        Chunk size (default from settings).

    Returns
    -------
    (success, sha256_hex)
    """
    if chunk_size == 0:
        chunk_size = settings.transfer.chunk_size

    file_size = file_path.stat().st_size
    hasher = hashlib.sha256()

    # Build header
    header = {
        "transfer_id": transfer_id,
        "file_path": str(file_path.name),
        "offset": offset,
        "chunk_size": chunk_size,
        "total_size": file_size,
        "sha256": "",  # filled after sending
        "auth_token": auth_token,
    }
    header_bytes = json.dumps(header).encode("utf-8")

    # Send header length + header
    writer.write(struct.pack(HEADER_LENGTH_FORMAT, len(header_bytes)))
    writer.write(header_bytes)
    await writer.drain()

    # Send file data
    bytes_sent = 0
    try:
        with open(file_path, "rb") as f:
            if offset > 0:
                f.seek(offset)
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hasher.update(chunk)
                writer.write(chunk)
                await writer.drain()
                bytes_sent += len(chunk)
    except Exception as e:
        logger.error(f"Send error: {e}")
        return False, ""

    return True, hasher.hexdigest()


async def receive_file(
    reader: asyncio.StreamReader,
    dest_dir: Path,
) -> tuple[dict | None, Path | None, str]:
    """Receive a file over a TCP connection.

    Parameters
    ----------
    reader : asyncio.StreamReader
        TCP reader.
    dest_dir : Path
        Directory to write the file to.

    Returns
    -------
    (header_dict, file_path, sha256_hex) or (None, None, "") on error.
    """
    try:
        # Read header length
        length_bytes = await reader.readexactly(HEADER_LENGTH_SIZE)
        header_length = struct.unpack(HEADER_LENGTH_FORMAT, length_bytes)[0]

        # Read header
        header_bytes = await reader.readexactly(header_length)
        header = json.loads(header_bytes.decode("utf-8"))

        file_name = header.get("file_path", "received")
        total_size = header.get("total_size", 0)
        offset = header.get("offset", 0)
        chunk_size = header.get("chunk_size", settings.transfer.chunk_size)

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_name

        hasher = hashlib.sha256()
        bytes_received = 0
        remaining = total_size - offset

        mode = "ab" if offset > 0 else "wb"
        with open(dest_path, mode) as f:
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                chunk = await reader.read(to_read)
                if not chunk:
                    break
                f.write(chunk)
                hasher.update(chunk)
                bytes_received += len(chunk)
                remaining -= len(chunk)

        return header, dest_path, hasher.hexdigest()

    except asyncio.IncompleteReadError:
        logger.warning("Incomplete transfer — connection closed early")
        return None, None, ""
    except Exception as e:
        logger.error(f"Receive error: {e}")
        return None, None, ""
