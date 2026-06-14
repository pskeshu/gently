"""
Tests for bulk transfer wire protocol — send_file / receive_file.
"""

import asyncio
import hashlib

import pytest

from gently.mesh.transfer.protocol import (
    receive_file,
    send_file,
)


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample file for transfer tests."""
    f = tmp_path / "test_data.bin"
    content = b"Hello, this is test data for transfer protocol!\n" * 100
    f.write_bytes(content)
    return f


@pytest.fixture
def dest_dir(tmp_path):
    """Destination directory for received files."""
    d = tmp_path / "received"
    d.mkdir()
    return d


class TestSendReceiveRoundTrip:
    @pytest.mark.asyncio
    async def test_basic_transfer(self, sample_file, dest_dir):
        """Send a file through a TCP loopback and verify it arrives intact."""
        received_result = {}

        async def server_handler(reader, writer):
            header, path, sha = await receive_file(reader, dest_dir)
            received_result["header"] = header
            received_result["path"] = path
            received_result["sha256"] = sha
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(server_handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            success, sha256_sent = await send_file(
                writer,
                sample_file,
                transfer_id="test-001",
                auth_token="tok123",
                chunk_size=1024,
            )
            writer.close()
            await writer.wait_closed()

            # Wait for server to process
            await asyncio.sleep(0.1)

            assert success is True
            assert sha256_sent != ""
            assert received_result["header"] is not None
            assert received_result["header"]["transfer_id"] == "test-001"
            assert received_result["header"]["auth_token"] == "tok123"

            # Verify file content matches
            original = sample_file.read_bytes()
            received = received_result["path"].read_bytes()
            assert original == received

            # Verify SHA-256
            expected_sha = hashlib.sha256(original).hexdigest()
            assert sha256_sent == expected_sha
            assert received_result["sha256"] == expected_sha
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_small_file(self, tmp_path, dest_dir):
        """Transfer a very small file (< chunk size)."""
        small_file = tmp_path / "tiny.txt"
        small_file.write_bytes(b"tiny")

        received = {}

        async def handler(reader, writer):
            h, p, s = await receive_file(reader, dest_dir)
            received["path"] = p
            received["sha"] = s
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            ok, sha = await send_file(
                writer,
                small_file,
                transfer_id="tiny-01",
                chunk_size=4096,
            )
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.1)

            assert ok is True
            assert received["path"].read_bytes() == b"tiny"
        finally:
            server.close()
            await server.wait_closed()

    @pytest.mark.asyncio
    async def test_large_file(self, tmp_path, dest_dir):
        """Transfer a file larger than chunk size."""
        large_file = tmp_path / "large.bin"
        data = bytes(range(256)) * 1000  # 256KB
        large_file.write_bytes(data)

        received = {}

        async def handler(reader, writer):
            h, p, s = await receive_file(reader, dest_dir)
            received["path"] = p
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            ok, _ = await send_file(
                writer,
                large_file,
                transfer_id="large-01",
                chunk_size=4096,
            )
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.1)

            assert ok is True
            assert received["path"].read_bytes() == data
        finally:
            server.close()
            await server.wait_closed()


class TestProtocolEdgeCases:
    @pytest.mark.asyncio
    async def test_receive_incomplete_header(self, dest_dir):
        """Connection closed before full header → returns None."""
        reader = asyncio.StreamReader()
        reader.feed_data(b"\x00\x00")  # only 2 bytes, need 4
        reader.feed_eof()
        header, path, sha = await receive_file(reader, dest_dir)
        assert header is None
        assert path is None

    @pytest.mark.asyncio
    async def test_header_fields(self, sample_file, dest_dir):
        """Verify header contains expected fields."""
        received = {}

        async def handler(reader, writer):
            h, p, s = await receive_file(reader, dest_dir)
            received["header"] = h
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            await send_file(
                writer,
                sample_file,
                transfer_id="hdr-test",
                auth_token="mytoken",
                chunk_size=4096,
            )
            writer.close()
            await writer.wait_closed()
            await asyncio.sleep(0.1)

            h = received["header"]
            assert h["transfer_id"] == "hdr-test"
            assert h["auth_token"] == "mytoken"
            assert h["total_size"] == sample_file.stat().st_size
            assert h["file_path"] == sample_file.name
        finally:
            server.close()
            await server.wait_closed()
