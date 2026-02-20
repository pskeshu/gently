"""
HTTP client for fetching full peer status from remote Gently instances.

Uses aiohttp (already a project dependency) to GET /api/mesh/status
from each discovered peer.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

import aiohttp

from .models import PeerInfo

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 5  # seconds


class PeerClient:
    """Fetches full status from a peer's viz server over HTTP."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def fetch_peer_info(self, peer: PeerInfo) -> Optional[Dict[str, Any]]:
        """
        GET /api/mesh/status from a peer.

        Returns the parsed JSON dict on success, or None on failure.
        """
        await self._ensure_session()
        url = f"{peer.base_url}/api/mesh/status"

        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.debug(
                    f"Peer {peer.instance_id[:8]} returned HTTP {resp.status}"
                )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to fetch from {peer.instance_id[:8]}: {e}")

        return None

    async def close(self):
        """Clean up the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
