"""
HTTP client for fetching full peer status from remote Gently instances.

Uses aiohttp (already a project dependency) to GET /api/mesh/status
from each discovered peer.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

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

    # ------------------------------------------------------------------
    # Campaign coordination methods
    # ------------------------------------------------------------------

    async def fetch_peer_campaigns(self, peer: PeerInfo) -> Optional[List]:
        """GET /api/campaigns from a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("campaigns", [])
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to fetch campaigns from {peer.instance_id[:8]}: {e}")
        return None

    async def fetch_campaign_export(self, peer: PeerInfo, campaign_id: str) -> Optional[Dict]:
        """GET /api/campaigns/{id}/export from a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/export"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to fetch campaign export from {peer.instance_id[:8]}: {e}")
        return None

    async def join_campaign(
        self, peer: PeerInfo, campaign_id: str, instance_id: str, hostname: str,
    ) -> bool:
        """POST /api/campaigns/{id}/join on a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/join"
        try:
            async with self._session.post(url, json={
                "instance_id": instance_id,
                "hostname": hostname,
            }) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to join campaign on {peer.instance_id[:8]}: {e}")
        return False

    async def claim_item(
        self,
        peer: PeerInfo,
        campaign_id: str,
        item_id: str,
        instance_id: str,
        hostname: str,
    ) -> bool:
        """POST /api/campaigns/{id}/items/{item_id}/claim on a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/items/{item_id}/claim"
        try:
            async with self._session.post(url, json={
                "instance_id": instance_id,
                "hostname": hostname,
            }) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to claim item on {peer.instance_id[:8]}: {e}")
        return False

    async def unclaim_item(
        self, peer: PeerInfo, campaign_id: str, item_id: str,
    ) -> bool:
        """POST /api/campaigns/{id}/items/{item_id}/unclaim on a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/items/{item_id}/unclaim"
        try:
            async with self._session.post(url) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to unclaim item on {peer.instance_id[:8]}: {e}")
        return False

    async def update_item_status(
        self,
        peer: PeerInfo,
        campaign_id: str,
        item_id: str,
        status: str,
        outcome: Optional[str] = None,
    ) -> bool:
        """POST /api/campaigns/{id}/items/{item_id}/status on a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/items/{item_id}/status"
        body: Dict[str, Any] = {"status": status}
        if outcome is not None:
            body["outcome"] = outcome
        try:
            async with self._session.post(url, json=body) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to update item status on {peer.instance_id[:8]}: {e}")
        return False

    async def close(self):
        """Clean up the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
