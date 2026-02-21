"""
HTTP client for fetching full peer status from remote Gently instances.

Uses aiohttp (already a project dependency) to GET /api/mesh/status
from each discovered peer.

Phase 3: TLS certificate fingerprint pinning on authenticated requests.
"""

import asyncio
import logging
import ssl
from typing import Any, Dict, List, Optional

import aiohttp

from .models import PeerInfo

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 5  # seconds


class PeerClient:
    """Fetches full status from a peer's viz server over HTTP."""

    def __init__(self, pairing_manager=None, audit_log=None):
        self._session: Optional[aiohttp.ClientSession] = None
        self._pairing_manager = pairing_manager
        self._audit_log = audit_log
        self._pinning_verified: set = set()  # track first-success per peer

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
            # Use permissive SSL context — we verify by cert fingerprint, not CA chain
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            self._pinning_verified.clear()

    def _auth_headers(self, peer: PeerInfo) -> Dict[str, str]:
        """Build auth headers for a trusted peer."""
        if self._pairing_manager is None:
            return {}
        token = self._pairing_manager.get_token_for_peer(peer.instance_id)
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

    def _ssl_for_peer(self, peer: PeerInfo):
        """
        Determine SSL verification mode for a peer.

        For trusted peers with a stored cert fingerprint, returns an
        aiohttp.Fingerprint that verifies the peer's TLS cert SHA-256
        hash on each request. Otherwise returns False (use connector default).
        """
        if self._pairing_manager is None:
            return False
        fingerprint = self._pairing_manager.get_cert_fingerprint_for_peer(
            peer.instance_id
        )
        if fingerprint:
            try:
                return aiohttp.Fingerprint(bytes.fromhex(fingerprint))
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid cert fingerprint for {peer.instance_id[:8]}, "
                    "falling back to unpinned"
                )
        return False

    def _log_pinning_success(self, peer: PeerInfo, ssl_fp):
        """Log cert pinning success on first verified request per peer."""
        if ssl_fp is not False and peer.instance_id not in self._pinning_verified:
            self._pinning_verified.add(peer.instance_id)
            if self._audit_log:
                from .audit import AuditEvent
                self._audit_log.log(
                    AuditEvent.CERT_PIN_OK, outcome="allow",
                    peer_id=peer.instance_id, ip=peer.ip_address,
                )

    def _log_pinning_failure(self, peer: PeerInfo, error):
        """Log cert pinning failure."""
        logger.warning(f"CERT PINNING FAILED for {peer.instance_id[:8]}: {error}")
        if self._audit_log:
            from .audit import AuditEvent
            self._audit_log.log(
                AuditEvent.CERT_PIN_FAIL, outcome="deny",
                peer_id=peer.instance_id, ip=peer.ip_address,
                detail=str(error),
            )

    async def fetch_peer_info(self, peer: PeerInfo) -> Optional[Dict[str, Any]]:
        """
        GET /api/mesh/status from a peer.

        Returns the parsed JSON dict on success, or None on failure.
        """
        await self._ensure_session()
        url = f"{peer.base_url}/api/mesh/status"
        headers = self._auth_headers(peer)
        ssl_fp = self._ssl_for_peer(peer)
        try:
            async with self._session.get(url, headers=headers, ssl=ssl_fp) as resp:
                if resp.status == 200:
                    self._log_pinning_success(peer, ssl_fp)
                    return await resp.json()
                logger.debug(
                    f"Peer {peer.instance_id[:8]} returned HTTP {resp.status}"
                )
        except aiohttp.ServerFingerprintMismatch as e:
            self._log_pinning_failure(peer, e)
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
        headers = self._auth_headers(peer)
        ssl_fp = self._ssl_for_peer(peer)
        try:
            async with self._session.get(url, headers=headers, ssl=ssl_fp) as resp:
                if resp.status == 200:
                    self._log_pinning_success(peer, ssl_fp)
                    data = await resp.json()
                    return data.get("campaigns", [])
        except aiohttp.ServerFingerprintMismatch as e:
            self._log_pinning_failure(peer, e)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to fetch campaigns from {peer.instance_id[:8]}: {e}")
        return None

    async def fetch_campaign_export(self, peer: PeerInfo, campaign_id: str) -> Optional[Dict]:
        """GET /api/campaigns/{id}/export from a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/export"
        headers = self._auth_headers(peer)
        ssl_fp = self._ssl_for_peer(peer)
        try:
            async with self._session.get(url, headers=headers, ssl=ssl_fp) as resp:
                if resp.status == 200:
                    self._log_pinning_success(peer, ssl_fp)
                    return await resp.json()
        except aiohttp.ServerFingerprintMismatch as e:
            self._log_pinning_failure(peer, e)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to fetch campaign export from {peer.instance_id[:8]}: {e}")
        return None

    async def join_campaign(
        self, peer: PeerInfo, campaign_id: str, instance_id: str, hostname: str,
    ) -> bool:
        """POST /api/campaigns/{id}/join on a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/join"
        headers = self._auth_headers(peer)
        ssl_fp = self._ssl_for_peer(peer)
        try:
            async with self._session.post(url, json={
                "instance_id": instance_id,
                "hostname": hostname,
            }, headers=headers, ssl=ssl_fp) as resp:
                if resp.status == 200:
                    self._log_pinning_success(peer, ssl_fp)
                return resp.status == 200
        except aiohttp.ServerFingerprintMismatch as e:
            self._log_pinning_failure(peer, e)
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
        headers = self._auth_headers(peer)
        ssl_fp = self._ssl_for_peer(peer)
        try:
            async with self._session.post(url, json={
                "instance_id": instance_id,
                "hostname": hostname,
            }, headers=headers, ssl=ssl_fp) as resp:
                if resp.status == 200:
                    self._log_pinning_success(peer, ssl_fp)
                return resp.status == 200
        except aiohttp.ServerFingerprintMismatch as e:
            self._log_pinning_failure(peer, e)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to claim item on {peer.instance_id[:8]}: {e}")
        return False

    async def unclaim_item(
        self, peer: PeerInfo, campaign_id: str, item_id: str,
    ) -> bool:
        """POST /api/campaigns/{id}/items/{item_id}/unclaim on a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/campaigns/{campaign_id}/items/{item_id}/unclaim"
        headers = self._auth_headers(peer)
        ssl_fp = self._ssl_for_peer(peer)
        try:
            async with self._session.post(url, headers=headers, ssl=ssl_fp) as resp:
                if resp.status == 200:
                    self._log_pinning_success(peer, ssl_fp)
                return resp.status == 200
        except aiohttp.ServerFingerprintMismatch as e:
            self._log_pinning_failure(peer, e)
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
        headers = self._auth_headers(peer)
        ssl_fp = self._ssl_for_peer(peer)
        try:
            async with self._session.post(url, json=body, headers=headers, ssl=ssl_fp) as resp:
                if resp.status == 200:
                    self._log_pinning_success(peer, ssl_fp)
                return resp.status == 200
        except aiohttp.ServerFingerprintMismatch as e:
            self._log_pinning_failure(peer, e)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to update item status on {peer.instance_id[:8]}: {e}")
        return False

    # ------------------------------------------------------------------
    # Pairing methods (no auth headers — these bootstrap trust)
    # ------------------------------------------------------------------

    async def send_pair_request(
        self, peer: PeerInfo, initiator_id: str, hostname: str, nonce: str,
        cert_fingerprint: str = "", udp_sign_key: str = "",
    ) -> Optional[Dict]:
        """POST /api/mesh/pair — initiate pairing with a peer."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/mesh/pair"
        try:
            async with self._session.post(url, json={
                "initiator_id": initiator_id,
                "hostname": hostname,
                "nonce": nonce,
                "cert_fingerprint": cert_fingerprint,
                "udp_sign_key": udp_sign_key,
            }) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.debug(f"Pair request to {peer.instance_id[:8]} returned {resp.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to send pair request to {peer.instance_id[:8]}: {e}")
        return None

    async def poll_pair_status(self, peer: PeerInfo, pairing_id: str) -> Optional[Dict]:
        """GET /api/mesh/pair/{id}/status — poll pairing status."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/mesh/pair/{pairing_id}/status"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to poll pair status from {peer.instance_id[:8]}: {e}")
        return None

    async def confirm_pair_remote(
        self, peer: PeerInfo, pairing_id: str, confirmer_id: str,
    ) -> bool:
        """POST /api/mesh/pair/{id}/confirm — confirm pairing on remote side."""
        await self._ensure_session()
        url = f"{peer.base_url}/api/mesh/pair/{pairing_id}/confirm"
        try:
            async with self._session.post(url, json={
                "confirmer_id": confirmer_id,
            }) as resp:
                return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            logger.debug(f"Failed to confirm pairing on {peer.instance_id[:8]}: {e}")
        return False

    async def close(self):
        """Clean up the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
