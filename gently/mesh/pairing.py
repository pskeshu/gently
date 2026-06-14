"""
Bluetooth-style mesh pairing — PIN verification + HMAC token auth.

Protocol:
1. A discovers B via UDP (unauthenticated)
2. A sends POST /api/mesh/pair with nonce_a
3. B generates nonce_b, both compute PIN = SHA256(sorted nonces)[:4] mod 1000000
4. User on B verifies PIN and /pair accept
5. Both derive token = HMAC-SHA256(key="gently-mesh-v1", msg=sorted nonces)
6. Token never transmitted — used as Bearer auth on all future requests

Phase 2 additions:
- TLS cert fingerprint exchange during pairing
- UDP signing key derivation and exchange
- Rate limiting on pairing attempts (exponential backoff)
"""

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PAIRING_EXPIRY = 120.0  # seconds before pending session expires
RATE_LIMIT_WINDOW = 3600.0  # 1 hour
RATE_LIMIT_MAX = 10  # max attempts per IP per window
RATE_LIMIT_MAX_BACKOFF = 60.0  # max backoff seconds

# Capability scopes
ALL_SCOPES = ["status", "campaigns", "campaigns:admin", "data", "ml"]
DEFAULT_SCOPES = list(ALL_SCOPES)


@dataclass
class PairingSession:
    """In-memory state for an active pairing negotiation."""

    pairing_id: str
    initiator_id: str
    initiator_hostname: str
    responder_id: str = ""
    responder_hostname: str = ""
    nonce_initiator: str = ""
    nonce_responder: str = ""
    pin: str = ""
    status: str = "pending"  # pending | confirmed | rejected | expired
    confirmed_by_initiator: bool = False
    confirmed_by_responder: bool = False
    created_at: float = field(default_factory=time.time)
    # Phase 2: cert fingerprints and UDP keys exchanged during pairing
    initiator_cert_fingerprint: str = ""
    initiator_udp_sign_key: str = ""
    responder_cert_fingerprint: str = ""
    responder_udp_sign_key: str = ""


@dataclass
class TrustedPeer:
    """Persisted trust record for a paired peer."""

    instance_id: str
    hostname: str
    base_token: str  # hex-encoded HMAC base secret (daily tokens derived from this)
    paired_at: str = ""  # ISO timestamp
    cert_fingerprint: str = ""  # SHA256 of peer's TLS cert (DER)
    udp_signing_key: str = ""  # hex-encoded key for UDP HMAC verification
    scopes: list[str] = field(default_factory=lambda: list(ALL_SCOPES))


class PairingManager:
    """
    Manages pairing sessions and persistent trust.

    Parameters
    ----------
    instance_id : str
        This node's stable UUID.
    hostname : str
        This node's hostname.
    config_dir : Path
        Directory for persistent config (mesh_trusted_peers.json).
    """

    def __init__(self, instance_id: str, hostname: str, config_dir: Path, audit_log=None):
        self.instance_id = instance_id
        self.hostname = hostname
        self._config_dir = config_dir
        self._audit_log = audit_log

        self._sessions: dict[str, PairingSession] = {}
        self._trusted: dict[str, TrustedPeer] = {}  # keyed by instance_id
        self._trust_file = config_dir / "mesh_trusted_peers.json"

        # Phase 2: TLS cert fingerprint (set by launch_gently after cert gen)
        self.cert_fingerprint: str = ""

        # Phase 2: UDP signing key (derived from instance_id)
        self.udp_sign_key: str = hmac.new(
            b"gently-mesh-udp-v1",
            instance_id.encode(),
            hashlib.sha256,
        ).hexdigest()

        # Phase 2: rate limiting state
        self._pair_attempts: dict[str, list[float]] = {}  # IP -> timestamps

        self._load_trusted()

    # ------------------------------------------------------------------
    # Crypto helpers (all pure, no side effects)
    # ------------------------------------------------------------------

    @staticmethod
    def generate_nonce() -> str:
        """Generate a 16-byte random nonce as hex."""
        return os.urandom(16).hex()

    @staticmethod
    def compute_pin(nonce_a: str, nonce_b: str) -> str:
        """Derive a 6-digit PIN from two nonces."""
        sorted_nonces = "".join(sorted([nonce_a, nonce_b]))
        digest = hashlib.sha256(sorted_nonces.encode()).digest()
        num = int.from_bytes(digest[:4], "big") % 1_000_000
        return f"{num:06d}"

    @staticmethod
    def _derive_token(nonce_a: str, nonce_b: str) -> str:
        """Derive shared secret token from nonces (never transmitted)."""
        sorted_nonces = "".join(sorted([nonce_a, nonce_b]))
        token = hmac.new(
            b"gently-mesh-v1",
            sorted_nonces.encode(),
            hashlib.sha256,
        ).hexdigest()
        return token

    @staticmethod
    def _derive_daily_token(base_token: str, epoch_day: int) -> str:
        """Derive a daily-rotating token from the base secret."""
        return hmac.new(
            base_token.encode(),
            str(epoch_day).encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _current_epoch_day() -> int:
        """Return current UTC day number (deterministic across peers)."""
        return int(time.time()) // 86400

    # ------------------------------------------------------------------
    # Trust queries
    # ------------------------------------------------------------------

    def is_trusted(self, instance_id: str) -> bool:
        """Check if a peer is trusted."""
        return instance_id in self._trusted

    def get_token_for_peer(self, instance_id: str) -> str | None:
        """Get the current daily auth token for a trusted peer."""
        tp = self._trusted.get(instance_id)
        if tp is None:
            return None
        epoch_day = self._current_epoch_day()
        return self._derive_daily_token(tp.base_token, epoch_day)

    def verify_token(self, token: str) -> str | None:
        """
        Check if a token matches any trusted peer (timing-safe).

        Returns the matching peer's instance_id, or None.
        Accepts tokens from today and yesterday (midnight boundary).
        """
        epoch_day = self._current_epoch_day()
        for tp in self._trusted.values():
            current = self._derive_daily_token(tp.base_token, epoch_day)
            previous = self._derive_daily_token(tp.base_token, epoch_day - 1)
            if hmac.compare_digest(current, token) or hmac.compare_digest(previous, token):
                return tp.instance_id
        return None

    def get_all_trusted(self) -> list[TrustedPeer]:
        """Return all trusted peers."""
        return list(self._trusted.values())

    def get_udp_key_for_peer(self, instance_id: str) -> str | None:
        """Get the UDP signing key for a trusted peer."""
        tp = self._trusted.get(instance_id)
        return tp.udp_signing_key if tp else None

    def get_cert_fingerprint_for_peer(self, instance_id: str) -> str | None:
        """Get the TLS cert fingerprint for a trusted peer."""
        tp = self._trusted.get(instance_id)
        return tp.cert_fingerprint if tp else None

    def get_scopes_for_peer(self, instance_id: str) -> list[str]:
        """Get the permission scopes for a trusted peer."""
        tp = self._trusted.get(instance_id)
        return list(tp.scopes) if tp else []

    def set_scopes(self, identifier: str, scopes: list[str]) -> bool:
        """
        Set permission scopes for a peer (by instance_id, prefix, or hostname).

        Returns True if the peer was found and updated.
        """
        # Validate scopes
        for s in scopes:
            if s not in ALL_SCOPES:
                return False

        # Find the peer (same resolution as unpair)
        target_id = None
        if identifier in self._trusted:
            target_id = identifier
        else:
            for iid in self._trusted:
                if iid.startswith(identifier):
                    target_id = iid
                    break
            if target_id is None:
                for iid, tp in self._trusted.items():
                    if tp.hostname.lower() == identifier.lower():
                        target_id = iid
                        break

        if target_id is None:
            return False

        self._trusted[target_id].scopes = list(scopes)
        self._save_trusted()
        logger.info(f"Scopes for {self._trusted[target_id].hostname} ({target_id[:8]}): {scopes}")
        return True

    def unpair(self, identifier: str) -> bool:
        """
        Remove trust by instance_id (full or prefix) or hostname.

        Returns True if a peer was removed.
        """
        removed_id = None

        # Try exact instance_id match
        if identifier in self._trusted:
            removed_id = identifier
        else:
            # Try prefix match on instance_id
            for iid in list(self._trusted.keys()):
                if iid.startswith(identifier):
                    removed_id = iid
                    break

            # Try hostname match (case-insensitive)
            if removed_id is None:
                for iid, tp in list(self._trusted.items()):
                    if tp.hostname.lower() == identifier.lower():
                        removed_id = iid
                        break

        if removed_id is not None:
            del self._trusted[removed_id]
            self._save_trusted()
            if self._audit_log:
                from .audit import AuditEvent

                self._audit_log.log(
                    AuditEvent.PEER_UNPAIRED,
                    outcome="info",
                    peer_id=removed_id,
                )
            return True

        return False

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def check_rate_limit(self, ip: str) -> tuple[bool, float]:
        """
        Check if a pairing attempt from this IP is allowed.

        Returns (allowed, retry_after_seconds).
        """
        now = time.time()
        attempts = self._pair_attempts.get(ip, [])

        # Prune old attempts outside the window
        attempts = [t for t in attempts if (now - t) < RATE_LIMIT_WINDOW]
        self._pair_attempts[ip] = attempts

        if len(attempts) >= RATE_LIMIT_MAX:
            retry_after = RATE_LIMIT_WINDOW - (now - attempts[0])
            if self._audit_log:
                from .audit import AuditEvent

                self._audit_log.log(
                    AuditEvent.RATE_LIMITED,
                    outcome="deny",
                    ip=ip,
                    detail=f"max_attempts={RATE_LIMIT_MAX}",
                )
            return False, max(retry_after, 1.0)

        # Exponential backoff between consecutive attempts
        if attempts:
            n = len(attempts)
            backoff = min(2 ** (n - 1), RATE_LIMIT_MAX_BACKOFF)
            elapsed = now - attempts[-1]
            if elapsed < backoff:
                if self._audit_log:
                    from .audit import AuditEvent

                    self._audit_log.log(
                        AuditEvent.RATE_LIMITED,
                        outcome="deny",
                        ip=ip,
                        detail=f"backoff={backoff:.1f}s",
                    )
                return False, backoff - elapsed

        return True, 0.0

    def record_attempt(self, ip: str):
        """Record a pairing attempt from an IP."""
        if ip not in self._pair_attempts:
            self._pair_attempts[ip] = []
        self._pair_attempts[ip].append(time.time())

    # ------------------------------------------------------------------
    # Initiator flow
    # ------------------------------------------------------------------

    def create_initiation(self) -> str:
        """Generate a nonce for initiating a pairing. Returns the nonce."""
        return self.generate_nonce()

    def process_initiation_response(
        self,
        peer_id: str,
        peer_hostname: str,
        nonce_local: str,
        nonce_remote: str,
        pairing_id: str,
    ) -> PairingSession:
        """
        Process the response from the responder and create a local session.

        Called after receiving {nonce, pairing_id} from POST /api/mesh/pair.
        """
        pin = self.compute_pin(nonce_local, nonce_remote)

        session = PairingSession(
            pairing_id=pairing_id,
            initiator_id=self.instance_id,
            initiator_hostname=self.hostname,
            responder_id=peer_id,
            responder_hostname=peer_hostname,
            nonce_initiator=nonce_local,
            nonce_responder=nonce_remote,
            pin=pin,
            confirmed_by_initiator=True,  # initiator auto-confirms
        )
        self._sessions[pairing_id] = session
        return session

    # ------------------------------------------------------------------
    # Responder flow
    # ------------------------------------------------------------------

    def handle_pair_request(
        self,
        initiator_id: str,
        initiator_hostname: str,
        nonce_initiator: str,
        initiator_cert_fingerprint: str = "",
        initiator_udp_sign_key: str = "",
    ) -> PairingSession:
        """
        Handle an incoming pairing request. Generates own nonce and creates session.

        Returns the session (caller sends back {nonce, pairing_id, status}).
        """
        nonce_responder = self.generate_nonce()
        pin = self.compute_pin(nonce_initiator, nonce_responder)
        pairing_id = str(uuid.uuid4())

        session = PairingSession(
            pairing_id=pairing_id,
            initiator_id=initiator_id,
            initiator_hostname=initiator_hostname,
            responder_id=self.instance_id,
            responder_hostname=self.hostname,
            nonce_initiator=nonce_initiator,
            nonce_responder=nonce_responder,
            pin=pin,
            initiator_cert_fingerprint=initiator_cert_fingerprint,
            initiator_udp_sign_key=initiator_udp_sign_key,
            responder_cert_fingerprint=self.cert_fingerprint,
            responder_udp_sign_key=self.udp_sign_key,
        )
        self._sessions[pairing_id] = session
        return session

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------

    def confirm_pairing(self, pairing_id: str, confirmer_id: str) -> PairingSession | None:
        """
        Mark one side as confirmed.

        When both sides are confirmed, finalizes the pairing (derives token,
        persists trust).

        Returns the session, or None if not found.
        """
        session = self._sessions.get(pairing_id)
        if not session or session.status != "pending":
            return session

        if confirmer_id == session.initiator_id:
            session.confirmed_by_initiator = True
        elif confirmer_id == session.responder_id:
            session.confirmed_by_responder = True

        if session.confirmed_by_initiator and session.confirmed_by_responder:
            self._finalize_pairing(session)

        return session

    def reject_pairing(self, pairing_id: str) -> PairingSession | None:
        """Reject a pending pairing session."""
        session = self._sessions.get(pairing_id)
        if session and session.status == "pending":
            session.status = "rejected"
            if self._audit_log:
                from .audit import AuditEvent

                self._audit_log.log(
                    AuditEvent.PAIR_REJECTED,
                    outcome="deny",
                    peer_id=session.initiator_id,
                )
        return session

    def get_session(self, pairing_id: str) -> PairingSession | None:
        """Get a pairing session by ID."""
        return self._sessions.get(pairing_id)

    def get_pending_sessions(self) -> list[PairingSession]:
        """Get all pending pairing sessions (for /pair accept)."""
        return [
            s
            for s in self._sessions.values()
            if s.status == "pending" and s.responder_id == self.instance_id
        ]

    def cleanup_expired(self):
        """Remove expired pending sessions."""
        now = time.time()
        expired = [
            pid
            for pid, s in self._sessions.items()
            if s.status == "pending" and (now - s.created_at) > PAIRING_EXPIRY
        ]
        for pid in expired:
            self._sessions[pid].status = "expired"
            logger.debug(f"Pairing session {pid[:8]} expired")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _finalize_pairing(self, session: PairingSession):
        """Derive token, persist trust, mark session confirmed."""
        token = self._derive_token(session.nonce_initiator, session.nonce_responder)
        session.status = "confirmed"

        # Determine which side is the remote peer
        if session.initiator_id == self.instance_id:
            peer_id = session.responder_id
            peer_hostname = session.responder_hostname
            peer_cert_fp = session.responder_cert_fingerprint
            peer_udp_key = session.responder_udp_sign_key
        else:
            peer_id = session.initiator_id
            peer_hostname = session.initiator_hostname
            peer_cert_fp = session.initiator_cert_fingerprint
            peer_udp_key = session.initiator_udp_sign_key

        self._trusted[peer_id] = TrustedPeer(
            instance_id=peer_id,
            hostname=peer_hostname,
            base_token=token,
            paired_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            cert_fingerprint=peer_cert_fp,
            udp_signing_key=peer_udp_key,
        )
        self._save_trusted()
        logger.info(f"Paired with {peer_hostname} ({peer_id[:8]})")
        if self._audit_log:
            from .audit import AuditEvent

            self._audit_log.log(
                AuditEvent.PAIR_COMPLETED,
                outcome="info",
                peer_id=peer_id,
                detail=f"hostname={peer_hostname}",
            )

    def _load_trusted(self):
        """Load trusted peers from disk."""
        if not self._trust_file.exists():
            return
        try:
            data = json.loads(self._trust_file.read_text())
            for entry in data:
                # Backward compat: read "base_token" or fall back to "token"
                base_token = entry.get("base_token") or entry.get("token", "")
                tp = TrustedPeer(
                    instance_id=entry["instance_id"],
                    hostname=entry.get("hostname", ""),
                    base_token=base_token,
                    paired_at=entry.get("paired_at", ""),
                    cert_fingerprint=entry.get("cert_fingerprint", ""),
                    udp_signing_key=entry.get("udp_signing_key", ""),
                    scopes=entry.get("scopes", list(ALL_SCOPES)),
                )
                self._trusted[tp.instance_id] = tp
            logger.info(f"Loaded {len(self._trusted)} trusted peers")
        except Exception as e:
            logger.warning(f"Failed to load trusted peers: {e}")

    def _save_trusted(self):
        """Persist trusted peers to disk."""
        data = [
            {
                "instance_id": tp.instance_id,
                "hostname": tp.hostname,
                "base_token": tp.base_token,
                "paired_at": tp.paired_at,
                "cert_fingerprint": tp.cert_fingerprint,
                "udp_signing_key": tp.udp_signing_key,
                "scopes": tp.scopes,
            }
            for tp in self._trusted.values()
        ]
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            self._trust_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save trusted peers: {e}")
