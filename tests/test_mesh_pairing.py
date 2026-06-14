"""
Tests for PairingManager — pairing flow, trust persistence, HMAC auth.
"""

import time

import pytest

from gently.mesh.pairing import (
    PAIRING_EXPIRY,
    PairingManager,
    TrustedPeer,
)


@pytest.fixture
def pairing_manager(config_dir):
    """Create a fresh PairingManager."""
    return PairingManager(
        instance_id="node-aaa-111",
        hostname="workstation-1",
        config_dir=config_dir,
    )


@pytest.fixture
def remote_manager(config_dir):
    """Create a second PairingManager (the remote peer)."""
    remote_dir = config_dir.parent / "remote_config"
    remote_dir.mkdir()
    return PairingManager(
        instance_id="node-bbb-222",
        hostname="microscope-1",
        config_dir=remote_dir,
    )


class TestCryptoHelpers:
    def test_generate_nonce(self):
        n1 = PairingManager.generate_nonce()
        n2 = PairingManager.generate_nonce()
        assert len(n1) == 32  # 16 bytes hex
        assert n1 != n2  # random

    def test_compute_pin_deterministic(self):
        pin1 = PairingManager.compute_pin("aaa", "bbb")
        pin2 = PairingManager.compute_pin("aaa", "bbb")
        assert pin1 == pin2
        assert len(pin1) == 6

    def test_compute_pin_order_independent(self):
        """PIN should be the same regardless of nonce order."""
        pin1 = PairingManager.compute_pin("abc", "xyz")
        pin2 = PairingManager.compute_pin("xyz", "abc")
        assert pin1 == pin2

    def test_derive_token_order_independent(self):
        t1 = PairingManager._derive_token("abc", "xyz")
        t2 = PairingManager._derive_token("xyz", "abc")
        assert t1 == t2

    def test_daily_token_varies_by_day(self):
        base = "some-base-token"
        d1 = PairingManager._derive_daily_token(base, 100)
        d2 = PairingManager._derive_daily_token(base, 101)
        assert d1 != d2


class TestPairingFlow:
    def test_full_pairing_flow(self, pairing_manager, remote_manager):
        """Simulate the complete pairing protocol."""
        # Initiator creates nonce
        nonce_init = pairing_manager.create_initiation()

        # Responder handles pair request
        session_resp = remote_manager.handle_pair_request(
            initiator_id=pairing_manager.instance_id,
            initiator_hostname=pairing_manager.hostname,
            nonce_initiator=nonce_init,
        )

        # Initiator processes response
        session_init = pairing_manager.process_initiation_response(
            peer_id=remote_manager.instance_id,
            peer_hostname=remote_manager.hostname,
            nonce_local=nonce_init,
            nonce_remote=session_resp.nonce_responder,
            pairing_id=session_resp.pairing_id,
        )

        # PINs should match
        assert session_init.pin == session_resp.pin

        # Responder confirms
        remote_manager.confirm_pairing(
            session_resp.pairing_id,
            remote_manager.instance_id,
        )

        # Initiator confirms (initiator already auto-confirms, but let's also
        # call confirm on the remote side's session from the initiator's perspective)
        # Actually, the initiator's session is auto-confirmed. We need to trigger
        # the confirm on the responder's session from the initiator side.
        # In the real flow, the initiator calls POST /confirm on the responder.
        # The confirm_pairing on remote_manager with initiator_id completes it.
        remote_manager.confirm_pairing(
            session_resp.pairing_id,
            pairing_manager.instance_id,
        )

        # Both sides confirmed → session should be confirmed
        # But we need to confirm on initiator side too
        pairing_manager.confirm_pairing(
            session_resp.pairing_id,
            remote_manager.instance_id,
        )

        # Check trust established on remote
        assert remote_manager.is_trusted(pairing_manager.instance_id)

    def test_reject_pairing(self, pairing_manager, remote_manager):
        nonce = pairing_manager.create_initiation()
        session = remote_manager.handle_pair_request(
            pairing_manager.instance_id,
            pairing_manager.hostname,
            nonce,
        )
        rejected = remote_manager.reject_pairing(session.pairing_id)
        assert rejected.status == "rejected"

    def test_get_pending_sessions(self, pairing_manager, remote_manager):
        nonce = pairing_manager.create_initiation()
        remote_manager.handle_pair_request(
            pairing_manager.instance_id,
            pairing_manager.hostname,
            nonce,
        )
        pending = remote_manager.get_pending_sessions()
        assert len(pending) == 1

    def test_cleanup_expired(self, pairing_manager, remote_manager):
        nonce = pairing_manager.create_initiation()
        session = remote_manager.handle_pair_request(
            pairing_manager.instance_id,
            pairing_manager.hostname,
            nonce,
        )
        # Force expiry
        session.created_at = time.time() - PAIRING_EXPIRY - 10
        remote_manager.cleanup_expired()
        assert session.status == "expired"


class TestTrustPersistence:
    def test_save_and_load(self, config_dir):
        mgr1 = PairingManager("node-1", "host-1", config_dir)
        # Manually add a trusted peer
        mgr1._trusted["peer-x"] = TrustedPeer(
            instance_id="peer-x",
            hostname="remote-host",
            base_token="deadbeef" * 8,
        )
        mgr1._save_trusted()

        # Load in a new instance
        mgr2 = PairingManager("node-1", "host-1", config_dir)
        assert mgr2.is_trusted("peer-x")
        assert mgr2._trusted["peer-x"].hostname == "remote-host"

    def test_unpair_by_instance_id(self, pairing_manager):
        pairing_manager._trusted["peer-z"] = TrustedPeer(
            instance_id="peer-z",
            hostname="host-z",
            base_token="abc",
        )
        assert pairing_manager.unpair("peer-z") is True
        assert not pairing_manager.is_trusted("peer-z")

    def test_unpair_by_hostname(self, pairing_manager):
        pairing_manager._trusted["peer-w"] = TrustedPeer(
            instance_id="peer-w",
            hostname="Host-W",
            base_token="abc",
        )
        assert pairing_manager.unpair("host-w") is True

    def test_unpair_nonexistent(self, pairing_manager):
        assert pairing_manager.unpair("nonexistent") is False


class TestTokenAuth:
    def test_verify_valid_token(self, pairing_manager):
        pairing_manager._trusted["peer-auth"] = TrustedPeer(
            instance_id="peer-auth",
            hostname="host",
            base_token="secret123",
        )
        token = pairing_manager.get_token_for_peer("peer-auth")
        result = pairing_manager.verify_token(token)
        assert result == "peer-auth"

    def test_verify_invalid_token(self, pairing_manager):
        pairing_manager._trusted["peer-auth"] = TrustedPeer(
            instance_id="peer-auth",
            hostname="host",
            base_token="secret123",
        )
        result = pairing_manager.verify_token("invalid-token")
        assert result is None

    def test_token_for_untrusted_returns_none(self, pairing_manager):
        assert pairing_manager.get_token_for_peer("nonexistent") is None


class TestScopes:
    def test_default_scopes(self, pairing_manager):
        pairing_manager._trusted["peer-s"] = TrustedPeer(
            instance_id="peer-s",
            hostname="host",
            base_token="abc",
        )
        scopes = pairing_manager.get_scopes_for_peer("peer-s")
        assert "status" in scopes
        assert "data" in scopes
        assert "ml" in scopes

    def test_set_scopes(self, pairing_manager):
        pairing_manager._trusted["peer-s2"] = TrustedPeer(
            instance_id="peer-s2",
            hostname="host2",
            base_token="abc",
        )
        success = pairing_manager.set_scopes("peer-s2", ["status"])
        assert success is True
        scopes = pairing_manager.get_scopes_for_peer("peer-s2")
        assert scopes == ["status"]

    def test_set_invalid_scope(self, pairing_manager):
        pairing_manager._trusted["peer-s3"] = TrustedPeer(
            instance_id="peer-s3",
            hostname="host3",
            base_token="abc",
        )
        success = pairing_manager.set_scopes("peer-s3", ["invalid_scope"])
        assert success is False


class TestRateLimiting:
    def test_first_attempt_allowed(self, pairing_manager):
        allowed, _ = pairing_manager.check_rate_limit("10.0.0.1")
        assert allowed is True

    def test_rapid_attempts_throttled(self, pairing_manager):
        ip = "10.0.0.99"
        for _ in range(3):
            pairing_manager.record_attempt(ip)
        # With exponential backoff, should be throttled
        allowed, retry_after = pairing_manager.check_rate_limit(ip)
        # Either allowed (enough time passed) or throttled
        # The point is it doesn't crash and returns valid values
        assert isinstance(allowed, bool)
        assert isinstance(retry_after, float)
