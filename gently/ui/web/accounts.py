"""Self-managed user accounts for the web UI.

A small, dependency-free account store: users live in a YAML file under the
storage directory (NOT the repo), passwords are PBKDF2-hashed, and browser
sessions are stateless HMAC-signed cookies. This is the "self-managed
accounts" backend chosen for the LAN deployment; institute SSO can be layered
on later behind the same ``resolve_role`` surface in ``auth.py``.

Roles
-----
  viewer    -- read-only. Sees everything (today's watching experience).
  operator  -- viewer + may take the microscope control lock and drive.
  admin     -- operator + may manage users.

Layout (under <GENTLY_STORAGE>/auth/)
  users.yaml   -- { users: { <name>: {role, salt, hash, iterations, created_at} } }
  secret.key   -- random key used to sign session cookies (created on first run)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets
import time
from datetime import datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROLES = ("viewer", "operator", "admin")
CONTROL_ROLES = frozenset({"operator", "admin"})
_PBKDF2_ITERATIONS = 200_000
_SESSION_TTL_SECONDS = 7 * 24 * 3600  # 1 week


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


class AccountStore:
    """File-backed user accounts + signed session tokens."""

    def __init__(self, auth_dir: Path):
        self.auth_dir = Path(auth_dir)
        self.auth_dir.mkdir(parents=True, exist_ok=True)
        self.users_path = self.auth_dir / "users.yaml"
        self.secret_path = self.auth_dir / "secret.key"
        self._users: dict = self._load_users()
        self._secret: bytes = self._load_or_create_secret()

    # ── Persistence ───────────────────────────────────────────
    def _load_users(self) -> dict:
        if not self.users_path.exists():
            return {}
        try:
            data = yaml.safe_load(self.users_path.read_text(encoding="utf-8")) or {}
            return data.get("users", {}) or {}
        except Exception as e:
            logger.error("Failed to read users.yaml: %s", e)
            return {}

    def _save_users(self) -> None:
        tmp = self.users_path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump({"users": self._users}, sort_keys=True), encoding="utf-8")
        tmp.replace(self.users_path)  # atomic

    def _load_or_create_secret(self) -> bytes:
        if self.secret_path.exists():
            return self.secret_path.read_bytes()
        secret = secrets.token_bytes(32)
        self.secret_path.write_bytes(secret)
        try:
            self.secret_path.chmod(0o600)
        except OSError:
            pass  # best-effort on Windows
        return secret

    # ── Users ─────────────────────────────────────────────────
    def has_users(self) -> bool:
        return bool(self._users)

    def list_users(self) -> list:
        return [
            {"username": u, "role": r.get("role", "viewer")} for u, r in sorted(self._users.items())
        ]

    def get_role(self, username: str) -> str | None:
        rec = self._users.get(username)
        return rec.get("role") if rec else None

    def _hash(self, password: str, salt: bytes, iterations: int) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)

    def create_user(self, username: str, password: str, role: str = "viewer") -> None:
        username = (username or "").strip()
        if not username:
            raise ValueError("username required")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        salt = secrets.token_bytes(16)
        self._users[username] = {
            "role": role,
            "salt": salt.hex(),
            "hash": self._hash(password, salt, _PBKDF2_ITERATIONS).hex(),
            "iterations": _PBKDF2_ITERATIONS,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_users()

    def verify_password(self, username: str, password: str) -> str | None:
        """Return the user's role if the password matches, else None."""
        rec = self._users.get((username or "").strip())
        if not rec:
            return None
        try:
            salt = bytes.fromhex(rec["salt"])
            expected = bytes.fromhex(rec["hash"])
            iterations = int(rec.get("iterations", _PBKDF2_ITERATIONS))
        except (KeyError, ValueError):
            return None
        candidate = self._hash(password, salt, iterations)
        if hmac.compare_digest(candidate, expected):
            return rec.get("role", "viewer")
        return None

    def bootstrap_admin_if_empty(self) -> tuple[str, str] | None:
        """If no users exist, create an admin with a random password.

        Returns (username, password) so the launcher can print it once, or
        None if users already exist.
        """
        if self._users:
            return None
        password = secrets.token_urlsafe(12)
        self.create_user("admin", password, role="admin")
        logger.info("Bootstrapped default admin account")
        return ("admin", password)

    # ── Sessions (stateless signed cookie) ────────────────────
    def issue_session(self, username: str, ttl: int = _SESSION_TTL_SECONDS) -> str:
        expiry = int(time.time()) + ttl
        payload = f"{username}|{expiry}".encode()
        sig = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return f"{_b64(payload)}.{_b64(sig)}"

    def verify_session(self, token: str) -> str | None:
        """Return the username for a valid, unexpired token, else None."""
        if not token or "." not in token:
            return None
        try:
            payload_b64, sig_b64 = token.split(".", 1)
            payload = _unb64(payload_b64)
            sig = _unb64(sig_b64)
        except Exception:
            return None
        expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        try:
            username, expiry_s = payload.decode("utf-8").rsplit("|", 1)
            if int(expiry_s) < int(time.time()):
                return None
        except Exception:
            return None
        # The user may have been deleted since the token was issued.
        return username if username in self._users else None


# ── Module-level singleton (set during server init) ───────────
_store: AccountStore | None = None


def set_account_store(store: AccountStore | None) -> None:
    global _store
    _store = store


def get_account_store() -> AccountStore | None:
    return _store
