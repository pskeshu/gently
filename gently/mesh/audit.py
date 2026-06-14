"""
Structured security audit log for mesh operations.

Writes JSON-lines to mesh_audit.jsonl in the config directory.
Auto-rotates when the file exceeds MAX_LINES (keeps the tail).
"""

import json
import logging
import time
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_LINES = 10_000
KEEP_LINES = 5_000


class AuditEvent(str, Enum):
    """Enumeration of auditable security events."""

    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    CERT_PIN_OK = "cert_pinning_ok"
    CERT_PIN_FAIL = "cert_pinning_fail"
    SIG_INVALID = "sig_invalid"
    REPLAY_REJECTED = "replay_rejected"
    PAIR_REQUESTED = "pair_requested"
    PAIR_COMPLETED = "pair_completed"
    PAIR_REJECTED = "pair_rejected"
    PEER_UNPAIRED = "peer_unpaired"
    RATE_LIMITED = "rate_limited"
    SCOPE_DENIED = "scope_denied"


class MeshAuditLog:
    """
    Append-only JSON-lines audit log.

    Each entry is a single JSON object on one line:
    {"ts": <unix>, "event": <str>, "peer_id": <str>,
     "ip": <str>, "detail": <str>, "outcome": "allow"|"deny"|"info"}
    """

    def __init__(self, config_dir: Path):
        self._config_dir = config_dir
        self._log_file = config_dir / "mesh_audit.jsonl"
        self._line_count = 0
        self._count_lines()

    def _count_lines(self):
        """Count existing lines for rotation tracking."""
        if self._log_file.exists():
            try:
                with open(self._log_file) as f:
                    self._line_count = sum(1 for _ in f)
            except OSError:
                self._line_count = 0

    def log(
        self,
        event: AuditEvent,
        outcome: str = "info",
        peer_id: str = "",
        ip: str = "",
        detail: str = "",
    ):
        """Append one audit entry."""
        entry = {
            "ts": time.time(),
            "event": event.value,
            "peer_id": peer_id,
            "ip": ip,
            "detail": detail,
            "outcome": outcome,
        }
        try:
            self._config_dir.mkdir(parents=True, exist_ok=True)
            with open(self._log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            self._line_count += 1
            if self._line_count > MAX_LINES:
                self._rotate()
        except OSError as e:
            logger.debug(f"Audit log write failed: {e}")

    def _rotate(self):
        """Keep last KEEP_LINES, discard the rest."""
        try:
            with open(self._log_file) as f:
                lines = f.readlines()
            keep = lines[-KEEP_LINES:]
            with open(self._log_file, "w") as f:
                f.writelines(keep)
            self._line_count = len(keep)
            logger.debug(f"Audit log rotated: kept {len(keep)} lines")
        except OSError as e:
            logger.debug(f"Audit log rotation failed: {e}")
