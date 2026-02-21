"""
TLS certificate utilities for mesh peer communication.

Generates self-signed EC certificates via the openssl CLI (no Python
cryptography dependency). Provides SSL context builders for server and
client use.
"""

import hashlib
import logging
import shutil
import ssl
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CERT_FILENAME = "mesh_cert.pem"
KEY_FILENAME = "mesh_key.pem"
CERT_DAYS = 3650  # ~10 years


def ensure_tls_cert(config_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Ensure a TLS cert/key pair exists in config_dir.

    Generates a self-signed EC (prime256v1) certificate if one doesn't
    already exist. Uses the ``openssl`` CLI.

    Returns
    -------
    (cert_path, key_path) on success, (None, None) on failure.
    """
    cert_path = config_dir / CERT_FILENAME
    key_path = config_dir / KEY_FILENAME

    if cert_path.exists() and key_path.exists():
        logger.info(f"TLS cert already exists: {cert_path}")
        return cert_path, key_path

    openssl = shutil.which("openssl")
    if openssl is None:
        logger.warning("openssl not found in PATH — TLS disabled")
        return None, None

    config_dir.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                openssl, "req",
                "-x509",
                "-newkey", "ec",
                "-pkeyopt", "ec_paramgen_curve:prime256v1",
                "-keyout", str(key_path),
                "-out", str(cert_path),
                "-days", str(CERT_DAYS),
                "-nodes",
                "-subj", "/CN=gently-mesh",
                "-addext", "subjectAltName=IP:0.0.0.0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        logger.info(f"Generated TLS cert: {cert_path}")
        return cert_path, key_path
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Failed to generate TLS cert: {e}")
        # Clean up partial files
        for p in (cert_path, key_path):
            if p.exists():
                p.unlink()
        return None, None


def get_cert_fingerprint(cert_path: Path) -> str:
    """
    Compute SHA-256 fingerprint of a PEM certificate.

    Reads the cert, extracts DER bytes, and returns the hex digest.
    Uses Python's ssl module — no openssl CLI needed.
    """
    try:
        der_bytes = ssl.PEM_cert_to_DER_cert(cert_path.read_text())
        return hashlib.sha256(der_bytes).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute cert fingerprint: {e}")
        return ""


def build_server_ssl_context(
    cert_path: Path, key_path: Path,
) -> ssl.SSLContext:
    """Build an SSL context for the uvicorn/FastAPI server."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_path), str(key_path))
    return ctx


def build_client_ssl_context() -> ssl.SSLContext:
    """
    Build an SSL context for outgoing peer requests.

    Disables hostname and CA verification — we rely on certificate
    fingerprint pinning instead of the CA trust chain.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx
