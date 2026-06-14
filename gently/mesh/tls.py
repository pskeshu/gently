"""
TLS certificate utilities for mesh peer communication.

Generates self-signed EC certificates using the ``cryptography`` library
(pure Python, no CLI dependency). Provides SSL context builders for
server and client use.
"""

import datetime
import hashlib
import ipaddress
import logging
import ssl
from pathlib import Path

logger = logging.getLogger(__name__)

CERT_FILENAME = "mesh_cert.pem"
KEY_FILENAME = "mesh_key.pem"
CERT_DAYS = 3650  # ~10 years


def ensure_tls_cert(config_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Ensure a TLS cert/key pair exists in config_dir.

    Generates a self-signed EC (prime256v1) certificate if one doesn't
    already exist. Uses the ``cryptography`` library.

    Returns
    -------
    (cert_path, key_path) on success, (None, None) on failure.
    """
    cert_path = config_dir / CERT_FILENAME
    key_path = config_dir / KEY_FILENAME

    if cert_path.exists() and key_path.exists():
        logger.info(f"TLS cert already exists: {cert_path}")
        return cert_path, key_path

    config_dir.mkdir(parents=True, exist_ok=True)

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        # Generate EC private key (prime256v1 / SECP256R1)
        private_key = ec.generate_private_key(ec.SECP256R1())

        now = datetime.datetime.now(datetime.timezone.utc)
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "gently-mesh"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=CERT_DAYS))
            .add_extension(
                x509.SubjectAlternativeName(
                    [
                        x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
                    ]
                ),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )

        # Write PEM files
        key_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

        logger.info(f"Generated TLS cert: {cert_path}")
        return cert_path, key_path

    except ImportError:
        logger.warning(
            "cryptography package not installed — TLS disabled (pip install cryptography)"
        )
        return None, None
    except Exception as e:
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
    Uses Python's ssl module — no external dependency needed.
    """
    try:
        der_bytes = ssl.PEM_cert_to_DER_cert(cert_path.read_text())
        return hashlib.sha256(der_bytes).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute cert fingerprint: {e}")
        return ""


def build_server_ssl_context(
    cert_path: Path,
    key_path: Path,
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
