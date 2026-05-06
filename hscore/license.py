"""
HScore license verification and weights key decryption.

Flow at startup:
  1. Load ~/.hscore/license.json
  2. Verify Ed25519 signature (GIRD's public key is hardcoded here)
  3. Check expiry:
     - Not expired → proceed
     - Expired but within grace period → warn + proceed (no network needed)
     - Expired and past grace period → raise LicenseExpiredError
  4. Try online renewal if expiry < RENEW_THRESHOLD_DAYS away
  5. Decrypt weights_key_enc → return AES key bytes
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidSignature

from ._fingerprint import get_machine_fingerprint, get_machine_id

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# GIRD's Ed25519 public key — hardcoded, tamper-evident
_GIRD_PUBLIC_KEY_PEM = b"""-----BEGIN PUBLIC KEY-----
MCowBQYDK2VwAyEAulhPHG8VPxeQdiLSaO2ze/aIJlgUWyc269VM+WTPvzg=
-----END PUBLIC KEY-----"""

LICENSE_PATH    = Path.home() / ".hscore" / "license.json"
SERVER_URL      = os.environ.get("GIRD_LICENSE_SERVER", "https://gird-saa-s.vercel.app")
GRACE_DAYS      = 90   # days after expiry HScore still runs without network
RENEW_THRESHOLD_DAYS = 7
PBKDF2_ITERATIONS    = 600_000
SALT_LEN             = 32


# ── Exceptions ────────────────────────────────────────────────────────────────

class LicenseError(RuntimeError):
    """Base class for all license errors."""

class LicenseNotFoundError(LicenseError):
    """No license.json found — user needs to activate."""

class LicenseTamperedError(LicenseError):
    """Signature verification failed — file may have been modified."""

class LicenseExpiredError(LicenseError):
    """License expired and grace period has passed."""

class LicenseMachineMismatchError(LicenseError):
    """License was issued for a different machine."""


# ── Public API ────────────────────────────────────────────────────────────────

def check_license(model_id: str | None = None) -> bytes:
    """
    Verify the HScore license and return the 32-byte AES key for decrypting probe weights.

    Args:
        model_id: If provided, also checks that this model is in allowed_models.

    Returns:
        32-byte AES-256 key.

    Raises:
        LicenseNotFoundError, LicenseTamperedError, LicenseExpiredError,
        LicenseMachineMismatchError
    """
    if not LICENSE_PATH.exists():
        raise LicenseNotFoundError(
            f"No HScore license found at {LICENSE_PATH}.\n"
            f"Run: python -m hscore.activate --token <YOUR_TOKEN>"
        )

    data = json.loads(LICENSE_PATH.read_text())
    payload, signature = _split_payload(data)

    _verify_signature(payload, signature)

    fingerprint = get_machine_fingerprint()
    if payload.get("machine_fingerprint") != fingerprint:
        raise LicenseMachineMismatchError(
            "This license is bound to a different machine. "
            "Contact support@gird.ai to transfer your license."
        )

    if model_id and model_id not in payload.get("allowed_models", []):
        raise LicenseError(
            f"Model '{model_id}' is not included in your HScore license.\n"
            f"Licensed models: {payload['allowed_models']}\n"
            f"Contact support@gird.ai to upgrade."
        )

    now = datetime.now(timezone.utc)
    expires_at  = _parse_dt(payload["expires_at"])
    grace_until = _parse_dt(payload["grace_until"])

    if now > grace_until:
        raise LicenseExpiredError(
            f"HScore license expired on {expires_at.date()} and the "
            f"{GRACE_DAYS}-day grace period ended on {grace_until.date()}.\n"
            f"Run: python -m hscore.activate --token <YOUR_TOKEN>  (or contact support@gird.ai)"
        )

    if now > expires_at:
        days_left = (grace_until - now).days
        logger.warning(
            "HScore license expired on %s. Running in grace period (%d days remaining). "
            "Please renew: python -m hscore.activate --renew",
            expires_at.date(), days_left,
        )
    elif (expires_at - now).days < RENEW_THRESHOLD_DAYS:
        _try_renew_license(payload)

    return _decrypt_weights_key(payload["weights_key_enc"], fingerprint)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_payload(data: dict) -> tuple[dict, str]:
    signature = data.pop("signature", None)
    if not signature:
        raise LicenseTamperedError("License file has no signature.")
    return data, signature


def _canonical_json(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _verify_signature(payload: dict, signature_b64url: str) -> None:
    try:
        public_key = serialization.load_pem_public_key(
            _GIRD_PUBLIC_KEY_PEM, backend=default_backend()
        )
        assert isinstance(public_key, Ed25519PublicKey)
        sig_bytes = base64.urlsafe_b64decode(signature_b64url + "==")
        public_key.verify(sig_bytes, _canonical_json(payload))
    except (InvalidSignature, Exception) as e:
        raise LicenseTamperedError(
            "License signature verification failed — the license file may have been tampered with."
        ) from e


def _decrypt_weights_key(encrypted_b64: str, machine_fingerprint: str) -> bytes:
    """
    Decrypt the AES-256 weights key.
    Format: base64(salt[32] + iv[12] + auth_tag[16] + ciphertext[32])
    Key derived via PBKDF2-SHA512(machine_fingerprint, salt, 600_000 iterations).
    """
    blob = base64.b64decode(encrypted_b64)
    salt       = blob[:SALT_LEN]
    iv         = blob[SALT_LEN:SALT_LEN + 12]
    auth_tag   = blob[SALT_LEN + 12:SALT_LEN + 28]
    ciphertext = blob[SALT_LEN + 28:]

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA512(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend(),
    )
    derived_key = kdf.derive(machine_fingerprint.encode())
    aesgcm = AESGCM(derived_key)
    try:
        return aesgcm.decrypt(iv, ciphertext + auth_tag, None)
    except Exception as e:
        raise LicenseTamperedError(
            "Weights key decryption failed — fingerprint mismatch or tampered blob."
        ) from e


def _try_renew_license(payload: dict) -> None:
    try:
        body = json.dumps({
            "token": _load_token(),
            "machine_id": get_machine_id(),
            "machine_fingerprint": get_machine_fingerprint(),
        }).encode()
        req = urllib.request.Request(
            f"{SERVER_URL}/api/license/renew",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            new_data = json.loads(resp.read())
        _save_license(new_data)
        logger.info("HScore license renewed. New expiry: %s", new_data.get("expires_at"))
    except Exception as exc:
        logger.debug("License auto-renewal failed (will retry next run): %s", exc)


def _load_token() -> str:
    token_path = Path.home() / ".hscore" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    raise LicenseError("Activation token not found. Run: python -m hscore.activate --token <TOKEN>")


def _save_license(data: dict) -> None:
    LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_PATH.write_text(json.dumps(data, indent=2))


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
