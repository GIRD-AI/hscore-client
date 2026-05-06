"""
HScore probe weight loader.

Downloads the encrypted probe weights (.enc file) on first use, verifies
the HSCP file header, decrypts in-memory with the AES-256-GCM key derived
from the signed license, and returns a FrictionProbe ready for inference.

Encrypted file format
---------------------
    Offset  Size  Content
    ──────────────────────────────────────────────────────────────────────
    0       4     Magic: b"HSCP"  (HScore Probe)
    4       1     Version: 0x01
    5       12    AES-GCM nonce (IV), random per encryption
    17      n     AES-256-GCM( payload ) + 16-byte GCM auth tag

The auth tag is appended by cryptography's AESGCM — no separate field.
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from tqdm import tqdm

from .license import LICENSE_PATH, LicenseError, check_license

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".hscore" / "models"
MAGIC     = b"HSCP"
VERSION   = b"\x01"

_DOWNLOAD_CHUNK = 8_192  # bytes per streaming chunk


# ── Probe class ────────────────────────────────────────────────────────────────

class FrictionProbe:
    """
    GIRD friction probe — pure numpy, no external ML framework required.

    Scores a hidden-state vector and returns P(hallucinated) ∈ [0, 1].

    Attributes:
        hidden_size  int  — input dimension expected by this probe
        normalize    bool — whether inputs are L2-normalised before scoring
    """

    def __init__(
        self,
        coef: np.ndarray,
        intercept: np.ndarray,
        normalize: bool = False,
    ) -> None:
        self._w        = np.asarray(coef,      dtype=np.float32)
        self._b        = np.asarray(intercept, dtype=np.float32)
        self.normalize = bool(normalize)
        if self._w.ndim == 1:
            self._w = self._w.reshape(1, -1)

    @property
    def hidden_size(self) -> int:
        """Input dimension expected by this probe."""
        return int(self._w.shape[1])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute friction scores as class probabilities.

        Args:
            X: shape (n_samples, hidden_size)

        Returns:
            ndarray shape (n_samples, 2): [[P(honest), P(hallucinated)], ...]
        """
        if self.normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.maximum(norms, 1e-8)
        logit = X @ self._w.T + self._b
        p = 1.0 / (1.0 + np.exp(-logit))
        return np.column_stack([1.0 - p, p])

    def score_single(self, x: np.ndarray) -> float:
        """Return P(hallucinated) for a single hidden-state vector."""
        return float(self.predict_proba(x.reshape(1, -1))[0, 1])

    def __repr__(self) -> str:
        return f"FrictionProbe(hidden_size={self.hidden_size}, normalize={self.normalize})"


# ── Public API ─────────────────────────────────────────────────────────────────

def load_probe(model_id: str) -> FrictionProbe:
    """
    Load the FrictionProbe for *model_id*, downloading and decrypting as needed.

    Steps:
        1. ``check_license(model_id)`` → 32-byte AES key (verifies signature,
           machine binding, and expiry).
        2. Check ``~/.hscore/models/{model_id}.enc`` — stream-download if missing.
        3. Decrypt in-memory (decrypted bytes never written to disk).
        4. Deserialise probe weights from the decrypted payload.

    Returns:
        FrictionProbe ready for inference.

    Raises:
        LicenseError:  If the license is invalid, expired, or model not licensed.
        RuntimeError:  If the download URL is missing from the license.
        ValueError:    If the encrypted file is corrupted or has a wrong magic.
    """
    aes_key = check_license(model_id)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    enc_path = CACHE_DIR / f"{model_id}.enc"

    if not enc_path.exists():
        url = _get_download_url(model_id)
        logger.info("Downloading probe weights for %s …", model_id)
        _download_weights(model_id, url, enc_path)

    raw_bytes = _decrypt_enc_file(enc_path, aes_key)

    arrays    = np.load(BytesIO(raw_bytes))
    normalize = bool(arrays["normalize"]) if "normalize" in arrays else False
    probe     = FrictionProbe(
        coef=arrays["coef"],
        intercept=arrays["intercept"],
        normalize=normalize,
    )
    logger.debug("Loaded probe for %s: %r", model_id, probe)
    return probe


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_download_url(model_id: str) -> str:
    """
    Read ``~/.hscore/license.json`` and return ``model_downloads[model_id]``.

    Raises:
        LicenseError: If the license has no download URL for this model.
    """
    if not LICENSE_PATH.exists():
        raise LicenseError(
            f"No license found at {LICENSE_PATH}. "
            "Run: python -m hscore.activate --token <TOKEN>"
        )

    data = json.loads(LICENSE_PATH.read_text())
    downloads: dict = data.get("model_downloads", {})

    if model_id not in downloads:
        raise LicenseError(
            f"No download URL for model '{model_id}' in your license.\n"
            "Re-activate to refresh: python -m hscore.activate --token <TOKEN>\n"
            "Or contact support@gird.ai."
        )

    return downloads[model_id]


def _download_weights(model_id: str, url: str, dest: Path) -> None:
    """
    Stream-download the encrypted probe file to *dest* with a tqdm progress bar.

    Written atomically via a ``{dest}.tmp`` staging file so a failed download
    never leaves a corrupt cache entry.
    """
    tmp = dest.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            with (
                tmp.open("wb") as f,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=model_id,
                    leave=True,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=_DOWNLOAD_CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _decrypt_enc_file(path: Path, aes_key: bytes) -> bytes:
    """
    Read an HSCP encrypted file, verify the header, and AES-GCM decrypt.

    Args:
        path:    Path to the ``*.enc`` file on disk.
        aes_key: 32-byte AES-256 key returned by ``check_license()``.

    Returns:
        Decrypted raw bytes (numpy .npz archive).

    Raises:
        ValueError: Bad magic, unsupported version, or AES-GCM auth failure.
    """
    data = path.read_bytes()

    if len(data) < 17:
        raise ValueError(f"Encrypted probe file too short ({len(data)} bytes): {path}")

    if data[:4] != MAGIC:
        raise ValueError(
            f"Invalid probe file — expected magic b'HSCP', got {data[:4]!r}. "
            f"File may be corrupt: {path}"
        )

    if data[4:5] != VERSION:
        ver = data[4]
        raise ValueError(
            f"Unsupported probe file version 0x{ver:02x}. "
            "Please upgrade: pip install --upgrade hscore-client"
        )

    iv             = data[5:17]
    ciphertext_tag = data[17:]

    try:
        return AESGCM(aes_key).decrypt(iv, ciphertext_tag, None)
    except Exception as exc:
        raise ValueError(
            "Probe decryption failed — the file may be corrupt or the license key "
            "does not match. Try re-activating: "
            "python -m hscore.activate --token <TOKEN>"
        ) from exc
