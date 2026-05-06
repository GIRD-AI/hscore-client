"""
Machine fingerprinting for HScore license binding.

Computes a stable, hardware-specific SHA-256 fingerprint from:
  - /etc/machine-id  (Linux — most stable)
  - hostname
  - First non-loopback MAC address

The fingerprint is used as a KDF input to derive the AES key that
decrypts the probe weights. It never leaves the machine.
"""

from __future__ import annotations

import hashlib
import re
import socket
import subprocess
from pathlib import Path


def get_machine_fingerprint() -> str:
    """Return a 64-char hex SHA-256 fingerprint of this machine."""
    parts: list[str] = []

    # 1. Linux machine-id — most stable across reboots
    for path in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
        try:
            mid = Path(path).read_text().strip()
            if mid:
                parts.append(f"machine-id:{mid}")
                break
        except OSError:
            pass

    # 2. Hostname
    parts.append(f"hostname:{socket.gethostname()}")

    # 3. First non-loopback MAC address
    mac = _get_primary_mac()
    if mac:
        parts.append(f"mac:{mac}")

    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def get_machine_id() -> str:
    """Human-readable machine identifier (hostname) for display in admin dashboard."""
    return socket.gethostname()


def _get_primary_mac() -> str | None:
    """Extract the first non-loopback MAC from `ip link` (Linux) or `ifconfig` (macOS)."""
    try:
        out = subprocess.check_output(["ip", "link", "show"], text=True, stderr=subprocess.DEVNULL)
        macs = re.findall(r"link/ether\s+([0-9a-f:]{17})", out)
        if macs:
            return macs[0]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    try:
        out = subprocess.check_output(["ifconfig"], text=True, stderr=subprocess.DEVNULL)
        macs = re.findall(r"ether\s+([0-9a-f:]{17})", out)
        if macs:
            return macs[0]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return None
