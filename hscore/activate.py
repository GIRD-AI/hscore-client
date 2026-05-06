"""
HScore license activation CLI.

Usage:
  python -m hscore.activate --token <TOKEN>
  python -m hscore.activate --renew
  python -m hscore.activate --status
  python -m hscore.activate --fingerprint
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from ._fingerprint import get_machine_fingerprint, get_machine_id
from .license import LICENSE_PATH, SERVER_URL, _save_license, _load_token


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m hscore.activate",
        description="Activate or manage your HScore license",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--token",       metavar="TOKEN", help="Activate with a new license token")
    group.add_argument("--renew",       action="store_true", help="Renew the current license")
    group.add_argument("--status",      action="store_true", help="Show current license status")
    group.add_argument("--fingerprint", action="store_true", help="Print this machine's fingerprint (for support)")
    args = parser.parse_args()

    if args.fingerprint:
        print(f"Machine ID:          {get_machine_id()}")
        print(f"Machine fingerprint: {get_machine_fingerprint()}")
        return

    if args.status:
        _cmd_status()
    elif args.token:
        _cmd_activate(args.token)
    elif args.renew:
        _cmd_renew()


def _cmd_activate(token: str) -> None:
    fingerprint = get_machine_fingerprint()
    machine_id  = get_machine_id()

    print(f"Activating HScore license on: {machine_id}")
    print(f"Contacting {SERVER_URL} …")

    data = _post(f"{SERVER_URL}/api/license/activate", {
        "token":               token,
        "machine_id":          machine_id,
        "machine_fingerprint": fingerprint,
    })

    _save_license(data)

    token_path = Path.home() / ".hscore" / "token"
    token_path.write_text(token)

    print()
    print("✓ License activated successfully!")
    print(f"  Models:      {', '.join(data.get('allowed_models', []))}")
    print(f"  Expires:     {data.get('expires_at', '—')}")
    print(f"  Grace until: {data.get('grace_until', '—')}")
    print()
    print(f"License saved to {LICENSE_PATH}")

    # Download probe weights for all licensed models
    downloads: dict = data.get("model_downloads", {})
    if downloads:
        from .model import CACHE_DIR, _download_weights
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print()
        print("Downloading probe weights…")
        for mid, url in downloads.items():
            dest = CACHE_DIR / f"{mid}.enc"
            try:
                _download_weights(mid, url, dest)
                size_mb = dest.stat().st_size / 1_048_576
                print(f"  ✓ {mid}  [{size_mb:.1f} MB]")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ {mid}: download failed ({exc})")
                print("    You can retry by re-running: python -m hscore.activate --token <TOKEN>")


def _cmd_renew() -> None:
    token       = _load_token()
    fingerprint = get_machine_fingerprint()
    machine_id  = get_machine_id()

    print(f"Renewing HScore license for: {machine_id}")
    print(f"Contacting {SERVER_URL} …")

    data = _post(f"{SERVER_URL}/api/license/renew", {
        "token":               token,
        "machine_id":          machine_id,
        "machine_fingerprint": fingerprint,
    })

    _save_license(data)
    print()
    print("✓ License renewed!")
    print(f"  New expiry:  {data.get('expires_at', '—')}")
    print(f"  Grace until: {data.get('grace_until', '—')}")


def _cmd_status() -> None:
    if not LICENSE_PATH.exists():
        print("✗ No license found.")
        print(f"  Run: python -m hscore.activate --token <TOKEN>")
        sys.exit(1)

    try:
        from datetime import datetime, timezone
        data        = json.loads(LICENSE_PATH.read_text())
        expires_at  = datetime.fromisoformat(data.get("expires_at", "").replace("Z", "+00:00"))
        grace_until = datetime.fromisoformat(data.get("grace_until", "").replace("Z", "+00:00"))
        now         = datetime.now(timezone.utc)
        days_left   = (expires_at - now).days
        grace_left  = (grace_until - now).days

        print("HScore License Status")
        print(f"  Machine:     {data.get('machine_id', '—')}")
        print(f"  Models:      {', '.join(data.get('allowed_models', []))}")
        print(f"  Expires:     {expires_at.date()} ({days_left} days)")
        if days_left < 0:
            print(f"  Grace ends:  {grace_until.date()} ({grace_left} days remaining)")
        print(f"  Signature:   {'✓ valid' if data.get('signature') else '✗ missing'}")
    except Exception as e:
        print(f"✗ Could not read license: {e}")
        sys.exit(1)


def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read()).get("error", str(e))
        print(f"\n✗ Activation failed: {err}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\n✗ Could not reach license server: {e.reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
