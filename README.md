# HScore Client

**HScore** detects hallucination and deception in large language models by reading internal model friction — no ground truth required.

This repository contains the **customer-facing client package** for activating and running HScore on-premise.

---

## Requirements

- Python ≥ 3.10
- A valid HScore license token (issued by [GIRD AI](https://gird.ai))

---

## Installation

```bash
pip install git+https://github.com/GIRD-AI/hscore-client.git
```

Or clone and install locally:

```bash
git clone https://github.com/GIRD-AI/hscore-client.git
cd hscore-client
pip install .
```

---

## Activation

You need a license token from GIRD AI. Once you have one:

```bash
python -m hscore.activate --token <YOUR_TOKEN>
```

This will:
1. Bind the license to this machine
2. Download and save your signed license to `~/.hscore/license.json`

To check your license status:

```bash
python -m hscore.activate --status
```

To renew (within 7 days of expiry):

```bash
python -m hscore.activate --renew
```

---

## Offline operation

HScore supports a **90-day offline grace period**. Once activated, it will continue to run without network access for up to 90 days after the license expiry date.

---

## License transfer

Licenses are bound to a single machine. To transfer to a new machine, contact [support@gird.ai](mailto:support@gird.ai) with your machine fingerprint:

```bash
python -m hscore.activate --fingerprint
```

---

## Support

- Email: [support@gird.ai](mailto:support@gird.ai)
- Web: [gird.ai](https://gird.ai)

---

*© GIRD AI. All rights reserved. Proprietary and confidential.*
