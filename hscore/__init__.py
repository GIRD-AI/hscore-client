"""
HScore public API.

Basic usage (license check only):
    import hscore
    hscore.check_license()

Probe loading (requires scikit-learn + joblib):
    probe = hscore.load_probe("hscore-qwen2.5-7b")

Full inference (requires pip install hscore-client[inference]):
    score = hscore.score("The Eiffel Tower is in Berlin.")
    print(score)  # e.g. 0.91
"""

from .license import (
    check_license,
    LicenseError,
    LicenseNotFoundError,
    LicenseTamperedError,
    LicenseExpiredError,
    LicenseMachineMismatchError,
)
from .model import load_probe
from .scorer import score, batch_score

__version__ = "0.2.0"
__all__ = [
    # Core
    "check_license",
    "load_probe",
    # Inference (requires [inference] extras)
    "score",
    "batch_score",
    # Exceptions
    "LicenseError",
    "LicenseNotFoundError",
    "LicenseTamperedError",
    "LicenseExpiredError",
    "LicenseMachineMismatchError",
]

