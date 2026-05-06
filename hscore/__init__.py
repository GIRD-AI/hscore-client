"""
HScore public API.

Customers import this package:
    import hscore
    hscore.check_license()
    score = hscore.score(text, model="hscore-base")
"""

from .license import (
    check_license,
    LicenseError,
    LicenseNotFoundError,
    LicenseTamperedError,
    LicenseExpiredError,
    LicenseMachineMismatchError,
)

__version__ = "0.1.0"
__all__ = [
    "check_license",
    "LicenseError",
    "LicenseNotFoundError",
    "LicenseTamperedError",
    "LicenseExpiredError",
    "LicenseMachineMismatchError",
]
