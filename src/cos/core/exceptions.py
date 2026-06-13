# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""COS exception hierarchy."""


class COSError(Exception):
    """Base exception for all COS errors."""


class ConnectorError(COSError):
    """Raised when an observation connector fails."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class AuthRequiredError(ConnectorError):
    """Raised when a connector needs credentials that were not supplied."""


class RateLimitError(ConnectorError):
    """Raised when a provider rate-limits us."""


class DataFormatError(ConnectorError):
    """Raised when a provider response doesn't match the expected format."""


class ReductionError(COSError):
    """Raised when a gridded product cannot be reduced to the requested geometry."""


class DependencyError(ConnectorError):
    """Raised when a connector's required (optional) dependencies are missing."""
