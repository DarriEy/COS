# SPDX-License-Identifier: GPL-3.0-or-later
"""Structured connector-validation reporting.

Parity evidence remains beside the SYMFLUENCE capability declarations because
that gate consumes it.  This module turns those declarations into a stable,
machine-readable public report without duplicating the evidence.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def validation_tier(grade: str | None) -> str:
    """Return the conservative validation tier encoded by a capability grade."""
    if not grade:
        return "unvalidated"
    normalized = grade.lower()
    if normalized.startswith("live-spec"):
        return "live-spec"
    if normalized.startswith("live:") or "live parity" in normalized:
        return "live-parity"
    if "parity-by-construction" in normalized or "value-identical" in normalized or "value-within" in normalized:
        return "parity-by-construction"
    return "graded"


def validation_report() -> list[dict[str, Any]]:
    """Return one deterministic validation row per registered connector."""
    from cos.integrations.symfluence import observation_capabilities

    return [
        {
            "provider": cap.provider_id,
            "kind": cap.kind.value,
            "tier": validation_tier(cap.parity_grade),
            "grade": cap.parity_grade,
            "auth": sorted(cap.auth) or ["anonymous"],
            "redistribution": cap.redistribution,
            "data_license": cap.data_license,
        }
        for cap in observation_capabilities()
    ]


def validation_summary(rows: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Count connectors by validation tier."""
    return dict(sorted(Counter(row["tier"] for row in (rows or validation_report())).items()))
