# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Connector tiering for COS.

COS observations are *evaluation pulls* (driven by a SYMFLUENCE experiment over
a fixed window), not a continuously-mirrored real-time gauge store like CSFS. So
"tiers" here group connectors by their natural cadence / latency rather than
scheduling a daemon. The table exists for the same reason CSFS's does: the
roster-integrity tests assert every registered connector belongs to exactly one
tier, catching a connector that ships unclassified.

Tiers:
* ``monthly`` — products updated on a monthly cadence (GRACE TWS).
* ``daily``   — daily point networks and daily gridded products (SNOTEL SWE).
* ``ondemand``— keyed / ensemble products pulled per request (OpenET ET).
"""

from __future__ import annotations

#: tier -> connector slugs. Every registered connector must appear in exactly
#: one tier (asserted by tests/test_connector_integrity.py).
PROVIDER_TIERS: dict[str, list[str]] = {
    "monthly": ["grace"],
    "daily": ["snotel"],
    "ondemand": ["openet"],
}

#: tier -> default lookback window (days) for an evaluation pull.
TIER_LOOKBACK_DAYS: dict[str, int] = {
    "monthly": 365 * 20,
    "daily": 365 * 5,
    "ondemand": 365 * 5,
}
