# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Connector-roster health helpers (mirrors CSFS's roster-padded health).

COS observations are evaluation pulls, not a continuously-mirrored gauge store,
so health here reports the *registered roster* and its declared readiness
(auth/structural-class/status) rather than a live store's freshness. The shape
mirrors CSFS so the CLI and any future API share it.
"""

from __future__ import annotations

from cos.core.registry import discover, get_connector, list_providers


def roster_health() -> list[dict]:
    """One row per registered connector: slug, kind, class, auth, readiness."""
    discover()
    rows: list[dict] = []
    for slug in list_providers():
        cls = get_connector(slug)
        auth = sorted(getattr(cls, "auth", frozenset()))
        kind = getattr(cls, "kind", None)
        rows.append(
            {
                "provider": slug,
                "kind": kind.value if kind is not None else None,
                "structural_class": getattr(cls, "structural_class", None),
                "auth": auth or ["anonymous"],
                "display_name": getattr(cls, "display_name", ""),
            }
        )
    return rows


def summarize_roster(rows: list[dict]) -> dict[str, int]:
    """Count connectors per observation kind."""
    summary: dict[str, int] = {}
    for r in rows:
        key = r.get("kind") or "unknown"
        summary[key] = summary.get(key, 0) + 1
    return summary
