# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""COS — Community Observation Service.

The community service for the **non-streamflow** hydrological observation kinds
(TWS, SWE, snow cover, ET, soil moisture, groundwater, LAI, LST, precip-as-obs,
surface water, water level). Streamflow belongs to CSFS; COS never duplicates it.

This module is the blessed public Python API; deeper imports
(``cos.connectors.*``) are internal. Connector modules are not imported here —
call :func:`discover` (or :func:`fetch_series`, which does it for you) to
populate the registry.

Typical usage::

    import cos
    from cos import ObservationKind, ReductionSpec

    spec = ReductionSpec(domain_name="bow", station_ids=("snotel:679",), options={"state": "WA"})
    series = cos.fetch_series_sync("snotel", spec, start, end)
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from cos.core.config import load_config, resolve_credentials
from cos.core.models import (
    KIND_UNITS,
    OBSERVATION_SCHEMA,
    ObservationKind,
    ObservationPoint,
    ObservationSeries,
    QualityFlag,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import discover, get_connector, list_providers

__version__ = "0.1.0"

__all__ = [
    "KIND_UNITS",
    "OBSERVATION_SCHEMA",
    "ObservationKind",
    "ObservationPoint",
    "ObservationSeries",
    "QualityFlag",
    "ReductionSpec",
    "SiteRef",
    "SpatialReduction",
    "__version__",
    "discover",
    "fetch_series",
    "fetch_series_sync",
    "get_connector",
    "list_providers",
    "load_config",
    "resolve_credentials",
]


async def fetch_series(
    provider_slug: str,
    spec: ReductionSpec,
    start: datetime,
    end: datetime,
    config: dict | None = None,
) -> list[ObservationSeries]:
    """Fetch canonical observation series from one connector (no store needed)."""
    discover()
    connector_cls = get_connector(provider_slug)
    async with connector_cls(config=config or {}) as connector:
        return await connector.fetch_series(spec, start, end)


def fetch_series_sync(
    provider_slug: str,
    spec: ReductionSpec,
    start: datetime,
    end: datetime,
    config: dict | None = None,
) -> list[ObservationSeries]:
    """Synchronous wrapper around :func:`fetch_series`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fetch_series(provider_slug, spec, start, end, config))
    raise RuntimeError(
        "fetch_series_sync() cannot be called while an event loop is running; "
        "use 'await cos.fetch_series(...)' instead."
    )
