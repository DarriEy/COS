# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Abstract base class for all COS observation connectors.

Unlike CSFS (one gauge → discharge), a COS connector serves exactly one
:class:`~cos.core.models.ObservationKind` and is one of three structural
classes — ``gridded`` (reduce a raster), ``point_network`` (select stations), or
``flux_tower`` (towers / model ensembles). The two entry points are:

* :meth:`list_sites` — the sites this connector would serve for a
  :class:`~cos.core.models.ReductionSpec` (stations for point networks; the
  reduced region(s) for gridded products);
* :meth:`fetch_series` — fetch + canonicalize + (for gridded) reduce, returning
  one or more :class:`~cos.core.models.ObservationSeries` already in the kind's
  canonical SI unit and UTC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cos.core.exceptions import ConnectorError, RateLimitError
from cos.core.models import ObservationKind, ObservationSeries, ReductionSpec, SiteRef

logger = structlog.get_logger()


class BaseObservationConnector(ABC):
    """Interface every observation connector implements."""

    slug: str                         # e.g. "grace", "snotel", "openet"
    display_name: str                 # e.g. "NASA GRACE/GRACE-FO"
    kind: ObservationKind             # the single kind this connector serves
    structural_class: str             # "gridded" | "point_network" | "flux_tower"
    base_url: str
    #: auth-provider ids this connector needs; empty frozenset = anonymous.
    auth: frozenset[str] = frozenset()

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> BaseObservationConnector:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(120.0, connect=15.0),
            headers={"User-Agent": "COS/0.1 (community-observation-service)"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise ConnectorError(self.slug, "Connector used outside async context manager")
        return self._client

    @abstractmethod
    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Return the sites this connector would serve for *spec*."""

    @abstractmethod
    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        """Fetch + canonicalize + (gridded) reduce to canonical series.

        Returns one series for a reduced region, or one per selected station.
        Values are in the kind's canonical SI unit; timestamps are UTC.
        """

    _RETRYABLE = (RateLimitError, httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout)

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
    )
    async def _get(
        self, path: str, params: dict | None = None, headers: dict | None = None,
    ) -> httpx.Response:
        """HTTP GET with retry on rate limits / connection errors."""
        resp = await self.client.get(path, params=params, headers=headers)
        if resp.status_code == 429:
            raise RateLimitError(self.slug, "Rate limited")
        if resp.status_code not in (200, 206):
            resp.raise_for_status()
        return resp
