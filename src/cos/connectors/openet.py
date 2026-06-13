# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""OpenET evapotranspiration connector (ensemble / tower-like, key-based).

Proves the **flux-tower / model-ensemble path**. OpenET serves satellite-derived
ET as an *ensemble* of 6 models plus an ensemble mean, via a keyed REST API that
returns a per-geometry timeseries (point or polygon-reduced). This is the
cleaner keyed-access choice over FLUXNET/AmeriFlux for the proof connector
(documented in design §3).

OpenET timeseries are mm per interval (daily/monthly); COS canonicalizes to
**mm/day** (the canonical ``et`` unit) at the connector boundary — for a monthly
interval that means dividing by the days in the interval; for daily it is a
pass-through. The requested model (default ``ensemble``) is recorded in
``site.extra['model']`` so an ensemble pull yields one series per model if asked.

The API call is exercised only with an OpenET key; the parse + mm/day
canonicalization is hermetically tested with a synthetic JSON payload.
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import AuthRequiredError, DataFormatError
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    ObservationPoint,
    ObservationSeries,
    QualityFlag,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()


@register("openet")
class OpenETConnector(BaseObservationConnector):
    slug = "openet"
    display_name = "OpenET ensemble ET"
    kind = ObservationKind.ET
    structural_class = "flux_tower"
    base_url = "https://openet-api.org"
    auth = frozenset({"openet"})

    DEFAULT_MODEL = "ensemble"

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        return [self._site(spec, m) for m in self._models(spec)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        token = self._token()
        out: list[ObservationSeries] = []
        for model in self._models(spec):
            payload = await self._fetch_timeseries(spec, start, end, model, token)
            points = self.parse_timeseries(payload, self._interval(spec))
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=self._site(spec, model),
                    reduction=SpatialReduction.BASIN_MEAN,
                    unit=KIND_UNITS[self.kind],
                    points=points,
                    source_info={
                        "source": "OpenET",
                        "model": model,
                        "url": self.base_url,
                        "interval": self._interval(spec),
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    async def _fetch_timeseries(
        self, spec: ReductionSpec, start: datetime, end: datetime, model: str, token: str,
    ) -> object:
        if spec.geometry is None and spec.bbox is None and spec.centroid is None:
            raise DataFormatError(self.slug, "OpenET needs a geometry, bbox, or centroid")
        body: dict[str, object] = {
            "date_range": [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")],
            "interval": self._interval(spec),
            "model": model,
            "variable": "ET",
            "units": "mm",
            "reducer": "mean",
        }
        if spec.centroid is not None:
            body["geometry"] = [spec.centroid[1], spec.centroid[0]]  # lon, lat
            endpoint = "/raster/timeseries/point"
        else:
            body["geometry"] = self._polygon_coords(spec)
            endpoint = "/raster/timeseries/polygon"
        resp = await self.client.post(
            endpoint, json=body, headers={"Authorization": token},
        )
        if resp.status_code == 429:
            from cos.core.exceptions import RateLimitError

            raise RateLimitError(self.slug, "Rate limited")
        if resp.status_code not in (200, 206):
            resp.raise_for_status()
        return resp.json()

    # -- pure parser (hermetically tested) -----------------------------------

    @staticmethod
    def parse_timeseries(payload: object, interval: str) -> list[ObservationPoint]:
        """Parse OpenET timeseries JSON → canonical ET points (mm → mm/day).

        OpenET returns a list of ``{"time": "YYYY-MM-DD", "et": <mm>}`` rows.
        Monthly mm are divided by the days in the month; daily pass through.
        """
        if not isinstance(payload, list):
            raise DataFormatError("openet", f"Expected a list of timeseries rows, got {type(payload).__name__}")
        points: list[ObservationPoint] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            t_raw = row.get("time") or row.get("date")
            v_raw = row.get("et", row.get("value"))
            if t_raw is None:
                continue
            try:
                ts = datetime.fromisoformat(str(t_raw)[:10]).replace(tzinfo=UTC)
            except ValueError:
                continue
            if v_raw is None:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            try:
                mm = float(v_raw)
            except (TypeError, ValueError):
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            if interval == "monthly":
                days = calendar.monthrange(ts.year, ts.month)[1]
                mm_day = mm / days
            else:
                mm_day = mm
            points.append(ObservationPoint(timestamp=ts, value=mm_day, quality=QualityFlag.GOOD))
        return points

    # -- helpers -------------------------------------------------------------

    def _models(self, spec: ReductionSpec) -> list[str]:
        m = spec.options.get("models") or self.config.get("models")
        if isinstance(m, str):
            return [m]
        if isinstance(m, (list, tuple)) and m:
            return list(m)
        return [self.DEFAULT_MODEL]

    def _interval(self, spec: ReductionSpec) -> str:
        return str(spec.options.get("interval") or self.config.get("interval") or "monthly")

    def _polygon_coords(self, spec: ReductionSpec) -> list[float]:
        if spec.geometry and "coordinates" in spec.geometry:
            ring = spec.geometry["coordinates"][0]
            return [c for pt in ring for c in (pt[0], pt[1])]
        if spec.bbox is not None:
            lat_min, lon_min, lat_max, lon_max = spec.bbox
            return [lon_min, lat_min, lon_max, lat_min, lon_max, lat_max, lon_min, lat_max, lon_min, lat_min]
        raise DataFormatError(self.slug, "No polygon geometry or bbox for OpenET")

    def _token(self) -> str:
        token = self.config.get("token") or self.config.get("api_key")
        if not token:
            raise AuthRequiredError(
                self.slug,
                "OpenET requires an API key. Set config 'token' or the OPENET_API_KEY "
                "env var (resolved by cos.core.config.resolve_credentials).",
            )
        return str(token)

    def _site(self, spec: ReductionSpec, model: str) -> SiteRef:
        return SiteRef(
            kind="reduced_region",
            site_id=f"openet:{spec.domain_name}:{model}",
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"OpenET {model} ET over {spec.domain_name}",
            extra={"model": model},
        )
