# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GRACE / GRACE-FO total water storage connector (gridded, basin-reduced).

Proves the **gridded spatial-reduction path** of the canonical contract. GRACE
mascon products are global monthly liquid-water-equivalent thickness grids
(``lwe_thickness``, cm) served as NetCDF behind NASA Earthdata. This connector:

1. opens a GRACE NetCDF (a local cached file, or a downloaded one — Earthdata
   auth via the resolved credential token);
2. extracts ``lat / lon / time / lwe_thickness`` as numpy arrays;
3. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the size policy the native
   ``grace.py`` uses, made explicit and configurable here);
4. converts cm → **mm** (the canonical ``tws`` unit) and subtracts the anomaly
   baseline mean (default 2003–2008, matching the native handler).

The fetch path is exercised only with Earthdata credentials; the reduce +
canonicalize path is hermetically tested with a synthetic in-memory NetCDF, so
the architecture-critical reduction logic is covered without network or auth.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import ConnectorError, ReductionError
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    ObservationSeries,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

CM_TO_MM = 10.0
DEFAULT_BASELINE = ("2003-01-01", "2008-12-31")
#: <= this area (km²) defaults to point sampling, mirroring native grace.py.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("grace")
class GRACEConnector(BaseObservationConnector):
    slug = "grace"
    display_name = "NASA GRACE/GRACE-FO TWS"
    kind = ObservationKind.TWS
    structural_class = "gridded"
    base_url = "https://archive.podaac.earthdata.nasa.gov"
    auth = frozenset({"earthdata"})

    VARIABLE = "lwe_thickness"

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """One reduced region: the basin (or its centroid cell)."""
        reduction = self._choose_reduction(spec)
        return [self._site_for(spec, reduction)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        nc_path = self.config.get("nc_path")
        if not nc_path:
            raise ConnectorError(
                self.slug,
                "GRACE live fetch needs a NetCDF path (config 'nc_path') or Earthdata "
                "download (not yet wired). The reduction path is the proven part; "
                "supply a downloaded GRACE mascon NetCDF to reduce it.",
            )
        return [self.reduce_file(Path(nc_path), spec, start, end)]

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open a GRACE NetCDF, reduce to the basin, canonicalize to mm anomaly."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            if self.VARIABLE not in ds:
                raise ConnectorError(self.slug, f"NetCDF missing '{self.VARIABLE}' variable")
            da = ds[self.VARIABLE]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("GRACE basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("GRACE nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values * CM_TO_MM,  # cm -> mm at the boundary
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)) then anomaly baseline.
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]
        points = self._apply_baseline(points, spec)

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "GRACE/GRACE-FO",
                "source_doi": "10.5067/TEMSC-3JC62",
                "url": "https://podaac.jpl.nasa.gov/GRACE",
                "baseline": "-".join(spec.options.get("baseline", DEFAULT_BASELINE)),
            },
            fetched_at=datetime.now(UTC),
        )

    def _apply_baseline(self, points: list, spec: ReductionSpec) -> list:
        """Subtract the baseline-window mean to make a TWS anomaly (mm)."""
        b_start, b_end = spec.options.get("baseline", DEFAULT_BASELINE)
        b0 = _utc(datetime.fromisoformat(b_start))
        b1 = _utc(datetime.fromisoformat(b_end))
        vals = [p.value for p in points if p.value is not None and b0 <= p.timestamp <= b1]
        if not vals:
            vals = [p.value for p in points if p.value is not None]
        if not vals:
            return points
        mean = sum(vals) / len(vals)
        for p in points:
            if p.value is not None:
                p.value = p.value - mean
        return points

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"grace:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"grace:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"GRACE TWS over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
