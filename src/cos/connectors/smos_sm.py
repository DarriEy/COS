# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""SMOS surface soil-moisture connector (gridded, basin-reduced).

Mirrors the SYMFLUENCE native ``smos`` / ``smos_sm`` observation handler
(``data/observation/handlers/soil_moisture.py::SMOSSMHandler``). SMOS is an ESA
L-band passive-microwave product at ~25 km, served as NetCDF (acquired natively
via Copernicus CDS ``satellite-soil-moisture``, passive-only sensor type). Each
file is a global lat/lon grid of *volumetric* surface soil moisture with a time
axis.

This connector:

1. opens a SMOS NetCDF (a local cached file supplied via config ``nc_path`` /
   ``path`` — live Earthdata/CDS download is wired per-connector only where
   trivial; the reduction path is the architecture-critical, hermetically-tested
   part);
2. extracts ``lat / lon / time`` and the soil-moisture variable (one of ``sm``,
   ``Soil_Moisture``, ``soil_moisture``, ``volumetric_surface_soil_moisture``,
   ``SM``) as numpy arrays;
3. masks each cell to the physical range ``0 < sm < 1`` (non-physical / fill
   values -> NaN -> ``QualityFlag.MISSING``), exactly as the native handler does;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` over the
   bbox for larger basins, ``nearest_cell`` at the centroid for small ones.

Units: the source is already volumetric m3/m3, which is the canonical
``soil_moisture`` unit (``KIND_UNITS[SOIL_MOISTURE] == "m3/m3"``); the boundary
conversion is therefore the identity, matching the native handler (it emits the
masked volumetric fraction unchanged).
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

#: candidate soil-moisture variable names, in the native handler's order.
SM_VARIABLES = (
    "sm",
    "Soil_Moisture",
    "soil_moisture",
    "volumetric_surface_soil_moisture",
    "SM",
)
#: <= this area (km²) defaults to nearest-cell sampling (coarse ~25 km product;
#: a single cell already covers a small basin). Mirrors the GRACE size policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("smos_sm")
class SMOSSMConnector(BaseObservationConnector):
    slug = "smos_sm"
    display_name = "ESA SMOS Surface Soil Moisture"
    kind = ObservationKind.SOIL_MOISTURE
    structural_class = "gridded"
    base_url = "https://cds.climate.copernicus.eu"
    auth = frozenset({"cds"})

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
        nc_path = self.config.get("nc_path") or self.config.get("path")
        if not nc_path:
            raise ConnectorError(
                self.slug,
                "SMOS live fetch needs a NetCDF path (config 'nc_path'/'path') or a CDS "
                "download (not yet wired). The reduction path is the proven part; supply "
                "a downloaded SMOS satellite-soil-moisture NetCDF to reduce it.",
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
        """Open a SMOS NetCDF, mask + reduce to the basin, canonicalize to m3/m3."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            sm_var = next((v for v in SM_VARIABLES if v in ds), None)
            if sm_var is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a soil-moisture variable (looked for {SM_VARIABLES})",
                )
            lat_name = "lat" if "lat" in ds else "latitude"
            lon_name = "lon" if "lon" in ds else "longitude"
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(ds[sm_var].values, dtype="float64")  # (time, lat, lon)

        # Native masking: keep only physical volumetric SM in (0, 1); everything
        # else (fill, NaN, saturation overflow) becomes NaN -> QualityFlag.MISSING.
        values = np.where((values > 0.0) & (values < 1.0), values, np.nan)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("SMOS basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("SMOS nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # already m3/m3 — identity boundary conversion
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)).
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "ESA SMOS (passive L-band)",
                "url": "https://cds.climate.copernicus.eu/datasets/satellite-soil-moisture",
                "variable": sm_var,
            },
            fetched_at=datetime.now(UTC),
        )

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"smos_sm:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"smos_sm:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"SMOS soil moisture over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
