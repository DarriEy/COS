# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""UCSB CHIRPS precipitation connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
satellite + station-blended rainfall product. CHIRPS (Climate Hazards Group
InfraRed Precipitation with Station data) is a quasi-global ~0.05deg (~5 km)
daily rainfall estimate served as NetCDF by the UC Santa Barbara Climate Hazards
Group with no authentication. This connector:

1. opens a CHIRPS NetCDF (a local cached file, or a downloaded one);
2. extracts ``lat / lon / time`` and the precipitation variable
   (``precip`` / ``precipitation`` / ``pr`` / ``prcp`` / ``ppt``) as numpy arrays,
   mirroring the native SYMFLUENCE handler's variable search;
3. masks the native fill value (``-9999``) and any negative depth to NaN exactly
   as the native handler does (``precip < 0 -> NaN``), so fill/missing cells
   reduce to MISSING;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the size policy made explicit and
   configurable here);
5. emits the canonical ``precipitation`` unit ``mm``. CHIRPS daily values are
   already a per-timestep depth in mm (mm/day == mm over a daily step), so the
   conversion at the boundary is the identity (no scaling), mirroring the native
   handler which carries the values through as ``precipitation_mm`` unchanged.

The fetch path is exercised only against the live UCSB server; the reduce +
canonicalize path is hermetically tested with a synthetic in-memory NetCDF, so
the architecture-critical reduction logic is covered without network.
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

#: CHIRPS native fill / no-data value.
FILL_VALUE = -9999.0
#: Candidate precipitation variable names, in preference order, mirroring the
#: native SYMFLUENCE CHIRPS handler's ``_find_precip_variable`` search.
PRECIP_VARIABLES = ("precip", "precipitation", "pr", "prcp", "ppt")
#: <= this area (km²) defaults to point sampling (nearest cell).
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("chirps_precip")
class CHIRPSPrecipitationConnector(BaseObservationConnector):
    slug = "chirps_precip"
    display_name = "UCSB CHIRPS Precipitation"
    kind = ObservationKind.PRECIPITATION
    structural_class = "gridded"
    base_url = "https://data.chc.ucsb.edu/products/CHIRPS-2.0"
    auth = frozenset()  # anonymous — UCSB CHG serves CHIRPS without auth

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
                "CHIRPS live fetch needs a NetCDF path (config 'nc_path'/'path') or "
                "a UCSB CHG download (not yet wired). The reduction path is the proven "
                "part; supply a downloaded CHIRPS NetCDF to reduce it.",
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
        """Open a CHIRPS NetCDF, mask fill/negatives, reduce to the basin (mm)."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a CHIRPS precipitation variable (tried {PRECIP_VARIABLES})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        # Mask the native fill value and any negative depth exactly as the native
        # handler does (``precip < 0 -> NaN``): invalid cells become NaN so the
        # reduction skips them and they surface as MISSING.
        invalid = (values == FILL_VALUE) | ~np.isfinite(values) | (values < 0.0)
        values = np.where(invalid, np.nan, values)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("CHIRPS basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("CHIRPS nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # already mm (daily depth) — identity conversion
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim, half-open UTC [start, end).
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
                "source": "UCSB CHIRPS v2.0",
                "url": "https://data.chc.ucsb.edu/products/CHIRPS-2.0",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    def _find_variable(self, ds: object) -> str | None:
        """Pick the precipitation variable, native preference order first."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in PRECIP_VARIABLES:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises precipitation / rain.
        for name in data_vars:
            lower = name.lower()
            if "precip" in lower or "rain" in lower:
                return name
        return None

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"chirps:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"chirps:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"CHIRPS precipitation over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
