# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""ORNL Daymet daily precipitation connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
daily precipitation product. Daymet provides 1 km gridded daily surface-weather
estimates over North America (1980-present) from the ORNL DAAC, with a daily
total-precipitation field ``prcp`` reported in **mm/day** (a daily accumulation).
This connector:

1. opens a Daymet NetCDF (a local cached file, or a downloaded one — ORNL DAAC
   distribution; the gridded download path is not wired here, only the reduction
   is the proven part);
2. extracts ``lat / lon / time`` and the precipitation variable (``prcp`` /
   ``precip`` / ``precipitation``) as numpy arrays;
3. masks Daymet's native missing value (``-9999``) and non-finite cells to NaN so
   they reduce to MISSING, exactly as the native SYMFLUENCE Daymet handler treats
   absent data;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (mirroring the native handler, whose
   gridded path computes a basin spatial mean and whose single-pixel path samples
   the bounding-box centroid cell);
5. emits the canonical ``precipitation`` unit ``mm``. Daymet ``prcp`` is the daily
   precipitation total in mm/day; the canonical kind unit is mm (a per-timestep,
   here daily, total), so the conversion at the boundary is the **identity** (no
   scaling), exactly as the native handler maps ``prcp`` -> ``precip_mm`` /
   ``prcp (mm/day)`` -> ``precip_mm`` without rescaling.

The fetch path is exercised only against a supplied file; the reduce +
canonicalize path is hermetically tested with a synthetic in-memory NetCDF, so
the architecture-critical reduction logic is covered without network or auth.

Daymet is open data hosted by the ORNL DAAC and requires no authentication.
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

#: Daymet native missing value used across the daily surface-weather grids.
FILL_VALUE = -9999.0
#: Candidate precipitation variable names, in preference order, mirroring the
#: native SYMFLUENCE Daymet handler's ``prcp`` mapping.
PRCP_VARIABLES = ("prcp", "precip", "precipitation", "pr")
#: <= this area (km²) defaults to point sampling (nearest cell), matching the
#: native handler which uses a single-pixel centroid request for small areas.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("daymet_precip")
class DaymetPrecipitationConnector(BaseObservationConnector):
    slug = "daymet_precip"
    display_name = "ORNL Daymet Daily Precipitation"
    kind = ObservationKind.PRECIPITATION
    structural_class = "gridded"
    base_url = "https://daymet.ornl.gov"
    auth = frozenset()  # ORNL DAAC open data — no authentication required

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
                "Daymet live fetch needs a NetCDF path (config 'nc_path'/'path') or an "
                "ORNL DAAC download (not yet wired). The reduction path is the proven "
                "part; supply a downloaded Daymet NetCDF to reduce it.",
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
        """Open a Daymet NetCDF, mask fill, reduce to the basin (mm/day -> mm)."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a Daymet precipitation variable (tried {PRCP_VARIABLES})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        # Mask the native missing value and non-finite cells to NaN so the
        # reduction skips them and they surface as MISSING, exactly as the native
        # handler treats absent Daymet data.
        invalid = (values == FILL_VALUE) | ~np.isfinite(values)
        values = np.where(invalid, np.nan, values)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("Daymet basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("Daymet nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # mm/day daily total == canonical mm (identity)
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
                "source": "ORNL DAAC Daymet V4",
                "source_doi": "10.3334/ORNLDAAC/2129",
                "url": "https://daymet.ornl.gov",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    def _find_variable(self, ds: object) -> str | None:
        """Pick the precipitation variable, ``prcp`` preferred (native order)."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in PRCP_VARIABLES:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises precipitation.
        for name in data_vars:
            lower = name.lower()
            if "prcp" in lower or "precip" in lower:
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
            site_id = f"daymet:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"daymet:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"Daymet precipitation over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
