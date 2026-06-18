# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NASA GPM IMERG precipitation connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
satellite precipitation product. GPM IMERG (Integrated Multi-satellitE Retrievals
for GPM) daily products are global 0.1° precipitation grids (mm/day) served as
NetCDF behind NASA Earthdata (GES DISC). This connector:

1. opens a GPM IMERG NetCDF (a local cached file, or a downloaded one — Earthdata
   auth via the resolved credential token);
2. extracts ``lat / lon / time`` and the precipitation variable
   (``precipitation`` / ``precipitationCal`` / ``HQprecipitation`` / ...) as numpy
   arrays, mirroring the variable-preference order of the native SYMFLUENCE
   handler;
3. clips negative values to 0 (the native handler clips precipitation to be
   non-negative) so spurious negatives never reach the reduction;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the size policy made explicit and
   configurable here);
5. emits the canonical ``precipitation`` unit ``mm``. GPM IMERG daily
   precipitation is reported as mm/day — i.e. an accumulated depth of mm over the
   day — so per-daily-timestep the source value is already the canonical mm depth
   and the boundary conversion is the identity (no scaling), exactly as the
   native handler passes the mm/day values through unchanged.

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

#: Candidate precipitation variable names, in preference order, mirroring the
#: native SYMFLUENCE GPM IMERG handler's ``_find_precip_variable`` order.
PRECIP_VARIABLES = (
    "precipitation",
    "precipitationCal",
    "precipitationUncal",
    "HQprecipitation",
    "IRprecipitation",
    "precip",
)
#: GPM IMERG missing/fill sentinel (negative; also covers the generic -9999.9).
FILL_VALUE = -9999.9
#: <= this area (km²) defaults to point sampling (nearest cell).
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("gpm_imerg_precip")
class GPMIMERGPrecipConnector(BaseObservationConnector):
    slug = "gpm_imerg_precip"
    display_name = "NASA GPM IMERG Precipitation"
    kind = ObservationKind.PRECIPITATION
    structural_class = "gridded"
    base_url = "https://gpm1.gesdisc.eosdis.nasa.gov"
    auth = frozenset({"earthdata"})

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
                "GPM IMERG live fetch needs a NetCDF path (config 'nc_path'/'path') or "
                "Earthdata download (not yet wired). The reduction path is the proven "
                "part; supply a downloaded GPM IMERG NetCDF to reduce it.",
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
        """Open a GPM IMERG NetCDF, clip negatives, reduce to the basin (mm)."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a GPM IMERG precipitation variable (tried {PRECIP_VARIABLES})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
            values = self._orient_time_lat_lon(da, values, lats.size, lons.size)

        # Mask the fill sentinel (turns into NaN -> MISSING), then clip the
        # surviving negatives to 0 exactly as the native handler does (it clips
        # precipitation to be non-negative before averaging).
        fill_mask = (values <= FILL_VALUE) | ~np.isfinite(values)
        values = np.where(fill_mask, np.nan, values)
        finite = np.isfinite(values)
        values[finite] = np.clip(values[finite], a_min=0.0, a_max=None)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("GPM IMERG basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("GPM IMERG nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # mm/day daily depth == canonical mm; identity
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
                "source": "NASA GPM IMERG (GES DISC)",
                "source_doi": "10.5067/GPM/IMERGDF/DAY/07",
                "url": "https://gpm.nasa.gov/data/imerg",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    def _orient_time_lat_lon(self, da: object, values, n_lat: int, n_lon: int):
        """Return values shaped (time, lat, lon) regardless of source dim order.

        GPM IMERG daily granules are commonly stored (time, lon, lat); the
        reduce kernels require (time, lat, lon). Reorder by dim names when they
        are available, else fall back to a shape-based transpose of the last two
        axes when they are (lon, lat).
        """
        import numpy as np

        dims = list(getattr(da, "dims", ()))
        if len(dims) == 3:
            lower = [str(d).lower() for d in dims]
            time_i = next((i for i, d in enumerate(lower) if "time" in d), 0)
            lat_i = next((i for i, d in enumerate(lower) if d in ("lat", "latitude")), None)
            lon_i = next((i for i, d in enumerate(lower) if d in ("lon", "longitude")), None)
            if lat_i is not None and lon_i is not None:
                return np.transpose(values, (time_i, lat_i, lon_i))
        # Shape-based fallback: (time, lon, lat) -> (time, lat, lon).
        if values.ndim == 3 and values.shape[1] == n_lon and values.shape[2] == n_lat and n_lon != n_lat:
            return np.transpose(values, (0, 2, 1))
        return values

    def _find_variable(self, ds: object) -> str | None:
        """Pick the precipitation variable in native preference order."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in PRECIP_VARIABLES:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises precipitation.
        for name in data_vars:
            if "precip" in str(name).lower():
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
            site_id = f"gpm_imerg:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"gpm_imerg:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"GPM IMERG precipitation over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
