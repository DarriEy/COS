# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GloH2O MSWEP precipitation connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
merged precipitation product. MSWEP (Multi-Source Weighted-Ensemble
Precipitation) is a global 0.1 deg product that blends gauge, satellite, and
reanalysis estimates, served as NetCDF tiles behind a GloH2O registration
(distributed via a Google-Drive share, accessed with rclone in the native
SYMFLUENCE acquirer). Each file carries a precipitation *depth* over the
file's accumulation window (3-hourly / daily / monthly). This connector:

1. opens an MSWEP NetCDF (a local cached file, or a downloaded one — GloH2O
   registration via the resolved credential);
2. extracts ``lat / lon / time`` and the precipitation variable
   (``precipitation`` / ``precip`` / ``pr`` / ``P`` / ``tp``) as numpy arrays,
   mirroring the native handler's variable search order;
3. masks non-finite cells to NaN so they reduce to MISSING;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the size policy made explicit and
   configurable here);
5. emits the canonical ``precipitation`` unit ``mm``. MSWEP precipitation is a
   depth *already in mm*, so the conversion at the boundary is the identity (no
   scaling), exactly as the native SYMFLUENCE handler passes the values through
   unchanged (it stores them as ``precip_mm`` and only sums for temporal
   aggregation — never rescales the per-cell depth).

The fetch path is exercised only with GloH2O credentials; the reduce +
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
#: native SYMFLUENCE MSWEP handler's search order.
PRECIP_VARIABLES = ("precipitation", "precip", "pr", "P", "tp")
#: <= this area (km²) defaults to point sampling (nearest cell).
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("mswep_precip")
class MSWEPPrecipConnector(BaseObservationConnector):
    slug = "mswep_precip"
    display_name = "GloH2O MSWEP Precipitation"
    kind = ObservationKind.PRECIPITATION
    structural_class = "gridded"
    base_url = "https://www.gloh2o.org"
    auth = frozenset({"gloh2o"})

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
                "MSWEP live fetch needs a NetCDF path (config 'nc_path'/'path') or a "
                "GloH2O rclone download (not yet wired). The reduction path is the "
                "proven part; supply a downloaded MSWEP NetCDF to reduce it.",
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
        """Open an MSWEP NetCDF, mask non-finite, reduce to the basin (mm)."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing an MSWEP precipitation variable (tried {PRECIP_VARIABLES})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        # Mask non-finite cells (fill / missing) to NaN so the reduction skips
        # them and they surface as MISSING. MSWEP precipitation is a depth in
        # mm; no scaling is applied — identity conversion at the boundary.
        values = np.where(np.isfinite(values), values, np.nan)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MSWEP basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MSWEP nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # already mm — identity conversion
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
                "source": "GloH2O MSWEP",
                "url": "https://www.gloh2o.org/mswep/",
                "license": "CC BY-NC 4.0",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    def _find_variable(self, ds: object) -> str | None:
        """Pick the precipitation variable, in the native preference order."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in PRECIP_VARIABLES:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises precipitation.
        for name in data_vars:
            lower = name.lower()
            if "precip" in lower or lower in ("pr", "tp", "p"):
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
            site_id = f"mswep:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"mswep:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MSWEP precipitation over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
