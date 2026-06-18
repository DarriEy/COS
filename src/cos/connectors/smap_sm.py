# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NASA SMAP surface soil-moisture connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
volumetric soil-moisture product. SMAP L3/L4 radiometer products carry surface
(and root-zone) volumetric soil moisture on a ~9 km grid behind NASA Earthdata.
This connector:

1. opens a SMAP NetCDF (a local cached file, or a downloaded one — Earthdata
   auth via the resolved credential token);
2. extracts ``lat / lon / time`` and the soil-moisture variable
   (``soil_moisture`` / ``sm_surface`` / ``sm_rootzone``) as numpy arrays;
3. masks the native fill value (``-9999``) and clips to the physical valid range
   ``0 < sm < 1`` (exactly the mask the native SYMFLUENCE handler applies),
   turning out-of-range / fill cells into NaN so they reduce to MISSING;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the size policy made explicit and
   configurable here);
5. emits the canonical ``soil_moisture`` unit ``m3/m3``. SMAP volumetric soil
   moisture is *already* m³/m³, so the conversion at the boundary is the identity
   (no scaling), mirroring the native handler which passes the values through
   unchanged.

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

#: SMAP native fill value for the soil-moisture retrieval arrays.
FILL_VALUE = -9999.0
#: Candidate soil-moisture variable names, in preference order (surface first),
#: mirroring the native SYMFLUENCE SMAP handler.
SM_VARIABLES = ("soil_moisture", "sm_surface", "sm_rootzone", "sm")
#: <= this area (km²) defaults to point sampling (nearest cell).
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("smap_sm")
class SMAPSoilMoistureConnector(BaseObservationConnector):
    slug = "smap_sm"
    display_name = "NASA SMAP Surface Soil Moisture"
    kind = ObservationKind.SOIL_MOISTURE
    structural_class = "gridded"
    base_url = "https://n5eil01u.ecs.nsidc.org"
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
                "SMAP live fetch needs a NetCDF path (config 'nc_path'/'path') or "
                "Earthdata download (not yet wired). The reduction path is the proven "
                "part; supply a downloaded SMAP NetCDF to reduce it.",
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
        """Open a SMAP NetCDF, mask fill/out-of-range, reduce to the basin (m³/m³)."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a SMAP soil-moisture variable (tried {SM_VARIABLES})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        # Mask the native fill value and clip to the physical valid range
        # (0 < sm < 1) exactly as the native handler does: invalid cells become
        # NaN so the reduction skips them and they surface as MISSING.
        invalid = (
            (values == FILL_VALUE)
            | ~np.isfinite(values)
            | (values <= 0.0)
            | (values >= 1.0)
        )
        values = np.where(invalid, np.nan, values)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("SMAP basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("SMAP nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # already m³/m³ — identity conversion
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
                "source": "NASA SMAP L3",
                "source_doi": "10.5067/HH4SZ2PXSP6A",
                "url": "https://nsidc.org/data/spl3smp",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    def _find_variable(self, ds: object) -> str | None:
        """Pick the soil-moisture variable, surface preferred (native order)."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in SM_VARIABLES:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises soil moisture.
        for name in data_vars:
            lower = name.lower()
            if "soil_moisture" in lower or "sm_surface" in lower or "sm_rootzone" in lower:
                return str(name)
        return None

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"smap:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"smap:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"SMAP soil moisture over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
