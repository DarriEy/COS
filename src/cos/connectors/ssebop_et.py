# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""USGS SSEBop actual-evapotranspiration connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** for the ``et`` kind. SSEBop
(operational Simplified Surface Energy Balance, USGS/EROS) is a satellite-derived
*actual* ET product served as gridded rasters (1 km daily CONUS, 10 km global
monthly). This connector mirrors the native SYMFLUENCE ``ssebop`` / ``ssebop_et``
observation handler:

1. opens a SSEBop NetCDF (a local cached file, or a downloaded one — USGS/EROS
   serves the rasters with no auth);
2. extracts ``lat / lon / time`` and the ET variable as numpy arrays;
3. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the size policy the gridded
   connectors share, made explicit and configurable here);
4. canonicalizes to **mm/day** — SSEBop ET is already mm/day, so this is a
   pass-through identity at the boundary (no scale factor for NetCDF; the native
   handler's /10 scaling applies only to the CONUS *GeoTIFF* product, not the
   processed NetCDF this connector reads) — and clips negatives to 0 and masks
   the native nodata sentinel, exactly mirroring the native handler.

The fetch path is exercised only against a supplied/downloaded file; the reduce +
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
    QualityFlag,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

#: SSEBop nodata sentinel used in the native handler's GeoTIFF/NetCDF reads.
NODATA = -9999.0
#: <= this area (km²) defaults to point sampling, mirroring the gridded policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: Candidate ET variable names, mirroring native ``_find_et_variable``.
ET_VAR_CANDIDATES = ("et", "ET", "eta", "ETa", "et_mm_day", "evapotranspiration")


@register("ssebop_et")
class SSEBopETConnector(BaseObservationConnector):
    slug = "ssebop_et"
    display_name = "USGS SSEBop actual ET"
    kind = ObservationKind.ET
    structural_class = "gridded"
    base_url = "https://edcintl.cr.usgs.gov"
    auth = frozenset()  # USGS/EROS SSEBop rasters are served anonymously.

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
                "SSEBop live fetch needs a NetCDF path (config 'nc_path' / 'path') or a "
                "USGS/EROS download (not yet wired). The reduction path is the proven "
                "part; supply a SSEBop ET NetCDF to reduce it.",
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
        """Open a SSEBop NetCDF, reduce to the basin, canonicalize to mm/day."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            et_var = self._find_et_variable(ds)
            if et_var is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF has no recognizable ET variable (looked for {ET_VAR_CANDIDATES})",
                )
            lat_name = self._find_coord(ds, ("lat", "latitude", "y"))
            lon_name = self._find_coord(ds, ("lon", "longitude", "x"))
            if lat_name is None or lon_name is None:
                raise ConnectorError(self.slug, "NetCDF missing lat/lon coordinates")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(ds[et_var].values, dtype="float64")  # (time, lat, lon)

        # Mask nodata + negatives -> NaN at the boundary (native handler masks
        # nodata and clips ET < 0). NaN propagates to QualityFlag.MISSING below.
        values = np.where(values == NODATA, np.nan, values)
        values = np.where(values < 0.0, np.nan, values)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("SSEBop basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("SSEBop nearest_cell requires spec.centroid")

        # SSEBop ET is already mm/day == KIND_UNITS[ET]; pass-through (no scale).
        points = reduce_grid(
            lats, lons, times, values,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)).
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]

        # Defensive non-negativity (matches native df['et_mm_day'].clip(lower=0)).
        for p in points:
            if p.value is not None and p.value < 0.0:
                p.value = 0.0
                p.quality = QualityFlag.SUSPECT

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "USGS SSEBop actual ET",
                "url": "https://www.usgs.gov/special-topics/water-resources/science/ssebop",
                "variable": "ETa",
                "native_unit": "mm/day",
            },
            fetched_at=datetime.now(UTC),
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _find_et_variable(ds: object) -> str | None:
        """Locate the ET variable, mirroring native ``_find_et_variable``."""
        data_vars = list(getattr(ds, "data_vars", {}))
        for var in ET_VAR_CANDIDATES:
            if var in data_vars:
                return var
        for var in data_vars:
            low = str(var).lower()
            if "et" in low or "evap" in low:
                return str(var)
        return None

    @staticmethod
    def _find_coord(ds: object, candidates: tuple[str, ...]) -> str | None:
        names = set(getattr(ds, "coords", {})) | set(getattr(ds, "dims", {}))
        lowered = {str(n).lower(): str(n) for n in names}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        return None

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"ssebop_et:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"ssebop_et:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region",
            site_id=site_id,
            latitude=lat,
            longitude=lon,
            name=f"SSEBop ET over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
