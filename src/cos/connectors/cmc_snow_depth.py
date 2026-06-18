# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""CMC Daily Snow Depth Analysis → physical snow depth (gridded, basin-reduced).

Ports SYMFLUENCE's native ``cmc_snow`` observation handler
(``data/observation/handlers/cmc_snow.py``) onto the COS canonical contract, but
emits **physical snow depth in metres** (:class:`ObservationKind.SNOW_DEPTH`)
rather than the SWE the sibling :mod:`cos.connectors.cmc_swe` connector derives.

CMC source (NSIDC-0447):
    * Canadian Meteorological Centre Daily Snow Depth Analysis (station +
      satellite assimilation), Northern-Hemisphere ~24 km grid.
    * Native distribution is yearly multi-band GeoTIFFs
      (``cmc_sdepth_dly_<year>_v01.2.tif``) with one band per day-of-year, the
      band value being **snow depth in cm**.
    * The native handler masks the file nodata plus any value ``< 0`` or
      ``> 999`` (cm), takes the basin-mean depth, then (for SWE) converts depth →
      SWE with a bulk density. This connector stops at the depth: it converts the
      masked basin-mean **depth (cm) → depth (m)** by dividing by 100 — the
      canonical unit for :class:`ObservationKind.SNOW_DEPTH` per
      :data:`cos.core.models.KIND_UNITS` (``"m"``).

This connector reuses the cm-grid extract path of the SWE connector exactly: it
extracts ``lats, lons, times, depth_cm`` from a supplied file (GeoTIFF or NetCDF —
a cached file is supplied via config ``nc_path`` / ``path``; live NSIDC/Earthdata
download is not yet wired) and reduces it with :func:`cos.core.reduce.reduce_grid`
— ``basin_mean`` (cos-lat weighted) for larger basins, ``nearest_cell`` for small
ones, the size policy the native handler's bbox-mean implies.

The architecture-critical extract→mask→convert→reduce→canonicalize path is
hermetically tested via :meth:`CMCSnowDepthConnector.reduce_arrays` on a synthetic
in-memory grid, with no network, no auth, and no rasterio dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import structlog

from cos.connectors.base import BaseObservationConnector
from cos.connectors.cmc_swe import CMCSnowSWEConnector
from cos.core.exceptions import ConnectorError, ReductionError
from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    ObservationPoint,
    ObservationSeries,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

#: Native physical-plausibility mask on snow depth (cm): keep 0 <= d <= this.
MAX_DEPTH_CM = 999.0
#: cm → m conversion (canonical SNOW_DEPTH unit is metres).
CM_TO_M = 1.0 / 100.0
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("cmc_snow_depth")
class CMCSnowDepthConnector(BaseObservationConnector):
    slug = "cmc_snow_depth"
    display_name = "CMC Daily Snow Depth Analysis (depth)"
    kind = ObservationKind.SNOW_DEPTH
    structural_class = "gridded"
    base_url = "https://n5eil01u.ecs.nsidc.org"
    auth = frozenset({"earthdata"})  # NSIDC-0447 download needs Earthdata

    VARIABLE = "snow_depth"  # NetCDF variable name when a NetCDF is supplied

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
        path = self.config.get("nc_path") or self.config.get("path")
        if not path:
            raise ConnectorError(
                self.slug,
                "CMC live fetch needs a cached file (config 'nc_path'/'path') — a "
                "yearly CMC snow-depth GeoTIFF or a NetCDF. NSIDC Earthdata download "
                "is not yet wired; the reduce + cm→m depth path is the proven part.",
            )
        return [self.reduce_file(Path(path), spec, start, end)]

    # -- file readers (reuse the SWE connector's proven cm-grid extract) ------

    def reduce_file(
        self,
        path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Read a CMC GeoTIFF/NetCDF, extract arrays, reduce + canonicalize.

        Reuses :class:`cos.connectors.cmc_swe.CMCSnowSWEConnector`'s GeoTIFF /
        NetCDF readers (the SAME yearly Polar-Stereographic GeoTIFF: native-
        resolution reproject + windowed day-of-year bands), then reduces to
        canonical depth in metres rather than SWE.
        """
        reader = CMCSnowSWEConnector(self.config)
        reader.VARIABLE = self.VARIABLE
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            lats, lons, times, depth_cm = reader._read_geotiff(path, start, end)
        else:
            lats, lons, times, depth_cm = reader._read_netcdf(path)
        return self.reduce_arrays(lats, lons, times, depth_cm, spec, start, end)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: object,
        lons: object,
        times: object,
        depth_cm: object,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Mask, basin-reduce, depth cm→m, window-trim → canonical series.

        *depth_cm* is shaped ``(time, lat, lon)`` snow depth in **cm**. Mirrors the
        native handler's depth path exactly: mask values outside ``[0, 999]`` cm
        (NaN already applied for the file nodata), reduce to the basin, then convert
        the basin-mean depth ``cm → m`` (× 1/100). The cm→m scale is linear so
        applying it pre-reduction is identical to the native post-reduction order
        and keeps the canonical unit (m) inside :func:`reduce_grid`.
        """
        lats_a = np.asarray(lats, dtype="float64")
        lons_a = np.asarray(lons, dtype="float64")
        times_a = np.asarray(times)
        depth = np.asarray(depth_cm, dtype="float64")

        # Native physical-plausibility mask on the depth grid (cm) -> fill/MISSING.
        depth = np.where((depth < 0) | (depth > MAX_DEPTH_CM), np.nan, depth)

        # Convert depth (cm) → depth (m) at the boundary, before the spatial mean.
        depth_m = depth * CM_TO_M

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("CMC basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("CMC nearest_cell requires spec.centroid")

        from cos.core.reduce import reduce_grid

        points = reduce_grid(
            lats_a, lons_a, times_a, depth_m,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim, half-open UTC [start, end).
        start_u = _utc(start)
        end_u = _utc(end)
        points = self._trim(points, start_u, end_u)

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "CMC Daily Snow Depth Analysis",
                "product": "NSIDC-0447",
                "url": "https://nsidc.org/data/nsidc-0447",
                "variable": "snow_depth",
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _trim(points: list[ObservationPoint], start_u: datetime, end_u: datetime) -> list[ObservationPoint]:
        return [p for p in points if start_u <= _utc(p.timestamp) < end_u]

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"cmc_snow_depth:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"cmc_snow_depth:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"CMC snow depth over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
