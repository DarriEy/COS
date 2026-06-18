# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""CMC Daily Snow Depth Analysis → SWE connector (gridded, basin-reduced).

Ports SYMFLUENCE's native ``cmc_snow`` / ``cmc_swe`` observation handler
(``data/observation/handlers/cmc_snow.py``) onto the COS canonical contract.

CMC source (NSIDC-0447):
    * Canadian Meteorological Centre Daily Snow Depth Analysis (station +
      satellite assimilation), Northern-Hemisphere ~24 km grid.
    * Native distribution is yearly multi-band GeoTIFFs
      (``cmc_sdepth_dly_<year>_v01.2.tif``) with one band per day-of-year, the
      band value being **snow depth in cm**.
    * The native handler masks the file nodata plus any value ``< 0`` or
      ``> 999`` (cm), takes the basin-mean depth, then converts depth → SWE with
      a configurable bulk snow density (default 200 kg/m³):

          swe_mm = mean_depth_cm * (snow_density / 100.0)

      (cm/100 = m of snow; × kg/m³ = kg/m² = mm w.e.). Finally it clips SWE to
      be non-negative.

This connector reproduces those semantics exactly, but on the COS gridded path:
it extracts ``lats, lons, times, depth_cm`` from a supplied file (GeoTIFF or
NetCDF — live Earthdata/NSIDC download is wired per-connector only where trivial,
so a cached file is supplied via config ``nc_path`` / ``path``) and reduces it
with :func:`cos.core.reduce.reduce_grid` — ``basin_mean`` (cos-lat weighted) for
larger basins, ``nearest_cell`` for small ones, the size policy the native
handler's bbox-mean implies.

The architecture-critical extract→mask→convert→reduce→canonicalize path is
hermetically tested via :meth:`CMCSnowSWEConnector.reduce_arrays` on a synthetic
in-memory grid, with no network, no auth, and no rasterio dependency.
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
    ObservationPoint,
    ObservationSeries,
    QualityFlag,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

#: Native default bulk snow density (kg/m³) for depth→SWE conversion.
DEFAULT_SNOW_DENSITY = 200.0
#: Native physical-plausibility mask on snow depth (cm): keep 0 <= d <= this.
MAX_DEPTH_CM = 999.0
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("cmc_swe")
class CMCSnowSWEConnector(BaseObservationConnector):
    slug = "cmc_swe"
    display_name = "CMC Daily Snow Depth Analysis (SWE)"
    kind = ObservationKind.SWE
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
                "is not yet wired; the reduce + depth→SWE path is the proven part.",
            )
        return [self.reduce_file(Path(path), spec, start, end)]

    # -- file readers (extract arrays, then defer to the pure core) ----------

    def reduce_file(
        self,
        path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Read a CMC GeoTIFF/NetCDF, extract arrays, reduce + canonicalize."""
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            lats, lons, times, depth_cm = self._read_geotiff(path)
        else:
            lats, lons, times, depth_cm = self._read_netcdf(path)
        return self.reduce_arrays(lats, lons, times, depth_cm, spec, start, end)

    def _read_netcdf(self, path: Path):
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var = self.VARIABLE if self.VARIABLE in ds else _first_3d_var(ds)
            if var is None:
                raise ConnectorError(self.slug, f"NetCDF missing '{self.VARIABLE}' variable")
            da = ds[var]
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            depth_cm = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
        return lats, lons, times, depth_cm

    def _read_geotiff(self, path: Path):
        """Read a yearly multi-band CMC GeoTIFF → (lats, lons, times, depth_cm).

        Band ``b`` (1-based) is day-of-year ``b`` of ``year`` (from the filename).
        Returns a ``(nbands, nlat, nlon)`` depth-cm cube on a regular lat/lon grid
        derived from the raster transform (reprojected to EPSG:4326 if needed).
        """
        import re

        import numpy as np
        import rasterio
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        m = re.search(r"(\d{4})", path.name)
        if not m:
            raise ConnectorError(self.slug, f"Cannot extract year from filename {path.name!r}")
        year = int(m.group(1))

        with rasterio.open(path) as src:
            nodata = src.nodata
            if src.crs is not None and not src.crs.is_geographic:
                # The CMC product is Polar Stereographic: lat/lon vary in 2D and
                # CANNOT be factored into 1D axes by warping a single row/column
                # (that produced physically-wrong axes and dropped every real NH
                # bbox). Reproject the whole raster onto a REGULAR EPSG:4326 grid
                # so lat/lon are true 1D axes that reduce_grid can consume.
                dst_crs = "EPSG:4326"
                dst_transform, dst_w, dst_h = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                data = np.full((src.count, dst_h, dst_w), np.nan, dtype="float64")
                for b in range(src.count):
                    reproject(
                        source=rasterio.band(src, b + 1),
                        destination=data[b],
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=dst_transform, dst_crs=dst_crs,
                        src_nodata=nodata, dst_nodata=np.nan,
                        resampling=Resampling.nearest,
                    )
                transform, rows, cols = dst_transform, dst_h, dst_w
            else:
                data = src.read().astype("float64")  # (bands, rows, cols)
                transform, rows, cols = src.transform, src.height, src.width

            # Cell-center lon/lat for each row/col on the (now-regular) grid.
            col_idx = np.arange(cols) + 0.5
            row_idx = np.arange(rows) + 0.5
            lons = (transform.c + transform.a * col_idx).astype("float64")
            lats = (transform.f + transform.e * row_idx).astype("float64")

        if nodata is not None:
            data[data == nodata] = np.nan

        # Day-of-year per band → timestamps for this year.
        from datetime import timedelta

        times = np.array(
            [
                np.datetime64(
                    (datetime(year, 1, 1) + timedelta(days=int(b))).strftime("%Y-%m-%d")
                )
                for b in range(data.shape[0])
            ],
            dtype="datetime64[ns]",
        )
        return lats, lons, times, data

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats,
        lons,
        times,
        depth_cm,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Mask, basin-reduce, depth→SWE, clip, window-trim → canonical series.

        *depth_cm* is shaped ``(time, lat, lon)`` snow depth in **cm**. Mirrors the
        native handler exactly: mask values outside ``[0, 999]`` cm (NaN already
        applied for the file nodata), reduce to the basin, convert
        ``swe_mm = depth_cm * density / 100``, then clip SWE to be non-negative.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        depth = np.asarray(depth_cm, dtype="float64")

        # Native physical-plausibility mask on the depth grid (cm).
        depth = np.where((depth < 0) | (depth > MAX_DEPTH_CM), np.nan, depth)

        density = float(spec.options.get("snow_density", self.config.get("snow_density", DEFAULT_SNOW_DENSITY)))
        # Convert depth (cm) → SWE (mm) at the boundary, before the spatial mean.
        # Linear, so applying it pre-reduction is identical to the native
        # post-reduction order and keeps the canonical unit (mm) inside reduce_grid.
        swe = depth * (density / 100.0)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("CMC basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("CMC nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, swe,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Clip SWE non-negative (native df['swe_mm'].clip(lower=0)).
        for p in points:
            if p.value is not None and p.value < 0:
                p.value = 0.0

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
                "snow_density_kg_m3": f"{density:g}",
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
            site_id = f"cmc_swe:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"cmc_swe:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"CMC SWE over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _first_3d_var(ds) -> str | None:
    for name, da in ds.data_vars.items():
        if da.ndim == 3:
            return str(name)
    return None


def _coord_like(ds, want: str) -> str:
    for name in ds.coords:
        if want in str(name).lower():
            return str(name)
    return want
