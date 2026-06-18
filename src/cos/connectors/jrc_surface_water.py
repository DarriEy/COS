# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""JRC Global Surface Water connector (gridded, basin-reduced).

Ports SYMFLUENCE's native ``jrc_water`` / ``jrc_gsw`` / ``surface_water``
observation handler (``data/observation/handlers/jrc_water.py``) onto the COS
canonical contract. The native handler ingests JRC (EC Joint Research Centre)
Global Surface Water rasters — 30 m Landsat-derived layers (occurrence,
recurrence, seasonality, ...) — and reduces them to basin-level statistics
(``occurrence_mean`` over valid pixels) for reservoir / lake / wetland
monitoring and model validation.

Source semantics mirrored exactly from the native handler:

* **variable**: the JRC ``occurrence`` layer by default — water occurrence as a
  *percent of valid observations* in ``[0, 100]`` over the full 1984-2021 epoch.
  The native handler's primary statistic is ``occurrence_mean`` = the arithmetic
  mean over valid pixels;
* **valid range / fill**: bytes outside ``[0, 100]`` are masked. The native
  handler uses ``nodata`` (255 fallback) and ``data >= 0`` as the valid mask;
  here every value outside ``[0, 100]`` (including the 255 fill byte) is masked
  to NaN and surfaces as ``QualityFlag.MISSING``;
* **reduction**: basin-mean over the bbox (the COS gridded path), or
  ``nearest_cell`` for small basins, matching grace.py's size policy. The
  native ``occurrence_mean`` is an unweighted pixel mean; the COS ``basin_mean``
  kernel applies cos-latitude weighting — a documented tolerance-based
  approximation, identical to the framework's other gridded connectors;
* **units**: source is occurrence *percent* (0-100). The canonical
  ``surface_water`` unit is ``"fraction"`` (``KIND_UNITS[SURFACE_WATER]``), so a
  percent->fraction (``/100``) conversion happens at the connector boundary.

JRC Global Surface Water is a *static* multi-decadal aggregate (no time axis):
the reduced value is emitted as a single :class:`ObservationPoint` stamped at the
window start, representing the 1984-2021 epoch occurrence fraction. The window
``[start, end)`` is half-open UTC; the epoch point is kept iff it falls inside.

The download path (Google Cloud Storage public bucket, no auth) mirrors the
native acquirer but is not wired here — as with grace.py / modis_sca.py the
proven part is the reduce + canonicalize path, exercised hermetically against a
supplied raster (config ``nc_path`` / ``path``).
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

#: JRC occurrence valid percent range; everything else is fill / nodata.
VALID_OCCURRENCE_RANGE = (0.0, 100.0)
#: JRC nodata byte fallback (native handler default when src.nodata is None).
JRC_FILL_VALUE = 255
PERCENT_TO_FRACTION = 1.0 / 100.0
#: <= this area (km²) defaults to nearest_cell, mirroring grace.py's policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: occurrence-layer variable names, native-handler / common netCDF order.
OCCURRENCE_VARIABLES = ("occurrence", "Band1", "band1", "surface_water", "water")
#: JRC product epoch — the static occurrence raster aggregates 1984-2021.
JRC_EPOCH_START = "1984-01-01"


@register("jrc_surface_water")
class JRCSurfaceWaterConnector(BaseObservationConnector):
    slug = "jrc_surface_water"
    display_name = "JRC Global Surface Water (occurrence)"
    kind = ObservationKind.SURFACE_WATER
    structural_class = "gridded"
    base_url = "https://storage.googleapis.com/global-surface-water"
    auth = frozenset()  # JRC GSW is a public GCS bucket — no auth.

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
                "JRC surface-water live fetch needs a raster path (config 'nc_path' "
                "or 'path') or a GCS download (not yet wired). The reduction path is "
                "the proven part; supply a JRC Global Surface Water occurrence "
                "GeoTIFF / NetCDF to reduce it.",
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
        """Open a JRC raster, reduce to the basin, canonicalize to fraction.

        Reads either a GeoTIFF (native JRC format, via rasterio) or a NetCDF,
        masks fill / out-of-range bytes to NaN, reduces over the basin, and
        converts percent->fraction at the boundary so values are in the canonical
        ``surface_water`` unit. The static epoch value is window-trimmed to the
        half-open UTC ``[start, end)`` interval.
        """
        suffix = nc_path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            lats, lons, values = self._read_geotiff(nc_path)
        else:
            lats, lons, values = self._read_netcdf(nc_path)
        return self.reduce_arrays(lats, lons, values, spec, start, end)

    def reduce_arrays(
        self,
        lats,
        lons,
        values,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Pure (network-free) reduce + canonicalize of a JRC occurrence grid.

        *values* is a single 2-D occurrence-percent layer ``(lat, lon)``. Masks
        out-of-range / fill to NaN, reduces over the basin, converts
        percent->fraction, and emits one epoch :class:`ObservationPoint` stamped
        at ``JRC_EPOCH_START`` — kept iff it falls in half-open ``[start, end)``.
        """
        import numpy as np

        reduction = self._choose_reduction(spec)

        # Quality filter: keep occurrence percent in [0, 100], mask fill (255)
        # and any out-of-range byte to NaN (mirrors native valid_mask:
        # (data != nodata) & (data >= 0), with the [0,100] upper bound).
        values = self._mask_invalid(values)

        # Percent -> fraction at the boundary (canonical surface_water unit).
        values = values * PERCENT_TO_FRACTION

        # The static raster has no time axis; add a length-1 time dimension so
        # the shared reduce_grid kernel ((time, lat, lon)) applies unchanged.
        values_3d = values[np.newaxis, :, :]
        epoch = _utc(datetime.fromisoformat(JRC_EPOCH_START))
        times = np.array([np.datetime64(JRC_EPOCH_START)], dtype="datetime64[ns]")

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("JRC surface-water basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("JRC surface-water nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values_3d,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)) — the epoch point survives iff
        # the requested window covers JRC_EPOCH_START.
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]
        _ = epoch  # documents the epoch stamp; kept for clarity.

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "JRC Global Surface Water",
                "source_doi": "10.1038/nature20584",
                "url": "https://global-surface-water.appspot.com",
                "layer": "occurrence",
                "epoch": "1984-2021",
            },
            fetched_at=datetime.now(UTC),
        )

    # -- IO (network-free file reads) ----------------------------------------

    def _read_geotiff(self, path: Path):
        """Read a JRC occurrence GeoTIFF -> (lats, lons, values) on EPSG:4326.

        Returns the first band as a ``(nlat, nlon)`` occurrence-percent layer on
        a regular lat/lon grid derived from the raster transform (reprojected to
        EPSG:4326 if the source CRS is projected).
        """
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform

        with rasterio.open(path) as src:
            data = src.read(1).astype("float64")  # first band, (rows, cols)
            nodata = src.nodata
            transform = src.transform
            crs = src.crs
            rows, cols = src.height, src.width

            col_idx = np.arange(cols) + 0.5
            row_idx = np.arange(rows) + 0.5
            xs = transform.c + transform.a * col_idx  # x at each column center
            ys = transform.f + transform.e * row_idx  # y at each row center
            if crs is not None and not crs.is_geographic:
                xs_lon, _ = warp_transform(crs, "EPSG:4326", xs.tolist(), [ys[0]] * cols)
                _, ys_lat = warp_transform(crs, "EPSG:4326", [xs[0]] * rows, ys.tolist())
                lons = np.asarray(xs_lon, dtype="float64")
                lats = np.asarray(ys_lat, dtype="float64")
            else:
                lons = xs.astype("float64")
                lats = ys.astype("float64")

        if nodata is not None:
            data[data == nodata] = np.nan
        return lats, lons, data

    def _read_netcdf(self, path: Path):
        """Read a JRC occurrence NetCDF -> (lats, lons, values), (lat, lon)."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var_name = self._find_occurrence_variable(ds)
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds.coords else ("y" if "y" in ds.coords else "lat")
            lon_name = "lon" if "lon" in ds.coords else ("x" if "x" in ds.coords else "lon")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            values = np.asarray(da.values, dtype="float64")
            # Squeeze a singleton time / band axis if present -> (lat, lon).
            values = np.squeeze(values)
        return lats, lons, values

    @staticmethod
    def _mask_invalid(values):
        """Mask occurrence bytes outside [0, 100] (fill / nodata) to NaN."""
        import numpy as np

        lo, hi = VALID_OCCURRENCE_RANGE
        out = np.asarray(values, dtype="float64").copy()
        invalid = ~((out >= lo) & (out <= hi))
        out[invalid] = np.nan
        return out

    @staticmethod
    def _find_occurrence_variable(ds) -> str:
        """Find the occurrence variable, native-handler / common order."""
        for var in OCCURRENCE_VARIABLES:
            if var in ds.data_vars:
                return str(var)
        suitable = [
            v for v in ds.data_vars
            if "occur" in str(v).lower() or "water" in str(v).lower()
        ]
        if suitable:
            return str(suitable[0])
        raise ConnectorError(
            "jrc_surface_water",
            f"No occurrence / water variable found in dataset. Available: {list(ds.data_vars)}",
        )

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"jrc_surface_water:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"jrc_surface_water:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"JRC surface water over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# Re-export so a quality flag is importable alongside the connector for tests.
__all__ = ["JRCSurfaceWaterConnector", "QualityFlag", "ObservationPoint"]
