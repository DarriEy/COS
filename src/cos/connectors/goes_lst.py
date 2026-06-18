# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NOAA GOES-R ABI L2 Land Surface Temperature connector (gridded, basin-reduced).

GOES ABI LST has **no** SYMFLUENCE native handler, so this connector is
*spec-validated*: its unit, fill semantics, and DQF masking reproduce the
published NOAA GOES-R ABI L2+ Land Surface Temperature product spec (the
``ABI-L2-LSTC`` CONUS / ``ABI-L2-LSTF`` full-disk product family), and the
hermetic tests assert that contract on a synthetic fixture rather than against a
native reference series.

Product (NOAA GOES-R ABI L2 LST, served as NetCDF from the AWS NODD open bucket —
``s3://noaa-goes16`` / ``noaa-goes18``, anonymous, no auth):

* the gridded variable is ``LST`` (Land Surface Temperature), already reported in
  **Kelvin** — which is exactly the COS canonical ``lst`` unit
  (:data:`cos.core.models.KIND_UNITS`). The boundary scale is therefore the
  identity (:data:`SOURCE_LST_SCALE` ``= 1.0``); a non-identity scale would be
  applied here and nowhere else.
* a companion ``DQF`` (Data Quality Flag) array carries the per-pixel retrieval
  quality. DQF ``0`` is good quality; any non-zero DQF (degraded / invalid /
  no-retrieval) is masked. The retrieval fill (``_FillValue``), non-finite, and
  physically implausible Kelvin values (:data:`VALID_LST_RANGE`) are masked too,
  so they reduce to :class:`~cos.core.models.QualityFlag.MISSING`.

Sub-hourly geostationary LST (CONUS LSTC is hourly; full-disk LSTF is hourly,
mesoscale sub-hourly) is the differentiation versus polar-orbiting MODIS LST:
many observations per day over a fixed footprint rather than one or two
overpasses.

The real NODD ABI L2 granule is on the **fixed-grid ABI projection**: it ships
**no** lat/lon, only the geostationary scan/elevation angles ``x``/``y`` (radians)
and a ``goes_imager_projection`` grid-mapping. This connector geolocates those to
a 2-D lat/lon grid (:func:`_abi_fixed_grid_latlon`, the NOAA GOES-R PUG
fixed-grid → geodetic transform) and takes the dedicated 2-D-coordinate reduction
path (bbox mask + cos-lat-weighted mean / nearest valid cell), mirroring the AMSR2
EASE-Grid handling. A pre-regridded file carrying explicit 1-D or 2-D lat/lon is
also supported (the 1-D :func:`cos.core.reduce.reduce_grid` path), with dim order
normalized to ``(time, lat, lon)`` before reducing.

This connector extracts ``lats, lons, times, lst, dqf`` from a supplied file
(config ``nc_path`` / ``path`` — live AWS NODD S3 download is not yet wired,
exactly as the other gridded connectors), applies the DQF + fill + range mask at
the boundary so the canonical series is **K**, then reduces to the basin via
:mod:`cos.core.reduce` — ``basin_mean`` (cos-lat weighted) for larger basins,
``nearest_cell`` for small ones.

The architecture-critical extract→mask→reduce→canonicalize path is hermetically
tested via :meth:`GOESLSTConnector.reduce_arrays` on a synthetic in-memory grid,
with no network and no auth.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from cos.connectors.base import BaseObservationConnector
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

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

logger = structlog.get_logger()

#: Source→canonical scale. ABI L2 LST is already reported in Kelvin (== canonical
#: ``lst`` unit), so the boundary conversion is the identity.
SOURCE_LST_SCALE = 1.0
#: NetCDF fill sentinel commonly carried on the ABI L2 LST retrieval array.
FILL_VALUE = -9999.0
#: Good-quality DQF value. ABI L2 LST DQF: 0 = good_quality_qf; any non-zero flag
#: (degraded / invalid / no-retrieval) is not usable LST and is masked.
GOOD_DQF = 0
#: Physical-plausibility band for land surface temperature (Kelvin). ABI LST
#: spans roughly 213 K (-60 C) to 343 K (70 C); values outside are masked as
#: implausible retrievals.
VALID_LST_RANGE = (200.0, 360.0)
#: Candidate LST variable names, in preference order (matches ABI L2 LST layout).
LST_VARIABLES = ("LST", "lst", "land_surface_temperature", "LST_C")
#: Candidate DQF (data-quality-flag) variable names, in preference order.
DQF_VARIABLES = ("DQF", "dqf", "DQF_overall", "quality_flag")
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("goes_lst")
class GOESLSTConnector(BaseObservationConnector):
    slug = "goes_lst"
    display_name = "NOAA GOES-R ABI L2 Land Surface Temperature"
    kind = ObservationKind.LST
    structural_class = "gridded"
    #: AWS NODD open bucket — anonymous (no auth), s3://noaa-goes16 / noaa-goes18.
    base_url = "https://noaa-goes16.s3.amazonaws.com"
    auth = frozenset()  # anonymous AWS NODD open data — no credentials

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
                "GOES LST live fetch needs a cached NetCDF (config 'nc_path'/'path') — "
                "an ABI-L2-LSTC/LSTF granule. AWS NODD S3 download is not yet wired "
                "(anonymous s3://noaa-goes16 / noaa-goes18); the DQF-mask + reduce path "
                "is the proven part.",
            )
        return [self.reduce_file(Path(path), spec, start, end)]

    # -- file reader (extract arrays, then defer to the pure core) -----------

    def reduce_file(
        self,
        path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open a GOES ABI LST NetCDF, extract arrays, reduce + canonicalize."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path, mask_and_scale=True) as ds:
            var_name = self._find_variable(ds, LST_VARIABLES, "LST")
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing an ABI L2 LST variable (tried {LST_VARIABLES})",
                )
            dqf_name = self._find_variable(ds, DQF_VARIABLES, "DQF")
            has_latlon = (("lat" in ds) or (_coord_like(ds, "lat") in ds)) and (
                ("lon" in ds) or (_coord_like(ds, "lon") in ds)
            )
            if not has_latlon and "goes_imager_projection" in ds and "x" in ds and "y" in ds:
                # Native ABI L2 fixed-grid granule (the NODD product): no lat/lon
                # shipped — only the geostationary scan angles x/y (radians) and a
                # goes_imager_projection grid-mapping. Geolocate them to a 2-D
                # lat/lon grid and take the 2-D reduction path. A single-time
                # granule is (y, x); add the leading time axis the reducer expects.
                lats, lons = _abi_fixed_grid_latlon(ds)
                times = _granule_times(ds)
                values = _with_time_axis(np.asarray(ds[var_name].values, dtype="float64"))
                dqf = None
                if dqf_name is not None and dqf_name in ds:
                    dqf = _with_time_axis(np.asarray(ds[dqf_name].values, dtype="float64"))
            else:
                lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
                lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
                time_name = "time" if "time" in ds else _coord_like(ds, "time")
                # ABI L2 grids may be dim-ordered (lat, lon, time) or carry a 2-D
                # geolocated grid; normalize the LST (and DQF) dim order to
                # (time, lat, lon) by the dataset's own dim names before reducing.
                da = _to_time_lat_lon(ds[var_name], time_name, lat_name, lon_name)
                lats = np.asarray(ds[lat_name].values, dtype="float64")
                lons = np.asarray(ds[lon_name].values, dtype="float64")
                times = np.asarray(ds[time_name].values)
                values = np.asarray(da.values, dtype="float64")
                dqf = None
                if dqf_name is not None and dqf_name in ds:
                    dqf_da = _to_time_lat_lon(ds[dqf_name], time_name, lat_name, lon_name)
                    dqf = np.asarray(dqf_da.values, dtype="float64")
        return self.reduce_arrays(
            lats, lons, times, values, spec, start, end, dqf=dqf, var_name=var_name
        )

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        lst: np.ndarray,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        dqf: np.ndarray | None = None,
        var_name: str = "LST",
    ) -> ObservationSeries:
        """Mask DQF/fill/out-of-range, basin-reduce, window-trim → canonical series.

        *lst* is shaped ``(time, lat, lon)`` Land Surface Temperature in the source
        unit (Kelvin, == canonical). Cells whose DQF is non-zero (degraded /
        invalid), equal to :data:`FILL_VALUE`, non-finite, or outside
        :data:`VALID_LST_RANGE` become NaN and surface as MISSING; the rest are
        multiplied by :data:`SOURCE_LST_SCALE` (identity) so the canonical unit
        ``K`` is preserved inside the reduction.

        Coordinate shape is honoured: 1-D ``lat``/``lon`` vectors defer to
        :func:`cos.core.reduce.reduce_grid`; 2-D geostationary lat/lon take a
        dedicated bbox-mask reduction path.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        values = np.asarray(lst, dtype="float64")

        lo, hi = VALID_LST_RANGE
        invalid = (
            (values == FILL_VALUE)
            | ~np.isfinite(values)
            | (values < lo)
            | (values > hi)
        )
        if dqf is not None:
            dqf_arr = np.asarray(dqf, dtype="float64")
            # DQF == 0 (GOOD_DQF) is good quality; any other flag (incl. fill /
            # non-finite DQF) marks the pixel unusable.
            invalid = invalid | ~np.isfinite(dqf_arr) | (dqf_arr != GOOD_DQF)

        # Apply the source→canonical scale at the boundary, then mask. (Scale is
        # the identity here; written explicitly so a future non-identity product
        # has exactly one place to set it.)
        values = np.where(invalid, np.nan, values * SOURCE_LST_SCALE)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("GOES LST basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("GOES LST nearest_cell requires spec.centroid")

        if lats.ndim == 2 or lons.ndim == 2:
            # Geolocated ABI fixed-grid product: 2-D lat/lon. reduce_grid assumes
            # 1-D coord vectors (it indexes lat/lon axes independently), which
            # IndexErrors on a 2-D grid -> reduce over a bbox cell-mask instead.
            points = self._reduce_grid_2d(lats, lons, times, values, reduction, bbox, point)
        else:
            points = reduce_grid(
                lats, lons, times, values,
                reduction=reduction, bbox=bbox, point=point,
                kind=self.kind, unit=KIND_UNITS[self.kind],
            )

        # Window-trim, half-open UTC [start, end).
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= _utc(p.timestamp) < end_u]

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "NOAA GOES-R ABI L2 Land Surface Temperature",
                "product": "ABI-L2-LSTC/LSTF",
                "bucket": "s3://noaa-goes16 / noaa-goes18 (AWS NODD, anonymous)",
                "url": "https://registry.opendata.aws/noaa-goes/",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    def _reduce_grid_2d(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        values: np.ndarray,
        reduction: SpatialReduction,
        bbox: tuple[float, float, float, float] | None,
        point: tuple[float, float] | None,
    ) -> list[ObservationPoint]:
        """Reduce a 2-D-coordinate (geostationary fixed-grid) product to points.

        ``lats``/``lons`` are 2-D (ny, nx); ``values`` is (time, ny, nx). Off-disk
        ABI cells carry non-finite lat/lon; the bbox mask requires finite coords so
        they drop out. ``basin_mean`` is the cos-lat-weighted mean over the bbox
        cells; ``nearest_cell`` is the nearest valid in-grid cell to the centroid.

        Longitude convention is reconciled with the 1-D reducer: the request bbox
        is shifted into the grid's 0–360 convention (:func:`_normalize_lons`), and
        nearest-cell uses a great-circle (haversine) distance, which is accurate at
        high latitude / the disk edge and seam-robust across the 0/360 meridian.
        """
        import numpy as np

        from cos.core.models import QualityFlag
        from cos.core.reduce import _as_datetime, _normalize_lons

        lats = np.broadcast_to(np.asarray(lats, dtype="float64"), values.shape[1:])
        lons = np.broadcast_to(np.asarray(lons, dtype="float64"), values.shape[1:])
        finite_coord = np.isfinite(lats) & np.isfinite(lons)

        if reduction == SpatialReduction.BASIN_MEAN:
            if bbox is None:
                raise ReductionError("GOES LST basin_mean requires spec.bbox")
            lat_min, lon_min, lat_max, lon_max = bbox
            # Normalize against only the FINITE grid lons: off-disk cells carry inf,
            # which would poison _normalize_lons' nanmax-based 0–360 detection.
            lon_min, lon_max = _normalize_lons(lons[finite_coord], lon_min, lon_max)
            cell_mask = (
                finite_coord
                & (lats >= lat_min) & (lats <= lat_max)
                & (lons >= lon_min) & (lons <= lon_max)
            )
            if not cell_mask.any():
                raise ReductionError(
                    f"No ABI fixed-grid cells inside bbox {bbox} on the 2-D coordinate grid"
                )
            weights = np.cos(np.deg2rad(np.where(cell_mask, lats, 0.0)))
            series = np.full(values.shape[0], np.nan, dtype="float64")
            for t in range(values.shape[0]):
                layer = values[t]
                use = cell_mask & np.isfinite(layer)
                if not use.any():
                    continue
                wsum = float(np.sum(weights[use]))
                if wsum > 0:
                    series[t] = float(np.sum(layer[use] * weights[use]) / wsum)
        else:
            if point is None:
                raise ReductionError("GOES LST nearest_cell requires spec.centroid")
            plat, plon = point
            # Great-circle (haversine) distance: accurate near the disk edge / high
            # latitude where planar degree-distance distorts, and seam-robust across
            # the 0/360 meridian (sin(Δλ/2) is periodic), so no lon normalization is
            # needed here. The haversine term is monotonic in distance -> argmin ok.
            dphi = np.deg2rad(lats - plat)
            dlam = np.deg2rad(lons - plon)
            hav = (
                np.sin(dphi / 2.0) ** 2
                + np.cos(np.deg2rad(plat)) * np.cos(np.deg2rad(lats)) * np.sin(dlam / 2.0) ** 2
            )
            dist = np.where(finite_coord, hav, np.inf)
            flat_idx = int(np.argmin(dist))
            i, j = np.unravel_index(flat_idx, dist.shape)
            series = values[:, i, j].astype("float64")

        points: list[ObservationPoint] = []
        for t, v in zip(times, series):
            ts = t if isinstance(t, datetime) else _as_datetime(t)
            finite = v is not None and np.isfinite(v)
            points.append(
                ObservationPoint(
                    timestamp=ts,
                    value=float(v) if finite else None,
                    quality=QualityFlag.GOOD if finite else QualityFlag.MISSING,
                )
            )
        return points

    def _find_variable(self, ds: object, candidates: tuple[str, ...], hint: str) -> str | None:
        """Pick a variable by published name, then by any *hint*-like name."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in candidates:
            if name in data_vars:
                return name
        hint_lower = hint.lower()
        for name in data_vars:
            if hint_lower in str(name).lower():
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
            site_id = f"goes_lst:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"goes_lst:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"GOES ABI LST over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _coord_like(ds: object, want: str) -> str:
    for name in getattr(ds, "coords", {}):
        if want in str(name).lower():
            return str(name)
    return want


def _with_time_axis(arr: np.ndarray) -> np.ndarray:
    """Add a leading singleton time axis to a single-time 2-D ``(y, x)`` array.

    The reducer indexes ``(time, lat, lon)``; a native single-time ABI granule
    ships ``(y, x)``. Higher-rank arrays pass through unchanged.
    """
    return arr[None, ...] if arr.ndim == 2 else arr


def _granule_times(ds: object) -> np.ndarray:
    """The ABI granule timestamp(s) as a 1-D array (scan-start ``t``, else ``time``)."""
    import numpy as np

    for name in ("t", "time"):
        if name in ds:  # type: ignore[operator]
            return np.atleast_1d(np.asarray(ds[name].values))  # type: ignore[index]
    return np.atleast_1d(np.datetime64("1970-01-01"))


def _abi_fixed_grid_latlon(ds: object) -> tuple[np.ndarray, np.ndarray]:
    """Geolocate ABI fixed-grid scan angles to 2-D geodetic lat/lon (degrees).

    Implements the NOAA GOES-R PUG fixed-grid → geodetic transform: the
    ``goes_imager_projection`` grid-mapping (perspective-point height, GRS-80
    semi-axes, sub-satellite longitude) plus the per-pixel scan/elevation angles
    ``x``/``y`` (radians) give each cell's latitude/longitude. Off-disk cells
    (negative intersection discriminant) come back ``NaN`` and are dropped by the
    2-D reducer's finite-coordinate mask.
    """
    import numpy as np

    p = ds["goes_imager_projection"]  # type: ignore[index]
    r_eq = float(p.attrs["semi_major_axis"])
    r_pol = float(p.attrs["semi_minor_axis"])
    h_sat = float(p.attrs["perspective_point_height"]) + r_eq
    lon0 = np.deg2rad(float(p.attrs["longitude_of_projection_origin"]))

    x = np.asarray(ds["x"].values, dtype="float64")  # type: ignore[index]
    y = np.asarray(ds["y"].values, dtype="float64")  # type: ignore[index]
    grid_x, grid_y = np.meshgrid(x, y)  # (ny, nx), matching the (y, x) data grid

    with np.errstate(invalid="ignore"):
        a = (
            np.sin(grid_x) ** 2
            + np.cos(grid_x) ** 2
            * (np.cos(grid_y) ** 2 + (r_eq**2 / r_pol**2) * np.sin(grid_y) ** 2)
        )
        b = -2.0 * h_sat * np.cos(grid_x) * np.cos(grid_y)
        c = h_sat**2 - r_eq**2
        disc = b**2 - 4.0 * a * c
        r_s = np.where(disc >= 0.0, (-b - np.sqrt(disc)) / (2.0 * a), np.nan)
        s_x = r_s * np.cos(grid_x) * np.cos(grid_y)
        s_y = -r_s * np.sin(grid_x)
        s_z = r_s * np.cos(grid_x) * np.sin(grid_y)
        lat = np.degrees(
            np.arctan((r_eq**2 / r_pol**2) * (s_z / np.sqrt((h_sat - s_x) ** 2 + s_y**2)))
        )
        lon = np.degrees(lon0 - np.arctan(s_y / (h_sat - s_x)))
    return lat, lon


def _to_time_lat_lon(
    da: xr.DataArray, time_name: str, lat_name: str, lon_name: str
) -> xr.DataArray:
    """Transpose a DataArray to ``(time, lat, lon)`` by its own dim names.

    ABI L2 grids may ship ``(lat, lon, time)`` (or a 2-D geolocated grid) while
    :func:`cos.core.reduce.reduce_grid`/``basin_mean`` index ``(time, lat, lon)``.
    Reorder only the dims that exist (a 2-D single-time grid has no time dim),
    keeping any unexpected leading dims ahead of the canonical trailing axes.
    """
    dims = tuple(str(d) for d in da.dims)
    wanted = [d for d in (time_name, lat_name, lon_name) if d in dims]
    if not wanted:
        return da
    leading = [d for d in dims if d not in wanted]
    order = leading + wanted
    if order == list(dims):
        return da
    return da.transpose(*order)
