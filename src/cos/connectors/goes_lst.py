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

The real ABI L2 grid is the **fixed-grid ABI projection** (geostationary
scan/elevation angles), so a supplied LST NetCDF may carry **2-D** lat/lon (the
geolocated curvilinear grid) as well as the simpler 1-D regridded case. This
connector therefore carries both the 1-D :func:`cos.core.reduce.reduce_grid`
path and a dedicated 2-D-coordinate reduction path (bbox mask + cos-lat-weighted
mean / nearest valid cell), mirroring the AMSR2 EASE-Grid handling, and
normalizes dim order to ``(time, lat, lon)`` before reducing.

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
        """
        import numpy as np

        from cos.core.models import QualityFlag
        from cos.core.reduce import _as_datetime

        lats = np.broadcast_to(np.asarray(lats, dtype="float64"), values.shape[1:])
        lons = np.broadcast_to(np.asarray(lons, dtype="float64"), values.shape[1:])
        finite_coord = np.isfinite(lats) & np.isfinite(lons)

        if reduction == SpatialReduction.BASIN_MEAN:
            if bbox is None:
                raise ReductionError("GOES LST basin_mean requires spec.bbox")
            lat_min, lon_min, lat_max, lon_max = bbox
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
            dist = np.where(
                finite_coord,
                (lats - plat) ** 2 + (lons - plon) ** 2,
                np.inf,
            )
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
