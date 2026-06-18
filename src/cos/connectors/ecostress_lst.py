# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""ECOSTRESS land-surface-temperature (LST) connector (gridded, basin-reduced).

ECOSTRESS LST has **no** SYMFLUENCE native handler, so this connector is
*spec-validated*: its scale, fill semantics, valid range, and unit reproduce the
published LP DAAC ECOSTRESS L2 LST product spec (ECO2LSTE.001) and the hermetic
tests assert that contract on a synthetic fixture rather than against a native
reference series.

Product (ECOSTRESS L2 LST, ``ECO2LSTE.001``, served by the LP DAAC behind NASA
Earthdata; ISS-borne ECOSTRESS radiometer, ~70 m high-resolution thermal):

* the gridded variable is the retrieved land-surface temperature (``LST`` /
  ``SDS/LST``), stored as a scaled unsigned integer where the canonical Kelvin
  value is ``DN * scale_factor`` with the published ``scale_factor = 0.02``
  (:data:`SOURCE_LST_SCALE`). The product is already in **Kelvin** once scaled,
  which is exactly the COS canonical ``lst`` unit (:data:`cos.core.models.KIND_UNITS`).
* the no-retrieval fill is ``0`` (:data:`LST_FILL_VALUE`); cells equal to the
  fill (DN 0), non-finite, or outside the physical valid band
  (:data:`VALID_LST_RANGE`, Kelvin) are masked to NaN so they reduce to
  :class:`~cos.core.models.QualityFlag.MISSING`.

This connector:

1. opens an ECOSTRESS LST file (a local cached file supplied via config
   ``nc_path`` / ``path`` — Earthdata/LP DAAC download is not wired here, the
   reduce + canonicalize path is the proven part);
2. extracts ``lat / lon / time`` and the LST variable as numpy arrays,
   normalizing any ``(lat, lon, time)`` dim ordering to ``(time, lat, lon)``;
3. masks fill / out-of-range cells, applies the source→canonical scale
   (``DN * 0.02`` → K) at the boundary;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` (cos-lat
   weighted) for larger basins, ``nearest_cell`` for small ones — and emits the
   canonical ``lst`` unit ``K``.

ECOSTRESS swaths are high-resolution and can ship **2-D** lat/lon geolocation
arrays; :meth:`reduce_arrays` therefore carries a dedicated 2-D-coordinate
reduction path (bbox cell-mask + cos-lat-weighted mean / nearest valid cell)
alongside the 1-D :func:`cos.core.reduce.reduce_grid` path, and reorders the
DataArray by its own dim names so a ``(lat, lon, time)`` product is normalized
before reducing.

The architecture-critical extract→transpose→mask→scale→reduce→canonicalize path
is hermetically tested via :meth:`ECOSTRESSLSTConnector.reduce_arrays` on a
synthetic in-memory grid, with no network, no auth, and no file dependency.
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

#: Published ECO2LSTE.001 source→canonical scale: stored DN * 0.02 -> Kelvin.
SOURCE_LST_SCALE = 0.02
#: ECOSTRESS LST no-retrieval / fill sentinel (stored DN 0 carries no LST).
LST_FILL_VALUE = 0.0
#: Physical-plausibility band for land-surface temperature (Kelvin). Earth LST
#: spans roughly 200..360 K; values outside this are masked as invalid.
VALID_LST_RANGE = (200.0, 360.0)
#: Candidate LST variable names, in preference order (mirrors the LP DAAC
#: ECO2LSTE / SDS layout and common flattened-group names).
LST_VARIABLES = ("LST", "lst", "SDS/LST", "SDS_LST", "land_surface_temperature")
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("ecostress_lst")
class ECOSTRESSLSTConnector(BaseObservationConnector):
    slug = "ecostress_lst"
    display_name = "ECOSTRESS L2 Land Surface Temperature (ECO2LSTE)"
    kind = ObservationKind.LST
    structural_class = "gridded"
    base_url = "https://e4ftl01.cr.usgs.gov"
    auth = frozenset({"earthdata"})  # LP DAAC ECO2LSTE download needs Earthdata

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
                "ECOSTRESS LST live fetch needs a cached file (config 'nc_path'/'path') — "
                "an ECO2LSTE.001 L2 LST granule. LP DAAC/Earthdata download is not yet "
                "wired; the scale + fill-mask + reduce path is the proven part.",
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
        """Open an ECOSTRESS LST file, extract arrays, then scale/mask/reduce."""
        import numpy as np
        import xarray as xr

        # mask_and_scale=False so we apply the published DN*0.02 scale ourselves
        # at the canonical boundary rather than relying on file metadata.
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"file missing an ECOSTRESS LST variable (tried {LST_VARIABLES})",
                )
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            time_name = "time" if "time" in ds else _coord_like(ds, "time")
            # ECOSTRESS granules may be dim-ordered (lat, lon, time) while
            # reduce_grid / basin_mean require (time, lat, lon). Transpose by the
            # dataset's own dim names so any ordering is normalized first.
            da = _to_time_lat_lon(da, time_name, lat_name, lon_name)
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds[time_name].values)
            dn = np.asarray(da.values, dtype="float64")  # (time, lat, lon), stored DN
        return self.reduce_arrays(lats, lons, times, dn, spec, start, end, var_name=var_name)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        lst_dn: np.ndarray,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        var_name: str = "LST",
    ) -> ObservationSeries:
        """Mask fill/out-of-range, scale DN→K, basin-reduce, window-trim → canonical series.

        *lst_dn* is shaped ``(time, lat, lon)`` of stored ECOSTRESS LST digital
        numbers. Reproduces the published ECO2LSTE.001 spec: cells whose DN is the
        fill sentinel (DN ``0``), non-finite, or whose scaled value falls outside
        :data:`VALID_LST_RANGE` (Kelvin) are masked to NaN; the remaining counts
        become Kelvin via :data:`SOURCE_LST_SCALE` (``DN * 0.02``); the masked NaN
        cells reduce to :class:`~cos.core.models.QualityFlag.MISSING`.

        Coordinate shape is honoured: 1-D ``lat``/``lon`` vectors defer to
        :func:`cos.core.reduce.reduce_grid`; 2-D geolocation lat/lon (a real swath)
        take a dedicated bbox-mask reduction path.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        dn = np.asarray(lst_dn, dtype="float64")

        # Apply the source->canonical scale (DN * 0.02 -> K) at the boundary, then
        # mask: the fill DN (0), non-finite cells, and anything outside the physical
        # Kelvin band are not LST -> NaN -> MISSING. Masking on the scaled value
        # keeps the valid-range check in canonical units.
        lo, hi = VALID_LST_RANGE
        lst_k = dn * SOURCE_LST_SCALE
        invalid = (
            (dn == LST_FILL_VALUE)
            | ~np.isfinite(lst_k)
            | (lst_k < lo)
            | (lst_k > hi)
        )
        lst_k = np.where(invalid, np.nan, lst_k)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("ECOSTRESS LST basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("ECOSTRESS LST nearest_cell requires spec.centroid")

        if lats.ndim == 2 or lons.ndim == 2:
            # High-res swath with 2-D geolocation lat/lon. reduce_grid assumes 1-D
            # coord vectors (it indexes lat/lon axes independently), which IndexErrors
            # on a 2-D grid -> reduce over a bbox cell-mask instead.
            points = self._reduce_grid_2d(lats, lons, times, lst_k, reduction, bbox, point)
        else:
            points = reduce_grid(
                lats, lons, times, lst_k,
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
                "source": "ECOSTRESS L2 Land Surface Temperature",
                "product": "ECO2LSTE.001",
                "source_doi": "10.5067/ECOSTRESS/ECO2LSTE.001",
                "url": "https://lpdaac.usgs.gov/products/eco2lstev001/",
                "variable": var_name,
                "scale_k_per_count": f"{SOURCE_LST_SCALE:g}",
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
        """Reduce a 2-D-geolocation (swath) product to canonical points.

        ``lats``/``lons`` are 2-D (ny, nx); ``values`` is (time, ny, nx). Cells with
        non-finite geolocation drop out of the bbox mask. ``basin_mean`` is the
        cos-lat-weighted mean over the bbox cells; ``nearest_cell`` is the nearest
        valid in-grid cell to the centroid.
        """
        import numpy as np

        from cos.core.models import QualityFlag
        from cos.core.reduce import _as_datetime

        lats = np.broadcast_to(np.asarray(lats, dtype="float64"), values.shape[1:])
        lons = np.broadcast_to(np.asarray(lons, dtype="float64"), values.shape[1:])
        finite_coord = np.isfinite(lats) & np.isfinite(lons)

        if reduction == SpatialReduction.BASIN_MEAN:
            if bbox is None:
                raise ReductionError("ECOSTRESS LST basin_mean requires spec.bbox")
            lat_min, lon_min, lat_max, lon_max = bbox
            cell_mask = (
                finite_coord
                & (lats >= lat_min) & (lats <= lat_max)
                & (lons >= lon_min) & (lons <= lon_max)
            )
            if not cell_mask.any():
                raise ReductionError(
                    f"No swath cells inside bbox {bbox} on the 2-D geolocation grid"
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
                raise ReductionError("ECOSTRESS LST nearest_cell requires spec.centroid")
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

    def _find_variable(self, ds: object) -> str | None:
        """Pick the LST variable by published name, then by any LST-like name."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in LST_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            lower = str(name).lower()
            if "lst" in lower or "land_surface_temperature" in lower:
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
            site_id = f"ecostress_lst:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"ecostress_lst:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"ECOSTRESS LST over {spec.domain_name}",
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
    """Transpose an LST DataArray to ``(time, lat, lon)`` by its own dim names.

    A granule may ship ``(lat, lon, time)`` while
    :func:`cos.core.reduce.basin_mean`/``nearest_cell`` index ``(time, lat, lon)``.
    Reorder only the dims that exist (a 2-D single-time grid has no time dim),
    keeping any unexpected leading dims ahead of the canonical trailing axes.
    """
    dims = tuple(str(d) for d in da.dims)
    # Only reorder when lat AND lon are real dimensions of the array. A swath with
    # 2-D geolocation carries lat/lon as 2-D *coords* over other dims (e.g. y, x);
    # there is nothing to transpose and matching only on time would wrongly move
    # the scan axes behind time, so leave such arrays untouched.
    if lat_name not in dims or lon_name not in dims:
        return da
    wanted = [d for d in (time_name, lat_name, lon_name) if d in dims]
    if not wanted:
        return da
    leading = [d for d in dims if d not in wanted]
    order = leading + wanted
    if order == list(dims):
        return da
    return da.transpose(*order)
