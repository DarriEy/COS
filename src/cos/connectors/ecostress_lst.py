# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""ECOSTRESS land-surface-temperature (LST) connector (gridded, basin-reduced).

ECOSTRESS LST has **no** SYMFLUENCE native handler, so this connector is
*spec-validated*: its scale, fill semantics, valid range, and unit reproduce the
published LP DAAC ECOSTRESS L2 LST product specs and the hermetic tests assert
that contract on synthetic fixtures rather than against a native reference series.

Two on-disk layouts are read, both yielding the canonical ``lst`` unit ``K``:

* the **gridded geographic** product ``ECO_L2G_LSTE`` v002 — a regular-grid
  HDF-EOS5 file whose ``LST`` field lives in the nested GRID group
  ``HDFEOS/GRIDS/<grid>/Data Fields/LST`` and is **already float32 Kelvin**
  (``scale_factor = 1.0``, ``_FillValue = NaN``). Geolocation is *not* shipped as
  lat/lon variables: it is reconstructed from the grid corner bounds in
  ``StructMetadata.0`` (``UpperLeftPointMtrs`` / ``LowerRightMtrs``, microdegrees
  under the geographic projection) plus the ``XDim`` / ``YDim`` grid shape into
  1-D lat/lon cell-centre vectors (:func:`_geo_grid_from_struct_metadata`). This
  is the live, reducible product served behind NASA Earthdata.
* the **flat scaled-integer** layout (``ECO2LSTE.001`` / common flattened
  exports) — a NetCDF with 1-D lat/lon and ``LST`` stored as a scaled unsigned
  integer where canonical Kelvin is ``DN * scale_factor`` with the published
  ``scale_factor = 0.02`` (:data:`SOURCE_LST_SCALE`) and a ``DN 0`` no-retrieval
  fill (:data:`LST_FILL_VALUE`).

For either layout, cells that are the layout's fill, non-finite, or outside the
physical valid band (:data:`VALID_LST_RANGE`, Kelvin) are masked to NaN so they
reduce to :class:`~cos.core.models.QualityFlag.MISSING`. The source→Kelvin scale
is **product-aware**: native-float Kelvin uses scale ``1.0``, the scaled-integer
layout uses ``0.02``.

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

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

logger = structlog.get_logger()

#: Published ECO2LSTE.001 source→canonical scale: stored DN * 0.02 -> Kelvin.
SOURCE_LST_SCALE = 0.02
#: Native-float scale for ECO_L2G_LSTE (LST is already Kelvin, scale_factor=1.0).
NATIVE_LST_SCALE = 1.0
#: ECOSTRESS LST no-retrieval / fill sentinel (stored DN 0 carries no LST).
LST_FILL_VALUE = 0.0
#: HDF-EOS5 GRID path under which the gridded ECO_L2G_LSTE LST field lives.
HDFEOS_GRIDS_PREFIX = "HDFEOS/GRIDS"
#: LST data-field name inside an HDF-EOS5 GRID "Data Fields" group.
HDFEOS_LST_FIELD = "LST"
#: Physical-plausibility band for land-surface temperature (Kelvin). Earth LST
#: spans roughly 200..360 K; values outside this are masked as invalid.
VALID_LST_RANGE = (200.0, 360.0)
#: Candidate LST variable names, in preference order (mirrors the LP DAAC
#: ECO2LSTE / SDS layout and common flattened-group names).
LST_VARIABLES = ("LST", "lst", "SDS/LST", "SDS_LST", "land_surface_temperature")
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: Per-layout LP DAAC provenance for source_info: the flat scaled-integer product
#: (ECO2LSTE.001) vs the gridded HDF-EOS5 product (ECO_L2G_LSTE.002).
_ECO2LSTE_PROVENANCE = {
    "product": "ECO2LSTE.001",
    "source_doi": "10.5067/ECOSTRESS/ECO2LSTE.001",
    "url": "https://lpdaac.usgs.gov/products/eco2lstev001/",
}
_ECO_L2G_LSTE_PROVENANCE = {
    "product": "ECO_L2G_LSTE.002",
    "source_doi": "10.5067/ECOSTRESS/ECO_L2G_LSTE.002",
    "url": "https://lpdaac.usgs.gov/products/eco_l2g_lstev002/",
}


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
        """Open an ECOSTRESS LST file, extract arrays, then scale/mask/reduce.

        Routes by on-disk layout: the gridded ECO_L2G_LSTE HDF-EOS5 file exposes
        no top-level data variables (its LST lives in a nested GRID group), so it
        takes the HDF-EOS5 path (native-float Kelvin, geolocation rebuilt from
        StructMetadata); the flat scaled-integer layout takes the legacy path.
        """
        import numpy as np
        import xarray as xr

        # mask_and_scale=False so we apply the published DN*0.02 scale ourselves
        # at the canonical boundary rather than relying on file metadata.
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            if not getattr(ds, "data_vars", {}):
                grid_arrays = self._read_hdfeos_grid(path)
                if grid_arrays is not None:
                    lats, lons, times, lst_k, grid_var = grid_arrays
                    # ECO_L2G_LSTE is already Kelvin (scale 1.0); the only fill is
                    # NaN, so DN-0 masking is a no-op here -> pass scale=1.0.
                    return self.reduce_arrays(
                        lats, lons, times, lst_k, spec, start, end,
                        var_name=grid_var, source_scale=NATIVE_LST_SCALE,
                        provenance=_ECO_L2G_LSTE_PROVENANCE,
                    )
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

    def _read_hdfeos_grid(
        self, path: Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
        """Read LST + reconstructed geolocation from a gridded HDF-EOS5 file.

        The gridded ECO_L2G_LSTE product nests its LST under
        ``HDFEOS/GRIDS/<grid>/Data Fields/LST`` (already float32 Kelvin) and ships
        *no* lat/lon variables — geolocation is rebuilt from the GRID corner bounds
        in ``StructMetadata.0``. Returns ``(lats, lons, times, lst, var_name)`` with
        a singleton time axis (the granule's acquisition instant), or ``None`` when
        the file is not a recognizable HDF-EOS5 GRID layout (so the caller falls
        back to the flat scaled-integer path).
        """
        import h5py
        import numpy as np

        with h5py.File(path, "r") as h5:
            grids = h5.get(HDFEOS_GRIDS_PREFIX)
            if grids is None or not len(grids):
                return None
            grid_name = next(iter(grids))
            fields = grids[grid_name].get("Data Fields")
            if fields is None or HDFEOS_LST_FIELD not in fields:
                return None
            lst = np.asarray(fields[HDFEOS_LST_FIELD][()], dtype="float64")  # (ny, nx) K
            struct_meta = _read_struct_metadata(h5)
            lats, lons = _geo_grid_from_struct_metadata(struct_meta, lst.shape)
            times = _granule_time(h5)
        # reduce_grid wants (time, lat, lon); the granule is a single instant.
        return lats, lons, times, lst[np.newaxis, :, :], f"{grid_name}/{HDFEOS_LST_FIELD}"

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
        source_scale: float = SOURCE_LST_SCALE,
        provenance: dict[str, str] | None = None,
    ) -> ObservationSeries:
        """Mask fill/out-of-range, scale source→K, basin-reduce, window-trim → canonical series.

        *lst_dn* is shaped ``(time, lat, lon)`` of source ECOSTRESS LST values —
        stored digital numbers for the scaled-integer layout, or already-Kelvin
        floats for the gridded ECO_L2G_LSTE product. *source_scale* is the
        product-aware source→Kelvin factor (:data:`SOURCE_LST_SCALE` ``0.02`` for
        the scaled-integer DN, :data:`NATIVE_LST_SCALE` ``1.0`` for native Kelvin).
        Cells whose raw value is the DN-``0`` fill sentinel, non-finite, or whose
        scaled value falls outside :data:`VALID_LST_RANGE` (Kelvin) are masked to
        NaN and reduce to :class:`~cos.core.models.QualityFlag.MISSING`.

        Coordinate shape is honoured: 1-D ``lat``/``lon`` vectors defer to
        :func:`cos.core.reduce.reduce_grid`; 2-D geolocation lat/lon (a real swath)
        take a dedicated bbox-mask reduction path.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid, reduce_grid_2d

        prov = provenance or _ECO2LSTE_PROVENANCE
        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        dn = np.asarray(lst_dn, dtype="float64")

        # Apply the source->canonical scale (DN * 0.02 -> K) at the boundary, then
        # mask: the fill DN (0), non-finite cells, and anything outside the physical
        # Kelvin band are not LST -> NaN -> MISSING. Masking on the scaled value
        # keeps the valid-range check in canonical units.
        lo, hi = VALID_LST_RANGE
        lst_k = dn * source_scale
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
            points = reduce_grid_2d(
                lats, lons, times, lst_k,
                reduction=reduction, bbox=bbox, point=point, grid_label="ECOSTRESS LST",
            )
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
                "product": prov["product"],
                "source_doi": prov["source_doi"],
                "url": prov["url"],
                "variable": var_name,
                "scale_k_per_count": f"{source_scale:g}",
            },
            fetched_at=datetime.now(UTC),
        )


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


#: Microdegrees per degree: HE5_GCTP_GEO corner points are stored as degrees * 1e6.
MICRODEGREES_PER_DEGREE = 1.0e6


def _read_struct_metadata(h5: object) -> str:
    """Return the HDF-EOS5 ``StructMetadata.0`` text (decoded), or '' if absent."""
    node = h5.get("HDFEOS INFORMATION/StructMetadata.0")  # type: ignore[attr-defined]
    if node is None:
        return ""
    raw = node[()]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def _geo_grid_from_struct_metadata(
    struct_meta: str, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct 1-D cell-centre lat/lon vectors from GRID corner bounds.

    ECO_L2G_LSTE is a regular geographic grid (``Projection=HE5_GCTP_GEO``) whose
    extent is given in ``StructMetadata.0`` by ``UpperLeftPointMtrs=(x_ul, y_ul)``
    and ``LowerRightMtrs=(x_lr, y_lr)`` — *(lon, lat)* corners in **microdegrees**
    under the geographic projection. With the array shape ``(ny, nx)`` the cell
    size is the corner span over the dim count; vectors are cell **centres**
    (corner + half a cell), lat descending from the upper-left as the rows do.
    """
    import numpy as np

    ny, nx = int(shape[0]), int(shape[1])
    ul = _struct_point(struct_meta, "UpperLeftPointMtrs")
    lr = _struct_point(struct_meta, "LowerRightMtrs")
    if ul is None or lr is None:
        raise ConnectorError(
            "ecostress_lst",
            "HDF-EOS5 GRID StructMetadata missing UpperLeftPointMtrs / LowerRightMtrs "
            "corner bounds; cannot reconstruct geolocation",
        )
    x_ul, y_ul = ul[0] / MICRODEGREES_PER_DEGREE, ul[1] / MICRODEGREES_PER_DEGREE
    x_lr, y_lr = lr[0] / MICRODEGREES_PER_DEGREE, lr[1] / MICRODEGREES_PER_DEGREE
    dx = (x_lr - x_ul) / nx
    dy = (y_lr - y_ul) / ny  # negative: lat decreases down the rows
    lons = x_ul + (np.arange(nx, dtype="float64") + 0.5) * dx
    lats = y_ul + (np.arange(ny, dtype="float64") + 0.5) * dy
    return lats, lons


def _struct_point(struct_meta: str, key: str) -> tuple[float, float] | None:
    """Parse ``key=(a,b)`` from StructMetadata text into a float pair."""
    match = re.search(rf"{re.escape(key)}=\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)", struct_meta)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _granule_time(h5: object) -> np.ndarray:
    """The granule acquisition instant from StandardMetadata, as a singleton array.

    Falls back to a NaT-free epoch only if the metadata is unreadable; the gridded
    granule carries exactly one acquisition, so the reduced series has one point.
    """
    import numpy as np

    base = "HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata/"
    date = _h5_text(h5, base + "RangeBeginningDate")
    clock = _h5_text(h5, base + "RangeBeginningTime")
    stamp = f"{date}T{clock}" if date else ""
    try:
        when = np.datetime64(stamp[:26]) if stamp else np.datetime64("1970-01-01")
    except (ValueError, TypeError):
        when = np.datetime64("1970-01-01")
    return np.array([when], dtype="datetime64[ns]")


def _h5_text(h5: object, key: str) -> str:
    """Read a scalar HDF5 string dataset as decoded text, or '' if absent."""
    import numpy as np

    node = h5.get(key)  # type: ignore[attr-defined]
    if node is None:
        return ""
    raw = node[()]
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0] if raw.size else b""
    return raw.decode() if isinstance(raw, bytes) else str(raw)
