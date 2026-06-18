# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GSFC (NASA Goddard) GRACE/GRACE-FO mascon TWS connector (gridded, basin-reduced).

A sibling of :mod:`cos.connectors.grace` specialised to the **GSFC mascon**
product line (``earth.gsfc.nasa.gov`` / PO.DAAC; the half-degree
``gsfc.glb_.*_rl06v2.0_obp-ice6gd_halfdegree.nc`` granule). The SYMFLUENCE native
``grace`` handler processes the GSFC mascon file through *exactly the same*
extract → cm→mm → 2003-2008-anomaly path it uses for the JPL/CSR mascons (see
``data/observation/handlers/grace.py``: ``gsfc`` is one of its three recognised
products). This connector is therefore **native-parity**: its reduction, unit
conversion, and anomaly baseline reproduce that handler's GSFC branch.

Product / access (GSFC mascon, served as NetCDF):

* the gridded variable is liquid-water-equivalent thickness (``lwe_thickness``)
  in **cm**, on a 0.5° global grid (``lat`` / ``lon`` 1-D vectors, monthly
  ``time``). The GSFC mascon carries a land mask, so off-land / no-data cells are
  fill / non-finite; those mask to NaN and surface as
  :class:`~cos.core.models.QualityFlag.MISSING`.

This connector:

1. opens a GSFC mascon NetCDF (a local cached file supplied via config
   ``nc_path`` / ``path`` — Earthdata/PO.DAAC download is not wired here, the
   reduce + canonicalize path is the proven part);
2. extracts ``lat / lon / time / lwe_thickness`` as numpy arrays, normalising the
   dim order to ``(time, lat, lon)`` so a granule shipped ``(lat, lon, time)``
   reduces correctly;
3. masks the fill value / non-finite cells, converts **cm → mm** (canonical
   ``tws`` unit) at the boundary, reduces to the basin via
   :mod:`cos.core.reduce` (``basin_mean`` cos-lat weighted for larger basins,
   ``nearest_cell`` for small ones), window-trims half-open UTC ``[start, end)``;
4. subtracts the 2003-2008 anomaly baseline mean, matching the native handler.

The architecture-critical extract→mask→scale→reduce→anomaly→canonicalize path is
hermetically tested via :meth:`GSFCGRACEConnector.reduce_arrays` on a synthetic
in-memory grid, with no network and no auth.
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

#: GSFC mascon ``lwe_thickness`` is centimetres; canonical ``tws`` unit is mm.
CM_TO_MM = 10.0
#: Source→canonical boundary scale (cm → mm). The single place the GSFC unit is
#: converted; equals :data:`CM_TO_MM`.
SOURCE_TWS_SCALE = CM_TO_MM
#: Anomaly baseline window, matching the native SYMFLUENCE ``grace`` handler.
DEFAULT_BASELINE = ("2003-01-01", "2008-12-31")
#: Fill sentinel sometimes carried on the GSFC mascon land-masked arrays.
FILL_VALUE = -99999.0
#: Candidate TWS variable names, in preference order (GSFC mascon layout first).
TWS_VARIABLES = ("lwe_thickness", "lwe", "cmwe", "tws")
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean — the
#: native handler's medium-basin threshold.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("gsfc_grace")
class GSFCGRACEConnector(BaseObservationConnector):
    slug = "gsfc_grace"
    display_name = "NASA GSFC GRACE/GRACE-FO Mascon TWS"
    kind = ObservationKind.TWS
    structural_class = "gridded"
    base_url = "https://earth.gsfc.nasa.gov"
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
        path = self.config.get("nc_path") or self.config.get("path")
        if not path:
            raise ConnectorError(
                self.slug,
                "GSFC GRACE live fetch needs a cached NetCDF (config 'nc_path'/'path') "
                "or Earthdata/PO.DAAC download (not yet wired). The reduce + canonicalize "
                "path is the proven part; supply a GSFC mascon TWS NetCDF to reduce it.",
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
        """Open a GSFC mascon NetCDF, extract arrays, reduce + canonicalize."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a GSFC mascon TWS variable (tried {TWS_VARIABLES})",
                )
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            time_name = "time" if "time" in ds else _coord_like(ds, "time")
            # The GSFC mascon grid is normally (time, lat, lon), but some granule
            # builds ship (lat, lon, time); reduce_grid / basin_mean require
            # (time, lat, lon). Transpose by the dataset's own dim names so any
            # ordering is normalized before reducing.
            da = _to_time_lat_lon(da, time_name, lat_name, lon_name)
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds[time_name].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon), cm
        return self.reduce_arrays(lats, lons, times, values, spec, start, end, var_name=var_name)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        lwe: np.ndarray,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        var_name: str = "lwe_thickness",
    ) -> ObservationSeries:
        """Mask fill, scale cm→mm, basin-reduce, window-trim, subtract anomaly baseline.

        *lwe* is shaped ``(time, lat, lon)`` liquid-water-equivalent thickness in
        the source unit (cm). Cells equal to :data:`FILL_VALUE` or non-finite
        become NaN and surface as MISSING; the rest are multiplied by
        :data:`SOURCE_TWS_SCALE` (cm → mm) so :func:`reduce_grid` works in the
        canonical unit. The 2003-2008 baseline-window mean is then subtracted to
        make a TWS anomaly, matching the native handler.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        values = np.asarray(lwe, dtype="float64")

        invalid = (values == FILL_VALUE) | ~np.isfinite(values)
        # Convert cm → mm at the boundary, then mask. (Scale is linear, so applying
        # it pre-reduction is identical to post-reduction and keeps the canonical
        # unit; this is the single place the GSFC source unit is converted.)
        values = np.where(invalid, np.nan, values * SOURCE_TWS_SCALE)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("GSFC GRACE basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("GSFC GRACE nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)) then anomaly baseline.
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= _utc(p.timestamp) < end_u]
        points = self._apply_baseline(points, spec)

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "GSFC GRACE/GRACE-FO mascon",
                "source_doi": "10.5067/TEMSC-3JC62",
                "url": "https://earth.gsfc.nasa.gov/geo/data/grace-mascons",
                "variable": var_name,
                "baseline": "-".join(spec.options.get("baseline", DEFAULT_BASELINE)),
            },
            fetched_at=datetime.now(UTC),
        )

    def _apply_baseline(
        self, points: list[ObservationPoint], spec: ReductionSpec
    ) -> list[ObservationPoint]:
        """Subtract the baseline-window mean to make a TWS anomaly (mm)."""
        b_start, b_end = spec.options.get("baseline", DEFAULT_BASELINE)
        b0 = _utc(datetime.fromisoformat(b_start))
        b1 = _utc(datetime.fromisoformat(b_end))
        vals = [
            p.value
            for p in points
            if p.value is not None and b0 <= _utc(p.timestamp) <= b1
        ]
        if not vals:
            vals = [p.value for p in points if p.value is not None]
        if not vals:
            return points
        mean = sum(vals) / len(vals)
        for p in points:
            if p.value is not None:
                p.value = p.value - mean
        return points

    def _find_variable(self, ds: object) -> str | None:
        """Pick the TWS variable in the GSFC mascon preference order."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in TWS_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            lower = str(name).lower()
            if "lwe" in lower or "thickness" in lower or "cmwe" in lower:
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
            site_id = f"gsfc_grace:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"gsfc_grace:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"GSFC GRACE TWS over {spec.domain_name}",
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
    """Transpose a GSFC mascon DataArray to ``(time, lat, lon)`` by its own dim names.

    :func:`cos.core.reduce.basin_mean` / ``nearest_cell`` index ``(time, lat, lon)``;
    a granule shipped ``(lat, lon, time)`` must be reordered first. Reorder only
    the dims that exist (a 2-D single-time grid has no time dim), keeping any
    unexpected leading dims ahead of the canonical trailing axes.
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
