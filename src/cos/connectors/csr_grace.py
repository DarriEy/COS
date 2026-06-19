# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""CSR (UTexas) GRACE / GRACE-FO mascon total water storage connector.

A second NASA-mascon proof of the **gridded spatial-reduction path**, sourced
from the **CSR RL06 mascon** solution (Center for Space Research, University of
Texas at Austin), the ``CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc``
product served anonymously from ``download.csr.utexas.edu`` (and mirrored via
PO.DAAC). The P3 study evaluated three mascon centres — JPL, **CSR**, GSFC — and
the SYMFLUENCE native ``grace`` handler processes all three identically; this
connector reimplements the native handler's **CSR reduction** for native parity.

Like the JPL mascon (:mod:`cos.connectors.grace`), the CSR mascon is a global
monthly liquid-water-equivalent-thickness grid (variable ``lwe_thickness``, cm)
served as NetCDF on a 0-360 longitude grid. This connector:

1. opens the CSR mascon NetCDF (a local cached file supplied via config
   ``nc_path`` / ``path``, or live-fetched from the open CSR host when none is
   supplied — :meth:`_live_fetch` downloads :data:`LIVE_URL` to the cache);
2. extracts ``lat / lon / time / lwe_thickness`` as numpy arrays, normalizing the
   dimension order to ``(time, lat, lon)`` (the CSR mascon ships ``(time, lat,
   lon)`` but the reorder is defensive against ``(lat, lon, time)`` real-data
   orderings) and decoding the CDF ``days since`` time axis the native handler
   tolerates;
3. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` (cos-lat
   weighted) for larger basins, ``nearest_cell`` for small ones (the basin-size
   policy the native ``grace.py`` handler uses, made explicit here);
4. converts cm -> **mm** (the canonical ``tws`` unit) at the boundary and
   subtracts the anomaly baseline mean (default 2003-2008, matching native).

The architecture-critical extract->reduce->canonicalize core is hermetically
tested via :meth:`CSRGRACEConnector.reduce_arrays` on a synthetic in-memory grid,
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

#: CSR mascon liquid-water-equivalent thickness is reported in cm; canonical
#: ``tws`` is mm. The boundary conversion is the only scale this connector applies.
CM_TO_MM = 10.0
#: Anomaly baseline window, matching the native handler's 2003-2008 re-reference.
DEFAULT_BASELINE = ("2003-01-01", "2008-12-31")
#: <= this area (km²) defaults to point sampling, mirroring native grace.py.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: The CSR mascon thickness variable (same name as the JPL mascon).
VARIABLE = "lwe_thickness"


@register("csr_grace")
class CSRGRACEConnector(BaseObservationConnector):
    slug = "csr_grace"
    display_name = "CSR (UTexas) GRACE/GRACE-FO RL06 Mascon TWS"
    kind = ObservationKind.TWS
    structural_class = "gridded"
    base_url = "https://download.csr.utexas.edu"
    auth = frozenset()  # CSR mascon host is public (no Earthdata auth required)

    VARIABLE = VARIABLE

    #: Open CSR RL0603 mascon (all-corrections) — anonymous, no Earthdata auth.
    #: Override via config ``live_url`` when a newer release supersedes it.
    LIVE_URL = (
        "https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/"
        "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"
    )

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
        nc_path = self.config.get("nc_path") or self.config.get("path") or self._live_fetch()
        return [self.reduce_file(Path(nc_path), spec, start, end)]

    def _live_fetch(self) -> str:
        """Download the open CSR mascon (single global file) to the cache, return its path.

        The CSR RL0603 mascon is one anonymous NetCDF covering the full record, so
        the basin reduction + window-trim happen entirely from this one file — no
        per-window granule search. Cached, so repeated reductions fetch once.
        """
        from cos.core.fetch import cache_dir, http_download

        url = str(self.config.get("live_url") or self.LIVE_URL)
        dest = cache_dir(self.config) / url.rsplit("/", 1)[-1]
        return str(http_download(url, dest, slug=self.slug))

    # -- file reader (extract arrays, then defer to the pure core) -----------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open a CSR mascon NetCDF, extract arrays, reduce + canonicalize to mm anomaly."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            if self.VARIABLE not in ds:
                raise ConnectorError(self.slug, f"NetCDF missing '{self.VARIABLE}' variable")
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            time_name = "time" if "time" in ds else _coord_like(ds, "time")
            # The CSR mascon ships (time, lat, lon), but reduce_grid / basin_mean
            # require (time, lat, lon); transpose by the dataset's own dim names so
            # a (lat, lon, time) real-data ordering is normalized before reducing.
            da = _to_time_lat_lon(ds[self.VARIABLE], time_name, lat_name, lon_name)
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = _decode_times(ds[time_name])
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
        return self.reduce_arrays(lats, lons, times, values, spec, start, end)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        lwe_cm: np.ndarray,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Convert cm->mm, basin-reduce, window-trim, anomaly-baseline -> canonical series.

        *lwe_cm* is shaped ``(time, lat, lon)`` liquid-water-equivalent thickness
        in the source unit (cm). The cm->mm conversion is applied at the boundary
        (linear, so pre- vs post-reduction order is identical) so the canonical
        ``tws`` unit (mm) is preserved inside :func:`reduce_grid`; non-finite cells
        surface as :class:`~cos.core.models.QualityFlag.MISSING`.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        values = np.asarray(lwe_cm, dtype="float64")

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("CSR GRACE basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("CSR GRACE nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values * CM_TO_MM,  # cm -> mm at the boundary
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
                "source": "CSR GRACE/GRACE-FO RL06 Mascon",
                "center": "CSR (Center for Space Research, UT Austin)",
                "product": "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections",
                "url": "https://www2.csr.utexas.edu/grace/RL06_mascons.html",
                "baseline": "-".join(spec.options.get("baseline", DEFAULT_BASELINE)),
            },
            fetched_at=datetime.now(UTC),
        )

    def _apply_baseline(self, points: list, spec: ReductionSpec) -> list:
        """Subtract the baseline-window mean to make a TWS anomaly (mm)."""
        b_start, b_end = spec.options.get("baseline", DEFAULT_BASELINE)
        b0 = _utc(datetime.fromisoformat(b_start))
        b1 = _utc(datetime.fromisoformat(b_end))
        vals = [p.value for p in points if p.value is not None and b0 <= _utc(p.timestamp) <= b1]
        if not vals:
            vals = [p.value for p in points if p.value is not None]
        if not vals:
            return points
        mean = sum(vals) / len(vals)
        for p in points:
            if p.value is not None:
                p.value = p.value - mean
        return points

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"csr_grace:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"csr_grace:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"CSR GRACE TWS over {spec.domain_name}",
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
    """Transpose an ``lwe_thickness`` DataArray to ``(time, lat, lon)`` by dim name.

    The CSR mascon ships ``(time, lat, lon)`` while
    :func:`cos.core.reduce.basin_mean`/``nearest_cell`` index ``(time, lat, lon)``.
    Reorder only the dims that exist (a 2-D single-time grid has no time dim),
    keeping any unexpected leading dims ahead of the canonical trailing axes so a
    ``(lat, lon, time)`` real-data ordering is normalized before reducing.
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


def _decode_times(time_da: xr.DataArray) -> np.ndarray:
    """Return a numpy time array, decoding a CDF ``days since`` axis if needed.

    xarray usually decodes the CSR mascon time axis to ``datetime64`` already.
    When the file carries an undecoded numeric axis with a ``units`` of
    ``days since <origin>`` (the convention the native handler tolerates), convert
    it to ``datetime64[ns]`` here so :func:`cos.core.reduce.reduce_grid` gets real
    timestamps rather than raw day offsets.
    """
    import numpy as np
    import pandas as pd

    values = np.asarray(time_da.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values

    units = None
    for key, val in getattr(time_da, "attrs", {}).items():
        if str(key).lower() == "units":
            units = str(val)
            break
    if units and "days since" in units.lower():
        origin = units.lower().split("since", 1)[1].strip()
        if "t" in origin:
            origin = origin.split("t", 1)[0]
        decoded = pd.to_datetime(values, unit="D", origin=origin.strip())
        return np.asarray(decoded.values)
    return np.asarray(pd.to_datetime(values).values)
