# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""TROPOMI solar-induced fluorescence (SIF) connector (gridded, basin-reduced).

TROPOMI SIF has **no** SYMFLUENCE native handler, so this connector is
*spec-validated*: its scale, valid range, and fill semantics reproduce the
published Caltech gridded TROPOMI SIF product spec (Köhler et al., 2018; the
GES DISC / Caltech gridded monthly product), and the hermetic tests assert that
contract on a synthetic fixture rather than against a native reference series.

Product (Caltech gridded TROPOMI SIF, served as NetCDF behind NASA Earthdata):

* the gridded variable is the retrieved SIF radiance (``sif`` / ``SIF_743`` /
  ``sif_dc``), already reported in **mW/m²/nm/sr** — which is exactly the COS
  canonical ``sif`` unit (:data:`cos.core.models.KIND_UNITS`). The boundary scale
  is therefore the identity (:data:`SOURCE_SIF_SCALE` ``= 1.0``); a non-identity
  scale would be applied here and nowhere else.
* the no-retrieval fill is ``-999`` (:data:`SIF_FILL_VALUE`); cells equal to the
  fill, non-finite, or outside the physical valid band
  (:data:`VALID_SIF_RANGE`, mW/m²/nm/sr) are masked to NaN so they reduce to
  :class:`~cos.core.models.QualityFlag.MISSING`.

This connector:

1. opens a TROPOMI SIF NetCDF (a local cached file supplied via config
   ``nc_path`` / ``path`` — Earthdata download is not wired here, the reduce +
   canonicalize path is the proven part);
2. extracts ``lat / lon / time`` and the SIF variable as numpy arrays;
3. masks fill / out-of-range cells, applies the (identity) source→canonical
   scale at the boundary;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` (cos-lat
   weighted) for larger basins, ``nearest_cell`` for small ones — and emits the
   canonical ``sif`` unit ``mW/m2/nm/sr``.

The architecture-critical extract→mask→scale→reduce→canonicalize path is
hermetically tested via :meth:`TROPOMISIFConnector.reduce_arrays` on a synthetic
in-memory grid, with no network, no auth, and no NetCDF dependency.
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

#: Published Caltech TROPOMI SIF fill / no-retrieval sentinel.
SIF_FILL_VALUE = -999.0
#: Source→canonical scale. The product is already mW/m²/nm/sr (== canonical
#: ``sif`` unit), so the boundary conversion is the identity.
SOURCE_SIF_SCALE = 1.0
#: Physical-plausibility band for SIF radiance (mW/m²/nm/sr). TROPOMI 740 nm SIF
#: spans roughly 0..8; a small negative tail is a legitimate retrieval artefact,
#: so the lower bound allows mildly negative values while masking gross outliers.
VALID_SIF_RANGE = (-2.0, 12.0)
#: Candidate SIF variable names, in preference order (mirrors the Caltech /
#: GES DISC gridded distributions).
SIF_VARIABLES = ("sif", "SIF_743", "SIF", "sif_dc", "sif_743", "daily_corr")
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("tropomi_sif")
class TROPOMISIFConnector(BaseObservationConnector):
    slug = "tropomi_sif"
    display_name = "TROPOMI Solar-Induced Fluorescence (Caltech gridded)"
    kind = ObservationKind.SIF
    structural_class = "gridded"
    base_url = "https://oco2.gesdisc.eosdis.nasa.gov"
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
                "TROPOMI SIF live fetch needs a cached NetCDF (config 'nc_path'/'path') "
                "or Earthdata download (not yet wired). The reduce + canonicalize path "
                "is the proven part; supply a Caltech gridded TROPOMI SIF NetCDF.",
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
        """Open a TROPOMI SIF NetCDF, extract arrays, reduce + canonicalize."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a TROPOMI SIF variable (tried {SIF_VARIABLES})",
                )
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            time_name = "time" if "time" in ds else _coord_like(ds, "time")
            # The Caltech gridded product is dim-ordered (lat, lon, time), but
            # reduce_grid / basin_mean require (time, lat, lon). Transpose by the
            # dataset's own dim names so any ordering — (time, lat, lon) or
            # (lat, lon, time) — is normalized before reducing.
            da = _to_time_lat_lon(da, time_name, lat_name, lon_name)
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds[time_name].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
        return self.reduce_arrays(lats, lons, times, values, spec, start, end, var_name=var_name)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        sif: np.ndarray,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        var_name: str = "sif",
    ) -> ObservationSeries:
        """Mask fill/out-of-range, scale source→canonical, basin-reduce, window-trim.

        *sif* is shaped ``(time, lat, lon)`` SIF radiance in the source unit
        (mW/m²/nm/sr). Cells equal to :data:`SIF_FILL_VALUE`, non-finite, or
        outside :data:`VALID_SIF_RANGE` become NaN and surface as MISSING; the
        rest are multiplied by :data:`SOURCE_SIF_SCALE` (identity) so the canonical
        unit is preserved inside :func:`reduce_grid`.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        values = np.asarray(sif, dtype="float64")

        lo, hi = VALID_SIF_RANGE
        invalid = (
            (values == SIF_FILL_VALUE)
            | ~np.isfinite(values)
            | (values < lo)
            | (values > hi)
        )
        # Apply the source→canonical scale at the boundary, then mask. (Scale is
        # the identity here; written explicitly so a future non-identity product
        # has exactly one place to set it.)
        values = np.where(invalid, np.nan, values * SOURCE_SIF_SCALE)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("TROPOMI SIF basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("TROPOMI SIF nearest_cell requires spec.centroid")

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
                "source": "Caltech gridded TROPOMI SIF",
                "source_doi": "10.5067/MEASURES/SIF/DATA210",
                "url": "https://disc.gsfc.nasa.gov/datasets/TROPOMI_SIF",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    def _find_variable(self, ds: object) -> str | None:
        """Pick the SIF variable in the published preference order."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in SIF_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            if "sif" in str(name).lower():
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
            site_id = f"tropomi_sif:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"tropomi_sif:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"TROPOMI SIF over {spec.domain_name}",
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
    """Transpose a SIF DataArray to ``(time, lat, lon)`` by its own dim names.

    The Caltech gridded TROPOMI SIF product ships ``(lat, lon, time)`` while
    :func:`cos.core.reduce.basin_mean`/``nearest_cell`` index ``(time, lat, lon)``.
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
