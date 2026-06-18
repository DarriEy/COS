# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NASA MODIS 16-day NDVI / EVI vegetation-index connector (gridded, basin-reduced).

A **multivariate-breadth** connector: the vegetation index is an orthogonal
constraint on the water-energy-carbon cycle (greenness / phenology) for
multi-objective model evaluation. It exercises the **gridded spatial-reduction
path** of the canonical contract for a MODIS land product.

MODIS source (LP DAAC, Earthdata):
    * MOD13A2 (1 km) / MOD13Q1 (250 m) 16-day vegetation indices — NDVI and EVI
      composites on a sinusoidal grid, distributed as HDF-EOS (and re-served as
      NetCDF subsets).
    * The NDVI / EVI scientific datasets are stored as **scaled 16-bit integers**:
          - scale factor ``0.0001`` (physical NDVI = DN * 0.0001, ratio in ~-1..1);
          - valid range ``-2000 .. 10000`` (DN), i.e. -0.2 .. 1.0 physical;
          - fill value ``-3000`` (DN).
      The native LP DAAC product spec is reproduced here: mask the fill value and
      any DN outside the valid range to NaN (→ MISSING), then apply the scale
      factor so the canonical series is the dimensionless ratio.

This connector extracts ``lats, lons, times, ndvi_dn`` from a supplied HDF /
NetCDF (config ``nc_path`` / ``path`` — live LP DAAC Earthdata download via netrc
is not yet wired; the reduce + scale + mask path is the proven part) and reduces
it with :func:`cos.core.reduce.reduce_grid` — ``basin_mean`` (cos-lat weighted)
for larger basins, ``nearest_cell`` for small ones.

There is no SYMFLUENCE native vegetation-index handler, so correctness is
**spec-validated**: the architecture-critical extract→mask→scale→reduce→
canonicalize path is hermetically tested against the published MOD13 product spec
(scale ``0.0001``, valid range ``-2000..10000``, fill ``-3000``) on a synthetic
in-memory grid — no network, no auth, no HDF dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

logger = structlog.get_logger()

#: MOD13 scaled-integer scale factor: physical NDVI/EVI = DN * SCALE_FACTOR.
SCALE_FACTOR = 0.0001
#: MOD13 valid DN range (inclusive); DN outside this is masked to NaN.
VALID_MIN_DN = -2000.0
VALID_MAX_DN = 10000.0
#: MOD13 fill value (DN) → NaN → QualityFlag.MISSING.
FILL_VALUE_DN = -3000.0
#: Candidate vegetation-index variable names, NDVI preferred then EVI.
NDVI_VARIABLES = (
    "NDVI",
    "ndvi",
    "_1_km_16_days_NDVI",
    "250m_16_days_NDVI",
    "EVI",
    "evi",
    "_1_km_16_days_EVI",
    "250m_16_days_EVI",
)
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("modis_ndvi")
class MODISNDVIConnector(BaseObservationConnector):
    slug = "modis_ndvi"
    display_name = "NASA MODIS 16-day NDVI/EVI (MOD13A2/MOD13Q1)"
    kind = ObservationKind.VEGETATION_INDEX
    structural_class = "gridded"
    base_url = "https://e4ftl01.cr.usgs.gov"  # LP DAAC archive
    auth = frozenset({"earthdata"})  # LP DAAC download needs an Earthdata netrc

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
                "MODIS NDVI live fetch needs a cached file (config 'nc_path'/'path') — "
                "a MOD13A2/MOD13Q1 HDF or NetCDF subset. LP DAAC Earthdata (netrc) "
                "download is not yet wired; the reduce + scale + mask path is the "
                "proven part. Supply a downloaded MOD13 file to reduce it.",
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
        """Open a MOD13 HDF/NetCDF, extract DN arrays, reduce + canonicalize."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"File missing a MODIS NDVI/EVI variable (tried {NDVI_VARIABLES})",
                )
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            ndvi_dn = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
        return self.reduce_arrays(lats, lons, times, ndvi_dn, spec, start, end)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: Any,
        lons: Any,
        times: Any,
        ndvi_dn: Any,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Mask fill/out-of-range, scale DN→ratio, basin-reduce, window-trim.

        *ndvi_dn* is shaped ``(time, lat, lon)`` scaled-integer DN. Reproduces the
        published MOD13 product spec: mask the fill value (``-3000``) and any DN
        outside the valid range (``-2000..10000``) to NaN (→ MISSING), then apply
        the scale factor (``DN * 0.0001``) at the boundary so the canonical series
        is the dimensionless NDVI/EVI ratio.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        dn = np.asarray(ndvi_dn, dtype="float64")

        # Product-spec mask: fill + out-of-valid-range → NaN (→ MISSING).
        invalid = (
            (dn == FILL_VALUE_DN)
            | ~np.isfinite(dn)
            | (dn < VALID_MIN_DN)
            | (dn > VALID_MAX_DN)
        )
        # Apply the documented scale factor at the boundary; masked cells are NaN.
        ratio = np.where(invalid, np.nan, dn * SCALE_FACTOR)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MODIS NDVI basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MODIS NDVI nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, ratio,  # already the dimensionless ratio
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
                "source": "NASA MODIS 16-day Vegetation Indices",
                "product": "MOD13A2/MOD13Q1",
                "source_doi": "10.5067/MODIS/MOD13A2.061",
                "url": "https://lpdaac.usgs.gov/products/mod13a2v061/",
                "scale_factor": f"{SCALE_FACTOR:g}",
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _trim(
        points: list[ObservationPoint], start_u: datetime, end_u: datetime,
    ) -> list[ObservationPoint]:
        return [p for p in points if start_u <= _utc(p.timestamp) < end_u]

    def _find_variable(self, ds: Any) -> str | None:
        """Pick the NDVI/EVI variable, NDVI preferred (spec order)."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in NDVI_VARIABLES:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises NDVI/EVI.
        for name in data_vars:
            lower = str(name).lower()
            if "ndvi" in lower or "evi" in lower:
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
            site_id = f"modis_ndvi:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"modis_ndvi:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MODIS NDVI over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _coord_like(ds: Any, want: str) -> str:
    for name in ds.coords:
        if want in str(name).lower():
            return str(name)
    return want
