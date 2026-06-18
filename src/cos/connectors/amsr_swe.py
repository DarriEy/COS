# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""AMSR2 passive-microwave gridded SWE connector (gridded, basin-reduced).

Fills the gridded-SWE gap in COS: today's SWE coverage is mostly point /
North-American (SNOTEL, CMC). AMSR2 (Advanced Microwave Scanning Radiometer 2,
aboard GCOM-W1) provides a *global* daily passive-microwave Snow Water Equivalent
retrieval — the NSIDC ``AU_DySno`` product family (AMSR2 Unified L3 Daily Snow
Water Equivalent, ~25 km EASE-Grid). This is a **spec-validated** connector: there
is no SYMFLUENCE native handler for AMSR2 SWE, so the offline tests reproduce the
*published product spec* (scale factor, valid range, flag/fill sentinels, unit)
on a synthetic inline fixture rather than asserting native parity.

Published spec reproduced here (AU_DySno / AMSR2 daily SWE):
    * SWE is distributed as a scaled-integer grid where the stored digital number
      (DN) becomes mm of SWE via the variable's own ``scale_factor``. The scale is
      **hemisphere-dependent**: the real granule metadata carries
      ``scale_factor=1.0`` on ``SWE_NorthernDaily`` ("0-240 SWE mm") and
      ``scale_factor=2.0`` on ``SWE_SouthernDaily`` (DN/2). The connector reads the
      per-variable ``scale_factor`` from metadata rather than hardcoding ``*2``.
    * Valid SWE DN range is ``0..240``; DN values ``241..255`` are reserved
      **flag/fill sentinels** (open water, mountainous mask, permanent snow/ice,
      off-Earth, etc.) and carry no SWE.
    * Any flag/fill cell, the NetCDF ``_FillValue``, off-Earth ``inf`` fills, and
      non-finite values are masked to NaN → they reduce to
      :class:`~cos.core.models.QualityFlag.MISSING`.

The real AU_DySno EASE-Grid product stores **2-D** lat/lon (721x721) with ``inf``
off-Earth fills, so :meth:`reduce_arrays` carries a dedicated 2-D-coordinate
reduction path (bbox mask + cos-lat-weighted mean / nearest valid cell) alongside
the 1-D :func:`cos.core.reduce.reduce_grid` path.

This connector extracts ``lats, lons, times, swe_dn`` from a supplied file
(config ``nc_path`` / ``path`` — live NSIDC/Earthdata download is not yet wired,
exactly as the other gridded connectors), applies the scale + flag/fill mask at
the boundary so the canonical series is **mm** w.e., then reduces to the basin via
:func:`cos.core.reduce.reduce_grid` — ``basin_mean`` (cos-lat weighted) for larger
basins, ``nearest_cell`` for small ones.

The architecture-critical extract→mask→scale→reduce→canonicalize path is
hermetically tested via :meth:`AMSR2SWEConnector.reduce_arrays` on a synthetic
in-memory grid, with no network and no auth.
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
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

#: Fallback AU_DySno scale when a variable carries no ``scale_factor`` metadata.
#: The published default count->mm scale is ``DN * SOURCE_SWE_SCALE``.
SOURCE_SWE_SCALE = 2.0
#: Per-variable fallback scales (mm per count) by published variable name. The
#: real granule metadata shows the Northern daily layer is already in mm
#: (scale 1.0, "0-240 SWE mm") while the Southern daily layer is DN/2 (scale 2.0).
HEMISPHERE_SCALE = {
    "SWE_NorthernDaily": 1.0,
    "SWE_SouthernDaily": 2.0,
}
#: Largest valid SWE digital number; DN above this is a flag/fill sentinel.
MAX_VALID_DN = 240.0
#: Maximum physically meaningful SWE (mm w.e.) implied by the valid DN range.
MAX_VALID_SWE_MM = MAX_VALID_DN * SOURCE_SWE_SCALE  # 480 mm
#: NetCDF fill sentinel commonly carried on the AMSR2 SWE arrays.
FILL_VALUE = -9999.0
#: Candidate SWE variable names, in preference order (matches AU_DySno layout).
SWE_VARIABLES = ("SWE", "swe", "snow_water_equivalent", "SWE_NorthernDaily", "SWE_SouthernDaily")
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("amsr_swe")
class AMSR2SWEConnector(BaseObservationConnector):
    slug = "amsr_swe"
    display_name = "AMSR2 Daily Snow Water Equivalent (AU_DySno)"
    kind = ObservationKind.SWE
    structural_class = "gridded"
    base_url = "https://n5eil01u.ecs.nsidc.org"
    auth = frozenset({"earthdata"})  # NSIDC AU_DySno download needs Earthdata

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
                "AMSR2 SWE live fetch needs a cached file (config 'nc_path'/'path') — an "
                "AU_DySno daily SWE NetCDF/HDF. NSIDC Earthdata download is not yet wired; "
                "the scale + flag-mask + reduce path is the proven part.",
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
        """Open an AMSR2 SWE NetCDF, extract arrays, then scale/mask/reduce."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path, mask_and_scale=False) as ds:
            var = self._find_variable(ds)
            if var is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing an AMSR2 SWE variable (tried {SWE_VARIABLES})",
                )
            da = ds[var]
            scale = self._resolve_scale(var, da.attrs)
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            swe_dn = np.asarray(da.values, dtype="float64")  # (time, lat, lon), stored DN
        return self.reduce_arrays(lats, lons, times, swe_dn, spec, start, end, scale=scale)

    @staticmethod
    def _resolve_scale(var_name: str, attrs: dict) -> float:
        """Per-variable mm-per-count scale.

        Prefer the variable's own ``scale_factor`` metadata (the real AU_DySno
        granule carries 1.0 on the Northern daily layer, 2.0 on the Southern),
        then the published hemisphere fallback, then the global default.
        """
        import numpy as np

        raw = attrs.get("scale_factor")
        if raw is not None:
            try:
                scale = float(np.asarray(raw).reshape(-1)[0]) if hasattr(raw, "__len__") else float(raw)
            except (TypeError, ValueError):
                scale = 0.0
            if scale > 0.0:
                return scale
        return HEMISPHERE_SCALE.get(var_name, SOURCE_SWE_SCALE)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats,
        lons,
        times,
        swe_dn,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        scale: float = SOURCE_SWE_SCALE,
    ) -> ObservationSeries:
        """Mask flags/fill, scale DN→mm, basin-reduce, window-trim → canonical series.

        *swe_dn* is shaped ``(time, lat, lon)`` of stored digital numbers. Reproduces
        the published AU_DySno spec exactly: cells whose DN is a flag/fill sentinel
        (``DN > 240``), the NetCDF fill value, off-Earth ``inf``, or non-finite are
        masked to NaN; the remaining counts become millimetres via the per-variable
        *scale* (mm per count); the masked NaN cells reduce to
        :class:`~cos.core.models.QualityFlag.MISSING`.

        Coordinate shape is honoured: 1-D ``lat``/``lon`` vectors defer to
        :func:`cos.core.reduce.reduce_grid`; 2-D EASE-Grid lat/lon (the real
        product) take a dedicated bbox-mask reduction path.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid, reduce_grid_2d

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        dn = np.asarray(swe_dn, dtype="float64")

        # Published spec mask: flag/fill sentinels (DN 241..255), the NetCDF fill
        # value, negative DN, and non-finite (incl. off-Earth inf) cells are not
        # SWE -> NaN.
        invalid = (
            (dn == FILL_VALUE)
            | ~np.isfinite(dn)
            | (dn < 0.0)
            | (dn > MAX_VALID_DN)
        )
        dn = np.where(invalid, np.nan, dn)

        # Convert stored count -> mm w.e. at the boundary (linear, so applying it
        # pre-reduction is identical to the post-reduction order and keeps the
        # canonical unit (mm)). *scale* is the per-variable mm-per-count factor
        # (hemisphere-dependent on the real AU_DySno product).
        swe_mm = dn * scale

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("AMSR2 SWE basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("AMSR2 SWE nearest_cell requires spec.centroid")

        if lats.ndim == 2 or lons.ndim == 2:
            # Real EASE-Grid product: 2-D lat/lon. reduce_grid assumes 1-D coord
            # vectors (it indexes lat/lon axes independently), which IndexErrors
            # on a (721,721) grid -> reduce over a bbox cell-mask instead.
            points = reduce_grid_2d(
                lats, lons, times, swe_mm,
                reduction=reduction, bbox=bbox, point=point, grid_label="EASE-Grid",
            )
        else:
            points = reduce_grid(
                lats, lons, times, swe_mm,
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
                "source": "AMSR2 Unified L3 Daily Snow Water Equivalent",
                "product": "AU_DySno",
                "url": "https://nsidc.org/data/au_dysno",
                "scale_mm_per_count": f"{scale:g}",
            },
            fetched_at=datetime.now(UTC),
        )


    @staticmethod
    def _trim(points: list[ObservationPoint], start_u: datetime, end_u: datetime) -> list[ObservationPoint]:
        return [p for p in points if start_u <= _utc(p.timestamp) < end_u]

    def _find_variable(self, ds: object) -> str | None:
        """Pick the SWE variable by published name, then by any SWE-like name."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in SWE_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            lower = name.lower()
            if "swe" in lower or "snow_water_equivalent" in lower:
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
            site_id = f"amsr_swe:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"amsr_swe:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"AMSR2 SWE over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _coord_like(ds: object, want: str) -> str:
    for name in getattr(ds, "coords", {}):
        if want in str(name).lower():
            return str(name)
    return want
