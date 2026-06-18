# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NASA MODIS MCD43A3 broadband albedo connector (gridded, basin-reduced).

A **multivariate-breadth** connector: surface albedo is an orthogonal
energy-balance constraint (distinct from the water-cycle kinds) for
multi-objective model evaluation. There is no SYMFLUENCE native albedo handler,
so this connector is **spec-validated** — it reproduces the published MCD43A3
product specification on a synthetic fixture rather than matching a native port.

MCD43A3 source (LP DAAC, NASA Earthdata):
    * MODIS/Terra+Aqua BRDF/Albedo Daily L3 Global 500 m (MCD43A3).
    * Broadband shortwave albedo, **white-sky** (bi-hemispherical reflectance,
      ``Albedo_BSA_shortwave`` is black-sky / directional-hemispherical; this
      connector defaults to white-sky ``Albedo_WSA_shortwave`` and accepts a
      ``albedo_type`` option of ``"white_sky"`` / ``"black_sky"``).
    * Stored as scaled 16-bit integers: **scale factor 0.001**, valid range
      ``0..1000`` (→ reflectance ``0..1``), fill value ``32767`` → NaN.

This connector extracts ``lats, lons, times, raw`` from a supplied file (config
``nc_path`` / ``path``) and reduces it with :func:`cos.core.reduce.reduce_grid` —
``basin_mean`` (cos-lat weighted) for larger basins, ``nearest_cell`` for small
ones. The unit conversion happens at the connector boundary: the documented scale
factor is applied, out-of-valid-range / fill cells become NaN (→ MISSING), and the
canonical ``albedo`` unit is the dimensionless ``"1"`` (0..1).

The architecture-critical extract→mask→scale→reduce→canonicalize path is
hermetically tested via :meth:`MODISAlbedoConnector.reduce_arrays` on a synthetic
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
    ObservationSeries,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

#: MCD43A3 documented scale factor: stored integer * SCALE = reflectance (0..1).
SCALE_FACTOR = 0.001
#: MCD43A3 documented fill value (no retrieval) — maps to NaN -> MISSING.
FILL_VALUE = 32767
#: Documented valid stored-integer range (inclusive); -> reflectance 0..1.
VALID_MIN = 0
VALID_MAX = 1000
#: White-sky (bi-hemispherical) vs black-sky (directional-hemispherical) bands.
ALBEDO_VARIABLES = {
    "white_sky": ("Albedo_WSA_shortwave", "albedo_wsa_shortwave", "wsa_shortwave"),
    "black_sky": ("Albedo_BSA_shortwave", "albedo_bsa_shortwave", "bsa_shortwave"),
}
DEFAULT_ALBEDO_TYPE = "white_sky"
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("modis_albedo")
class MODISAlbedoConnector(BaseObservationConnector):
    slug = "modis_albedo"
    display_name = "NASA MODIS MCD43A3 Broadband Albedo"
    kind = ObservationKind.ALBEDO
    structural_class = "gridded"
    base_url = "https://e4ftl01.cr.usgs.gov"  # LP DAAC MOTA archive
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
                "MODIS albedo live fetch needs a cached file (config 'nc_path'/'path') "
                "— an MCD43A3 NetCDF/HDF subset. LP DAAC Earthdata download is not yet "
                "wired; the reduce + scale path is the proven (spec-validated) part.",
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
        """Open an MCD43A3 NetCDF, extract raw arrays, reduce + canonicalize."""
        import numpy as np
        import xarray as xr

        albedo_type = self._albedo_type(spec)
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            var_name = self._find_variable(ds, albedo_type)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing an MCD43A3 {albedo_type} albedo variable "
                    f"(tried {ALBEDO_VARIABLES[albedo_type]})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            raw = np.asarray(da.values, dtype="float64")  # (time, lat, lon), stored ints
        return self.reduce_arrays(lats, lons, times, raw, spec, start, end, albedo_type=albedo_type)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats,
        lons,
        times,
        raw,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        albedo_type: str = DEFAULT_ALBEDO_TYPE,
    ) -> ObservationSeries:
        """Mask fill/out-of-range, scale to reflectance, basin-reduce, window-trim.

        *raw* is shaped ``(time, lat, lon)`` of MCD43A3 stored 16-bit integers.
        Reproduces the documented product spec exactly: cells equal to the fill
        value (``32767``), non-finite, or outside the valid stored range
        ``[0, 1000]`` become NaN (→ MISSING); the remainder are multiplied by the
        documented scale factor (``0.001``) to the canonical dimensionless albedo
        ``"1"`` (0..1) at the boundary, before the spatial reduction.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        raw = np.asarray(raw, dtype="float64")

        # Documented fill + valid-range mask (stored-integer domain). Invalid
        # cells become NaN so the reduction skips them and they surface MISSING.
        invalid = (
            (raw == FILL_VALUE)
            | ~np.isfinite(raw)
            | (raw < VALID_MIN)
            | (raw > VALID_MAX)
        )
        # Apply the documented scale factor at the boundary: stored * 0.001 ->
        # reflectance (0..1), the canonical albedo unit. Linear, so scaling
        # pre-reduction is identical to post-reduction and keeps reduce_grid in
        # the canonical unit.
        scaled = np.where(invalid, np.nan, raw * SCALE_FACTOR)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MODIS albedo basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MODIS albedo nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, scaled,
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
                "source": "NASA MODIS MCD43A3 (Terra+Aqua BRDF/Albedo)",
                "product": "MCD43A3.061",
                "url": "https://lpdaac.usgs.gov/products/mcd43a3v061/",
                "albedo_type": albedo_type,
                "scale_factor": f"{SCALE_FACTOR:g}",
            },
            fetched_at=datetime.now(UTC),
        )

    # -- helpers --------------------------------------------------------------

    def _albedo_type(self, spec: ReductionSpec) -> str:
        kind = spec.options.get("albedo_type", self.config.get("albedo_type", DEFAULT_ALBEDO_TYPE))
        if kind not in ALBEDO_VARIABLES:
            raise ConnectorError(
                self.slug,
                f"Unknown albedo_type {kind!r}; expected one of {sorted(ALBEDO_VARIABLES)}",
            )
        return str(kind)

    def _find_variable(self, ds: object, albedo_type: str) -> str | None:
        """Pick the albedo variable for *albedo_type*, by documented name then alias."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in ALBEDO_VARIABLES[albedo_type]:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises the requested band.
        token = "wsa" if albedo_type == "white_sky" else "bsa"
        for name in data_vars:
            lower = str(name).lower()
            if token in lower and "shortwave" in lower:
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
            site_id = f"modis_albedo:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"modis_albedo:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MODIS albedo over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
