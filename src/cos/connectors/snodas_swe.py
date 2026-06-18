# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NOAA SNODAS snow-water-equivalent connector (gridded, basin-reduced).

Proves the **gridded spatial-reduction path** for a daily, ~1 km assimilated snow
analysis. SNODAS (Snow Data Assimilation System, NSIDC product G02158) blends
satellite and ground observations into a daily 30-arc-second grid over CONUS and
southern Canada. SWE is served in **metres**. This connector:

1. opens a SNODAS NetCDF (a local cached / pre-merged file via config ``nc_path``
   / ``path`` — the COS pattern, since the live NSIDC FTP fetch + GeoTIFF→NetCDF
   assembly is wired per-connector only where trivial, and SNODAS assembly is
   not trivial);
2. extracts ``lat / lon / time / swe`` as numpy arrays;
3. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (mirroring the native handler's
   skipna basin average, made size-aware and configurable here);
4. converts **m → mm** (the canonical ``swe`` unit, ×1000) at the boundary and
   clips negatives to zero, exactly as the native ``snodas.py`` handler does.

The native handler (``symfluence ... observation/handlers/snodas.py``, registry
keys ``snodas`` / ``snodas_swe``) computes a NaN-skipping basin mean over the
bbox, keeps SWE in metres (writes both ``swe_m`` and ``swe_mm = swe_m*1000``) and
clips to ``>= 0``. We deliver canonical mm directly.

The fetch path needs a supplied NetCDF; the reduce + canonicalize path is
hermetically tested with a synthetic in-memory NetCDF, so the architecture-
critical reduction logic is covered without network or auth.
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
    QualityFlag,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

M_TO_MM = 1000.0
#: <= this area (km²) defaults to point sampling; larger basins get basin_mean.
#: SNODAS is ~1 km, so even modest basins span many cells — keep the threshold
#: small relative to GRACE's coarse mascons.
SMALL_BASIN_THRESHOLD_KM2 = 100.0
#: candidate SWE variable names in SNODAS-derived NetCDFs (mirrors the native
#: handler's ``_find_snow_variable`` SWE variations).
SWE_VAR_CANDIDATES = ("swe", "SWE", "snow_water_equivalent", "SnowWaterEquivalent")
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")


@register("snodas_swe")
class SNODASSWEConnector(BaseObservationConnector):
    slug = "snodas_swe"
    display_name = "NOAA SNODAS Snow Water Equivalent"
    kind = ObservationKind.SWE
    structural_class = "gridded"
    base_url = "https://noaadata.apps.nsidc.org/NOAA/G02158"
    auth = frozenset()  # anonymous NSIDC FTP/HTTPS

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
        nc_path = self.config.get("nc_path") or self.config.get("path")
        if not nc_path:
            raise ConnectorError(
                self.slug,
                "SNODAS live fetch needs a NetCDF path (config 'nc_path'/'path') or "
                "an NSIDC FTP download + GeoTIFF assembly (not yet wired). The "
                "reduction path is the proven part; supply a SNODAS SWE NetCDF "
                "(swe in metres) to reduce it.",
            )
        return [self.reduce_file(Path(nc_path), spec, start, end)]

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open a SNODAS NetCDF, reduce to the basin, canonicalize m→mm."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var = self._find_swe_var(ds)
            if var is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a SWE variable (tried {SWE_VAR_CANDIDATES} + "
                    "any var containing 'swe'/'snow')",
                )
            da = ds[var]
            lat_name = self._find_coord(ds, da, LAT_NAMES)
            lon_name = self._find_coord(ds, da, LON_NAMES)
            if lat_name is None or lon_name is None:
                raise ConnectorError(
                    self.slug, f"Could not locate lat/lon coords on variable '{var}'"
                )
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            # order dims as (time, lat, lon) so reduce_grid sees the contract shape
            da = da.transpose("time", lat_name, lon_name)
            values = np.asarray(da.values, dtype="float64")

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("SNODAS basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("SNODAS nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values * M_TO_MM,  # m -> mm at the boundary
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)).
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]

        # Clip SWE to non-negative, mirroring the native handler's .clip(lower=0).
        for p in points:
            if p.value is not None and p.value < 0.0:
                p.value = 0.0
                if p.quality == QualityFlag.GOOD:
                    p.quality = QualityFlag.ESTIMATED

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "NOAA SNODAS",
                "product": "G02158",
                "url": "https://nsidc.org/data/g02158",
                "resolution": "~1km (30 arc-second), daily",
            },
            fetched_at=datetime.now(UTC),
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _find_swe_var(ds: object) -> str | None:
        data_vars = list(getattr(ds, "data_vars", {}))
        for cand in SWE_VAR_CANDIDATES:
            if cand in data_vars:
                return cand
        for v in data_vars:
            low = str(v).lower()
            if "swe" in low or "snow" in low:
                return str(v)
        return None

    @staticmethod
    def _find_coord(ds: object, da: object, names: tuple[str, ...]) -> str | None:
        # prefer a dim on the variable, fall back to any matching coord on ds
        dims = [str(d) for d in getattr(da, "dims", ())]
        for d in dims:
            if d.lower() in names:
                return d
        for c in getattr(ds, "coords", {}):
            if str(c).lower() in names:
                return str(c)
        return None

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= SMALL_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"snodas_swe:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"snodas_swe:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region",
            site_id=site_id,
            latitude=lat,
            longitude=lon,
            name=f"SNODAS SWE over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
