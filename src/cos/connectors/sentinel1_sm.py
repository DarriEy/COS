# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Copernicus Sentinel-1 SAR surface soil-moisture connector (gridded, basin-reduced).

Ports SYMFLUENCE's native ``sentinel1_sm`` / ``s1_sm`` observation handler
(``data/observation/handlers/sentinel1_sm.py``) onto the COS canonical contract.

Sentinel-1 source (Copernicus Data Space / Copernicus Global Land Service):
    * C-band SAR-derived **surface soil moisture** at ~1 km, 6–12 day revisit —
      a high-resolution complement to the coarser passive-microwave products
      (SMAP/SMOS). Distributed by Copernicus as NetCDF/GeoTIFF carrying a
      soil-moisture variable (``soil_moisture`` / ``sm`` for a volumetric
      retrieval, or ``SSM`` / ``SWI`` for the Copernicus Global Land surface
      soil-moisture / soil-water-index product expressed as **% of saturation**,
      0–100).
    * The native SYMFLUENCE handler resolves the soil-moisture variable from the
      candidate list ``[soil_moisture, sm, SSM, SWI]``, subsets to the basin
      bounding box, and takes the ``skipna`` spatial mean (fill / NaN cells are
      skipped). It applies no scaling — it passes the variable's own values
      through as the basin-mean soil-moisture series.

This connector reproduces those semantics on the COS gridded path and adds the
explicit source→canonical (``m3/m3``) conversion at the boundary that the
framework's per-kind unit table requires:

    * a *volumetric* variable (``soil_moisture`` / ``sm``) is already m³/m³, so
      the boundary conversion is the identity (scale 1.0) — exactly the native
      pass-through;
    * the Copernicus *% saturation* variable (``SSM`` / ``SWI``, 0–100) is scaled
      by 0.01 to a 0–1 degree-of-saturation that the canonical ``m3/m3`` slot
      carries (configurable via ``source_scale``).

Fill / out-of-physical-range cells become NaN so the reduction skips them and
they surface as :class:`~cos.core.models.QualityFlag.MISSING`, matching the
native ``skipna`` mean. The architecture-critical extract→mask→scale→reduce→
canonicalize path is hermetically tested via
:meth:`Sentinel1SoilMoistureConnector.reduce_arrays` on a synthetic in-memory
grid, with no network and no auth.
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

#: Candidate soil-moisture variable names, in the native handler's preference
#: order: volumetric retrievals first (``soil_moisture`` / ``sm``), then the
#: Copernicus Global Land surface soil moisture / soil-water index (``SSM`` /
#: ``SWI``).
SM_VARIABLES = ("soil_moisture", "sm", "SSM", "SWI")
#: Volumetric variables — already m³/m³, identity boundary scale (native passes
#: them through unchanged).
VOLUMETRIC_VARIABLES = frozenset({"soil_moisture", "sm"})
#: Copernicus % saturation variables (0–100) — scaled to a 0–1 fraction.
SATURATION_VARIABLES = frozenset({"SSM", "SWI"})
#: Boundary scales mapping each source variable onto the canonical ``m3/m3``.
SOURCE_SCALE: dict[str, float] = {
    "soil_moisture": 1.0,
    "sm": 1.0,
    "SSM": 0.01,   # % of saturation (0–100) -> 0–1
    "SWI": 0.01,   # soil-water index (0–100) -> 0–1
}
#: Native fill sentinel commonly carried by Copernicus SSM rasters.
FILL_VALUE = -9999.0
#: Physically valid degree-of-saturation / volumetric band on the SCALED value
#: (m³/m³): keep 0 <= sm <= this. Cells outside become MISSING.
MAX_VALID_M3M3 = 1.0
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("sentinel1_sm")
class Sentinel1SoilMoistureConnector(BaseObservationConnector):
    slug = "sentinel1_sm"
    display_name = "Copernicus Sentinel-1 Surface Soil Moisture"
    kind = ObservationKind.SOIL_MOISTURE
    structural_class = "gridded"
    base_url = "https://catalogue.dataspace.copernicus.eu"
    auth = frozenset({"cdse"})  # Copernicus Data Space credentials

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
                "Sentinel-1 SM live fetch needs a cached file (config 'nc_path'/'path') "
                "— a Copernicus surface soil-moisture NetCDF/GeoTIFF. Copernicus Data "
                "Space download is not yet wired; the reduce + unit-conversion path is "
                "the proven part. Supply a downloaded file to reduce it.",
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
        """Open a Sentinel-1 SM NetCDF, extract arrays, reduce + canonicalize."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a Sentinel-1 soil-moisture variable (tried {SM_VARIABLES})",
                )
            da = ds[var_name]
            lat_name = _coord_like(ds, ("lat", "latitude", "y"))
            lon_name = _coord_like(ds, ("lon", "longitude", "x"))
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
        return self.reduce_arrays(lats, lons, times, values, var_name, spec, start, end)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats,
        lons,
        times,
        values,
        var_name: str,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Mask fill/out-of-range, scale source→m³/m³, basin-reduce, window-trim.

        *values* is shaped ``(time, lat, lon)`` in the source variable's own unit.
        Mirrors the native handler's basin-mean of valid (``skipna``) cells, with
        the explicit source→canonical scale applied at the boundary so the reduced
        series is in ``m3/m3``: a volumetric variable scales by 1.0 (identity, the
        native pass-through); the Copernicus % saturation product scales by 0.01.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        vals = np.asarray(values, dtype="float64")

        scale = float(self.config.get("source_scale", SOURCE_SCALE.get(var_name, 1.0)))
        # Mask the native fill sentinel and any non-finite cell before scaling.
        vals = np.where((vals == FILL_VALUE) | ~np.isfinite(vals), np.nan, vals)
        # Convert source → canonical m³/m³ at the boundary (linear, so applying it
        # pre-reduction is identical to scaling the native basin-mean afterwards).
        canon = vals * scale
        # Physical-plausibility mask on the SCALED value: 0 <= sm <= 1.
        canon = np.where((canon < 0.0) | (canon > MAX_VALID_M3M3), np.nan, canon)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("Sentinel-1 SM basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("Sentinel-1 SM nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, canon,
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
                "source": "Copernicus Sentinel-1 Surface Soil Moisture",
                "url": "https://dataspace.copernicus.eu/",
                "variable": var_name,
                "source_scale": f"{scale:g}",
            },
            fetched_at=datetime.now(UTC),
        )

    def _find_variable(self, ds: object) -> str | None:
        """Pick the soil-moisture variable, native preference order, then fuzzy."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in SM_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            lower = name.lower()
            if "soil_moisture" in lower or lower in ("ssm", "swi", "sm"):
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
            site_id = f"sentinel1_sm:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"sentinel1_sm:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"Sentinel-1 soil moisture over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _coord_like(ds: object, candidates: tuple[str, ...]) -> str:
    coords = set(getattr(ds, "coords", {})) | set(getattr(ds, "dims", {}))
    for name in candidates:
        if name in coords:
            return name
    for name in coords:
        low = str(name).lower()
        if any(c in low for c in candidates):
            return str(name)
    return candidates[0]
