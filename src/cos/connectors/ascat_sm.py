# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""EUMETSAT ASCAT surface soil-moisture connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
C-band active-microwave soil-moisture product. ASCAT (Advanced Scatterometer,
MetOp) retrievals are served on a ~25 km grid (resampled to 0.25 deg) via the
Copernicus CDS ``satellite-soil-moisture`` dataset (active-only sensor type).
This connector:

1. opens an ASCAT NetCDF (a local cached / CDS-downloaded file via config
   ``nc_path`` / ``path``);
2. extracts ``lat / lon / time`` and the soil-moisture variable
   (``surface_soil_moisture_saturation`` / ``ssm`` / ``sm`` / ... — the same
   candidate order the native SYMFLUENCE handler probes) as numpy arrays;
3. canonicalizes ASCAT's native **degree of saturation** to volumetric soil
   moisture EXACTLY as the native handler does:

   * if the layer mean exceeds 1.0 it is treated as a percentage and divided by
     100 (→ saturation fraction 0–1);
   * if the variable name advertises ``saturation`` (or the fraction mean still
     exceeds 0.5, the native heuristic), the saturation fraction is multiplied
     by a configurable ``porosity`` (default 0.45) to estimate volumetric
     m³/m³;
   * values outside the physical range ``0 < sm < 1`` are masked to NaN so they
     reduce to MISSING;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the size policy made explicit and
   configurable here);
5. emits the canonical ``soil_moisture`` unit ``m3/m3``.

The fetch path is exercised only with a supplied / CDS-downloaded NetCDF; the
reduce + canonicalize path is hermetically tested with a synthetic in-memory
NetCDF, so the architecture-critical reduction logic is covered without network
or auth.
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

#: Candidate soil-moisture variable names, in preference order, mirroring the
#: native SYMFLUENCE ASCAT handler's probe list.
SM_VARIABLES = (
    "sm",
    "ssm",
    "soil_moisture",
    "surface_soil_moisture",
    "surface_soil_moisture_saturation",
    "volumetric_surface_soil_moisture",
)
#: Default porosity for the saturation -> volumetric conversion (native default).
DEFAULT_POROSITY = 0.45
#: <= this area (km²) defaults to point sampling (nearest cell).
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("ascat_sm")
class ASCATSoilMoistureConnector(BaseObservationConnector):
    slug = "ascat_sm"
    display_name = "EUMETSAT ASCAT Surface Soil Moisture"
    kind = ObservationKind.SOIL_MOISTURE
    structural_class = "gridded"
    # ASCAT SM is served via the Copernicus Climate Data Store (CDS).
    base_url = "https://cds.climate.copernicus.eu"
    auth = frozenset({"cds"})  # Copernicus CDS (same as smos_sm / esa_cci_sm siblings)

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
                "ASCAT live fetch needs a NetCDF path (config 'nc_path'/'path') or a "
                "CDS satellite-soil-moisture download (not yet wired). The reduction "
                "path is the proven part; supply a downloaded ASCAT NetCDF to reduce it.",
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
        """Open an ASCAT NetCDF, canonicalize saturation -> volumetric m³/m³, reduce."""
        import numpy as np
        import xarray as xr

        porosity = float(self.config.get("porosity", DEFAULT_POROSITY))

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing an ASCAT soil-moisture variable (tried {SM_VARIABLES})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        values = _canonicalize_saturation(values, var_name, porosity)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("ASCAT basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("ASCAT nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # already m³/m³ — converted at the boundary
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim, half-open UTC [start, end).
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "EUMETSAT ASCAT via Copernicus CDS",
                "dataset": "satellite-soil-moisture",
                "sensor": "active",
                "url": "https://cds.climate.copernicus.eu",
                "variable": var_name,
                "porosity": f"{porosity}",
            },
            fetched_at=datetime.now(UTC),
        )

    def _find_variable(self, ds: object) -> str | None:
        """Pick the soil-moisture variable (native probe order, then heuristic)."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in SM_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            lower = name.lower()
            if "sm" in lower or "soil" in lower:
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
            site_id = f"ascat:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"ascat:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"ASCAT soil moisture over {spec.domain_name}",
        )


def _canonicalize_saturation(values, var_name: str, porosity: float):
    """ASCAT degree-of-saturation -> volumetric m³/m³, mirroring the native handler.

    Operates per-timestep on a ``(time, lat, lon)`` array. For each layer:

    * percentage (layer finite-mean > 1.0) is divided by 100 -> saturation 0–1;
    * if the variable name advertises ``saturation`` *or* the fraction mean still
      exceeds 0.5 (the native heuristic), multiply by *porosity* -> volumetric;
    * values outside ``0 < sm < 1`` become NaN (reduce to MISSING).

    Network-free and pure: unit-testable in isolation.
    """
    import numpy as np

    out = np.array(values, dtype="float64", copy=True)
    is_saturation = "saturation" in var_name.lower()

    layers = out if out.ndim == 3 else out[np.newaxis, ...]
    for layer in layers:
        finite = np.isfinite(layer)
        if not finite.any():
            continue
        layer_mean = float(np.nanmean(layer[finite]))
        # Percentage -> fraction.
        if layer_mean > 1.0:
            layer[finite] = layer[finite] / 100.0
            layer_mean = float(np.nanmean(layer[finite]))
        # Saturation -> volumetric.
        if is_saturation or layer_mean > 0.5:
            layer[finite] = layer[finite] * porosity

    # Mask the physical valid range (0 < sm < 1); fill/out-of-range -> MISSING.
    invalid = ~np.isfinite(out) | (out <= 0.0) | (out >= 1.0)
    out = np.where(invalid, np.nan, out)
    return out


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
