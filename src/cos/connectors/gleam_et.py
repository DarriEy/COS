# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GLEAM evapotranspiration connector (gridded, basin-reduced).

Ports the SYMFLUENCE native ``gleam_et`` observation handler
(``data/observation/handlers/gleam.py`` + the SFTP acquirer
``data/acquisition/handlers/gleam_et.py``) onto the canonical COS contract.

GLEAM (Global Land Evaporation Amsterdam Model) is a global 0.25 deg land-
evaporation product (Miralles et al. 2011; Martens et al. 2017), distributed as
yearly NetCDF files via SFTP after free registration at https://www.gleam.eu/.
The primary variable ``E`` (total evaporation) is reported in **mm/day**, which
is already the canonical ``et`` unit — so the conversion at the boundary is the
identity (native default), with an optional config multiplier mirroring the
native ``ET_UNIT_CONVERSION``.

This connector follows the GRACE gridded template:

1. opens a GLEAM NetCDF (a local cached file supplied via config ``nc_path`` /
   ``path`` — live SFTP fetch is auth-gated and wired per-deployment, exactly as
   GRACE's Earthdata download is);
2. extracts ``lat / lon / time / E`` as numpy arrays (variable auto-detected the
   same way the native ``_select_et_variable`` does);
3. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (size policy made explicit and
   configurable, matching the GRACE house pattern);
4. converts source mm/day → canonical mm/day (identity by default) at the
   boundary and trims to a half-open UTC ``[start, end)`` window.

The reduce + canonicalize path is hermetically tested with a synthetic in-memory
NetCDF, so the architecture-critical reduction logic is covered without network
or GLEAM credentials.
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

#: <= this area (km²) defaults to point sampling, mirroring the GRACE house policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0

#: candidate ET variable names, mirroring native ``_select_et_variable``.
_ET_NAMES = {"e", "et", "evap", "evaporation", "evapotranspiration"}


@register("gleam_et")
class GLEAMETConnector(BaseObservationConnector):
    slug = "gleam_et"
    display_name = "GLEAM Evapotranspiration"
    kind = ObservationKind.ET
    structural_class = "gridded"
    base_url = "https://www.gleam.eu"
    auth = frozenset({"gleam"})

    SOURCE_INFO = {
        "source": "GLEAM",
        "source_doi": "10.5194/gmd-10-1903-2017",
        "url": "https://www.gleam.eu/",
    }

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
                "GLEAM live fetch needs a NetCDF path (config 'nc_path'/'path') or an "
                "SFTP download (auth-gated, not wired here). The reduction path is the "
                "proven part; supply a downloaded GLEAM yearly NetCDF to reduce it.",
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
        """Open a GLEAM NetCDF, reduce to the basin, canonicalize to mm/day."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var = self._select_et_variable(ds)
            if var is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF has no recognizable ET variable (data_vars={list(ds.data_vars)})",
                )
            lat_name = _coord_name(ds, {"lat", "latitude"})
            lon_name = _coord_name(ds, {"lon", "longitude"})
            if lat_name is None or lon_name is None:
                raise ConnectorError(self.slug, "NetCDF missing lat/lon coordinates")
            da = ds[var]
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            # Order axes to (time, lat, lon) as reduce_grid expects.
            da = da.transpose("time", lat_name, lon_name)
            values = np.asarray(da.values, dtype="float64")

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("GLEAM basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("GLEAM nearest_cell requires spec.centroid")

        # Source -> canonical: GLEAM E is mm/day (== KIND_UNITS[ET]); identity by
        # default. An optional multiplier mirrors the native ET_UNIT_CONVERSION.
        factor = self._unit_factor()

        points = reduce_grid(
            lats, lons, times, values * factor,  # mm/day -> mm/day at the boundary
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)).
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
            source_info={**self.SOURCE_INFO, "variable": var},
            fetched_at=datetime.now(UTC),
        )

    # -- helpers -------------------------------------------------------------

    def _unit_factor(self) -> float:
        """Source -> canonical multiplier (identity unless ET_UNIT_CONVERSION set)."""
        conv = self.config.get("unit_conversion")
        if conv is None:
            conv = self.config.get("ET_UNIT_CONVERSION")
        if conv is None:
            return 1.0
        try:
            return float(conv)
        except (TypeError, ValueError):
            logger.warning("gleam_et invalid unit_conversion, ignoring", value=conv)
            return 1.0

    def _select_et_variable(self, ds) -> str | None:
        """Pick the ET variable, mirroring native ``_select_et_variable``."""
        preferred = self.config.get("et_variable") or self.config.get("ET_VARIABLE_NAME")
        if preferred and preferred in ds.data_vars:
            return str(preferred)

        candidates: list[str] = []
        for name in ds.data_vars:
            lower = str(name).lower()
            if lower in _ET_NAMES:
                candidates.append(str(name))
            elif "et" in lower and "pet" not in lower:
                candidates.append(str(name))
        if candidates:
            return candidates[0]

        if len(ds.data_vars) == 1:
            return str(next(iter(ds.data_vars)))
        return None

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"gleam_et:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"gleam_et:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"GLEAM ET over {spec.domain_name}",
        )


def _coord_name(ds, candidates: set[str]) -> str | None:
    for name in ds.coords:
        if str(name).lower() in candidates:
            return str(name)
    return None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
