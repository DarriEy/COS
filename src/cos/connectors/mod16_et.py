# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""MODIS MOD16A2 evapotranspiration connector (gridded, basin-reduced).

Ports the SYMFLUENCE native MOD16 handler (registry keys ``mod16`` / ``mod16a2``
/ ``modis_et``) into the canonical COS gridded contract. MOD16A2/MYD16A2 are NASA
8-day composite ET products at 500 m, served as HDF behind Earthdata. The native
handler reads the sinusoidal HDF tiles, masks fill/special values, optionally
applies the QC bitmask, takes the **basin spatial mean**, and converts the 8-day
composite to a daily mean (kg/m²/8day → mm/day).

COS follows the same boundary discipline but reduces a *supplied NetCDF* — the
shape the native acquirer (``MOD16ETAcquirer``) writes (a ``time, lat, lon`` ET
grid, or a pre-reduced ``ET_basin_mean`` series). Live Earthdata HDF fetch is
wired per-connector only where trivial; here the architecture-critical part is
the reduce + unit-canonicalize path, which is hermetically tested without network
or auth against a synthetic in-memory NetCDF.

Unit handling mirrors the native handler exactly:

* source ``ET_500m`` is kg/m²/8day **after** the 0.1 scale factor (which the
  acquirer already applied when it wrote the NetCDF). 1 kg/m² of water = 1 mm,
  so the 8-day composite → daily mean is a divide-by-8: ``mm/day = (kg/m²/8day)/8``;
* a NetCDF already in mm/day (``MOD16_CONVERT_TO_DAILY: true`` at acquisition,
  the default) needs no conversion. The connector detects the source unit from
  the variable's ``units`` attribute (or config ``source_units``) and only
  divides when the source is an 8-day composite.

Fill / special values (32761–32767 × 0.1 = 3276.1+) are masked to NaN upstream
by the acquirer; any residual NaN in the grid becomes a ``MISSING`` point, exactly
as ``reduce_grid`` does for GRACE.
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
    QualityFlag,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

#: number of days in a MOD16A2 composite (kg/m²/8day → mm/day divides by this).
DAYS_IN_COMPOSITE = 8.0
#: MOD16A2 fill / special-pixel floor after the 0.1 scale factor (32761 * 0.1).
#: cloud / not-processed / water / missing / barren / ice-snow / fill all map here.
SPECIAL_VALUE_FLOOR_MM = 3276.1
#: <= this area (km²) defaults to point sampling, mirroring native grace.py.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0

#: candidate ET variable names in the acquirer's NetCDF, in priority order.
_ET_VAR_CANDIDATES = ("ET_basin_mean", "ET", "et", "ET_500m", "et_mm_day")
#: source-unit strings (case/space-insensitive) that mean "8-day composite,
#: divide by 8 to get the canonical mm/day".
_COMPOSITE_UNITS = {"kg/m2/8day", "kgm-2/8day", "mm/8day", "kg/m^2/8day"}


@register("mod16_et")
class MOD16ETConnector(BaseObservationConnector):
    slug = "mod16_et"
    display_name = "NASA MODIS MOD16A2 ET"
    kind = ObservationKind.ET
    structural_class = "gridded"
    base_url = "https://e4ftl01.cr.usgs.gov"
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
        nc_path = self.config.get("nc_path") or self.config.get("path")
        if not nc_path:
            raise ConnectorError(
                self.slug,
                "MOD16 live fetch needs a NetCDF path (config 'nc_path'/'path') or "
                "Earthdata HDF download (not yet wired). The reduce + canonicalize "
                "path is the proven part; supply a MOD16A2 NetCDF (the shape "
                "MOD16ETAcquirer writes) to reduce it.",
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
        """Open a MOD16 NetCDF, reduce to the basin, canonicalize to mm/day."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            var = self._pick_variable(ds)
            da = ds[var]
            source_units = str(da.attrs.get("units", "")).strip()
            times = np.asarray(ds["time"].values)
            dims = list(da.dims)
            lat_name = next((d for d in dims if d in ("lat", "latitude", "y")), None)
            lon_name = next((d for d in dims if d in ("lon", "longitude", "x")), None)
            if lat_name is not None and lon_name is not None:
                lats = np.asarray(ds[lat_name].values, dtype="float64")
                lons = np.asarray(ds[lon_name].values, dtype="float64")
                # order to (time, lat, lon)
                da = da.transpose("time", lat_name, lon_name)
                values = np.asarray(da.values, dtype="float64")
                gridded = True
            else:
                # already a per-time basin-mean series (e.g. ET_basin_mean / ET(time))
                lats = lons = None
                values = np.asarray(da.values, dtype="float64")
                gridded = False

        # Source-unit decision (mirror native: only divide an 8-day composite).
        source_units, values = self._canonicalize_units(source_units, values)

        if gridded:
            points = self._reduce_gridded(lats, lons, times, values, spec)
            reduction = self._choose_reduction(spec)
        else:
            points = self._series_points(times, values)
            reduction = SpatialReduction.BASIN_MEAN

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
                "source": "MODIS MOD16A2",
                "source_doi": "10.5067/MODIS/MOD16A2.061",
                "url": "https://lpdaac.usgs.gov/products/mod16a2v061/",
                "source_units": source_units or "mm/day",
            },
            fetched_at=datetime.now(UTC),
        )

    def _canonicalize_units(self, source_units: str, values):
        """Mask fills, then convert source ET to canonical mm/day.

        Returns the (recognized) source-unit string and the converted array.
        Mirrors the native handler: fill/special pixels (>= 3276.1 after scale)
        are NaN, and an 8-day composite is divided by 8 to a daily mean.
        """
        import numpy as np

        source_units = self.config.get("source_units", source_units) or ""
        normalized = source_units.lower().replace(" ", "")

        values = np.asarray(values, dtype="float64")
        # Mask MOD16 fill / special values (already 0.1-scaled by the acquirer).
        values = np.where(values >= SPECIAL_VALUE_FLOOR_MM, np.nan, values)

        if normalized in {u.replace(" ", "") for u in _COMPOSITE_UNITS}:
            values = values / DAYS_IN_COMPOSITE  # kg/m²/8day -> mm/day
            source_units = "kg/m2/8day"
        else:
            # mm/day, mm day-1, or unlabeled — the acquirer's default output.
            source_units = source_units or "mm/day"
        return source_units, values

    def _reduce_gridded(self, lats, lons, times, values, spec: ReductionSpec):
        from cos.core.reduce import reduce_grid

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MOD16 basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MOD16 nearest_cell requires spec.centroid")

        return reduce_grid(
            lats, lons, times, values,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

    @staticmethod
    def _series_points(times, values) -> list[ObservationPoint]:
        """Wrap an already-reduced (time,) ET vector into canonical points."""
        import numpy as np

        from cos.core.reduce import _as_datetime

        out: list[ObservationPoint] = []
        for t, v in zip(times, values):
            ts = t if isinstance(t, datetime) else _as_datetime(t)
            finite = v is not None and np.isfinite(v)
            out.append(
                ObservationPoint(
                    timestamp=ts,
                    value=float(v) if finite else None,
                    quality=QualityFlag.GOOD if finite else QualityFlag.MISSING,
                )
            )
        return out

    def _pick_variable(self, ds) -> str:
        cfg_var = self.config.get("variable")
        if cfg_var and cfg_var in ds.data_vars:
            return cfg_var
        for name in _ET_VAR_CANDIDATES:
            if name in ds.data_vars:
                return name
        # last resort: first ET-ish, non-QC data var
        for name in ds.data_vars:
            low = str(name).lower()
            if "et" in low and "qc" not in low and "pixel" not in low:
                return str(name)
        raise ConnectorError(self.slug, f"No ET variable found in NetCDF (vars: {list(ds.data_vars)})")

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"mod16_et:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"mod16_et:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MOD16 ET over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
