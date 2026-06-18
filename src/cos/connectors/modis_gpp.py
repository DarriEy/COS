# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NASA MODIS MOD17A2H gross-primary-productivity connector (gridded, reduced).

Serves :class:`~cos.core.models.ObservationKind.GPP` on the COS canonical
gridded contract. MOD17A2H is the 500 m, 8-day composite Gross Primary
Productivity product distributed by the NASA LP DAAC behind Earthdata.

There is **no** SYMFLUENCE native handler for MOD17 GPP, so this connector is
*spec-validated*: its unit / scale / fill handling reproduces the published
MOD17A2H product specification rather than mirroring an internal native.

Published MOD17A2H product spec (validated by the offline tests):

* the ``Gpp_500m`` SDS is a 16-bit unsigned digital count;
* the **scale factor is 0.0001**, so ``kgC/m2/8day = digital * 0.0001``;
* digital values ``>= 32761`` are fill / special pixels (water / barren /
  cloud / unclassified / fill) and map to NaN;
* the valid digital range is therefore ``0 .. 32760`` (``0 .. 3.276 kgC/m2/8day``
  after scaling).

The canonical unit for GPP is ``gC/m2/day`` (see
:data:`cos.core.models.KIND_UNITS`). The boundary conversion is therefore::

    gC/m2/day = (digital * 0.0001) * 1000 / interval_days

where ``1000`` converts kgC -> gC and ``interval_days`` is the composite length
(8 days for a full MOD17A2H composite; the trailing composite of a year is 5 or
6 days, so ``interval_days`` is configurable / per-timestep).

A NetCDF that an upstream acquirer already wrote in ``gC/m2/day`` (the canonical
unit, advertised via the variable ``units`` attribute or config ``source_units``)
is passed through unchanged. The architecture-critical extract -> mask -> scale
-> convert -> reduce -> canonicalize path is hermetically tested with a synthetic
in-memory NetCDF, with no network and no auth.
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

#: MOD17A2H published scale factor: kgC/m2/8day = digital * SCALE_FACTOR.
SCALE_FACTOR = 0.0001
#: kgC -> gC.
KG_TO_G = 1000.0
#: Default composite length (days) for an 8-day MOD17A2H composite.
DAYS_IN_COMPOSITE = 8.0
#: MOD17A2H fill / special-pixel digital floor: digital >= this is invalid.
#: (32761 water, 32762 barren, ... 32767 fill — all >= 32761 -> NaN.)
SPECIAL_VALUE_MIN_DIGITAL = 32761
#: <= this area (km²) defaults to point sampling (nearest cell).
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0

#: candidate GPP variable names in priority order.
_GPP_VAR_CANDIDATES = ("GPP_basin_mean", "Gpp_500m", "GPP", "gpp", "gpp_gC_m2_day")
#: source-unit strings (case/space-insensitive) that are ALREADY the canonical
#: gC/m2/day output of an acquirer — pass through without scaling.
_DAILY_UNITS = {"gc/m2/day", "gcm-2/day", "gc/m^2/day", "gc/m2/d"}
#: source-unit strings meaning "raw kgC/m2/8day composite, scale + /interval".
_COMPOSITE_UNITS = {"kgc/m2/8day", "kgcm-2/8day", "kgc/m^2/8day"}


@register("modis_gpp")
class MODISGPPConnector(BaseObservationConnector):
    slug = "modis_gpp"
    display_name = "NASA MODIS MOD17A2H GPP"
    kind = ObservationKind.GPP
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
                "MODIS GPP live fetch needs a NetCDF path (config 'nc_path'/'path') "
                "or an Earthdata MOD17A2H HDF download (not yet wired). The reduce + "
                "canonicalize path is the proven part; supply a MOD17A2H NetCDF "
                "(a time, lat, lon Gpp_500m grid) to reduce it.",
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
        """Open a MOD17A2H NetCDF, reduce to the basin, canonicalize to gC/m2/day."""
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
            lats: np.ndarray | None
            lons: np.ndarray | None
            if lat_name is not None and lon_name is not None:
                lats = np.asarray(ds[lat_name].values, dtype="float64")
                lons = np.asarray(ds[lon_name].values, dtype="float64")
                da = da.transpose("time", lat_name, lon_name)
                values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
                gridded = True
            else:
                # already a per-time basin-mean series (e.g. GPP_basin_mean(time))
                lats = lons = None
                values = np.asarray(da.values, dtype="float64")
                gridded = False

        # Mask fills + apply the published scale + convert to canonical gC/m2/day.
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
                "source": "MODIS MOD17A2H",
                "source_doi": "10.5067/MODIS/MOD17A2H.061",
                "url": "https://lpdaac.usgs.gov/products/mod17a2hv061/",
                "source_units": source_units or "gC/m2/day",
                "scale_factor": f"{SCALE_FACTOR:g}",
            },
            fetched_at=datetime.now(UTC),
        )

    def _canonicalize_units(self, source_units: str, values):
        """Mask fills, scale, and convert the source GPP to canonical gC/m2/day.

        Returns the (recognized) source-unit string and the converted array.
        Reproduces the published MOD17A2H spec exactly:

        * a raw composite source (digital counts, the product's native form) has
          digital ``>= 32761`` masked to NaN, then is scaled by 0.0001 to
          kgC/m2/8day and converted to gC/m2/day via ``* 1000 / interval_days``;
        * a source already in gC/m2/day is passed through unchanged.
        """
        import numpy as np

        source_units = self.config.get("source_units", source_units) or ""
        normalized = source_units.lower().replace(" ", "")
        values = np.asarray(values, dtype="float64")

        if normalized in {u.replace(" ", "") for u in _DAILY_UNITS}:
            # already canonical gC/m2/day — only mask non-finite cells.
            return "gC/m2/day", values

        # Raw composite (digital counts): mask the published fill floor BEFORE
        # scaling, then scale 0.0001 (-> kgC/m2/8day) and convert to gC/m2/day.
        values = np.where(values >= SPECIAL_VALUE_MIN_DIGITAL, np.nan, values)
        interval_days = float(
            self.config.get("interval_days", DAYS_IN_COMPOSITE) or DAYS_IN_COMPOSITE
        )
        values = values * SCALE_FACTOR * KG_TO_G / interval_days
        recognized = "kgC/m2/8day" if normalized in {
            u.replace(" ", "") for u in _COMPOSITE_UNITS
        } else "digital"
        return recognized, values

    def _reduce_gridded(self, lats, lons, times, values, spec: ReductionSpec):
        from cos.core.reduce import reduce_grid

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MODIS GPP basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MODIS GPP nearest_cell requires spec.centroid")

        return reduce_grid(
            lats, lons, times, values,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

    @staticmethod
    def _series_points(times, values) -> list[ObservationPoint]:
        """Wrap an already-reduced (time,) GPP vector into canonical points."""
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
            return str(cfg_var)
        for name in _GPP_VAR_CANDIDATES:
            if name in ds.data_vars:
                return str(name)
        # last resort: first GPP-ish, non-QC data var
        for name in ds.data_vars:
            low = str(name).lower()
            if "gpp" in low and "qc" not in low and "pixel" not in low:
                return str(name)
        raise ConnectorError(self.slug, f"No GPP variable found in NetCDF (vars: {list(ds.data_vars)})")

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"modis_gpp:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"modis_gpp:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MODIS GPP over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
