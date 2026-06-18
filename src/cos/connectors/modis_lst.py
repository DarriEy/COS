# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""MODIS Land Surface Temperature connector (gridded, basin-reduced).

Ports SYMFLUENCE's native ``modis_lst`` / ``mod11`` observation handler
(``data/observation/handlers/modis_lst.py``) onto the COS canonical contract.
The native handler ingests MODIS MOD11A1 / MYD11A1 (daily) and MOD11A2 /
MYD11A2 (8-day composite) Land Surface Temperature rasters (NDSI-style packed
DN) and produces a basin-mean temperature time series.

Source semantics mirrored exactly from the native handler:

* **variable**: day-LST priority order ``LST_Day_1km`` -> ``LST_Day`` ->
  ``lst_day``, night-LST ``LST_Night_1km`` -> ``LST_Night`` -> ``lst_night``.
  The connector serves one band per series; ``config['band']`` selects
  ``"day"`` (default) or ``"night"``;
* **valid range**: packed DN in ``[7500, 65535]`` (``LST_VALID_RANGE``); the
  fill value ``0`` and everything below 7500 is invalid. Out-of-range cells are
  masked to NaN -> ``QualityFlag.MISSING`` in the canonical series;
* **scale**: source is packed unsigned-int DN; the native handler multiplies by
  ``LST_SCALE_FACTOR = 0.02`` to recover **Kelvin**. The canonical ``lst`` unit
  is ``"K"``, so the DN -> Kelvin scale happens at the boundary and no further
  conversion is applied (the native ``celsius`` output option is a downstream
  presentation choice, not the canonical unit);
* **reduction**: spatial basin-mean over the bbox (the COS gridded path);
  ``nearest_cell`` for small basins, matching grace.py's size policy.

The Earthdata (AppEEARS / netrc) download path is wired per-connector only where
trivial; here, as with grace.py / modis_sca.py, the proven part is the reduce +
canonicalize path, exercised hermetically against a supplied NetCDF (config
``nc_path`` / ``path``).
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

#: Packed-DN valid range; the fill value (0) and everything below 7500 is invalid.
LST_VALID_RANGE = (7500.0, 65535.0)
#: Native handler's packed-DN -> Kelvin scale factor.
LST_SCALE_FACTOR = 0.02
#: Day / night LST variable names, native-handler priority order.
DAY_VARIABLES = ("LST_Day_1km", "LST_Day", "lst_day")
NIGHT_VARIABLES = ("LST_Night_1km", "LST_Night", "lst_night")
#: <= this area (km²) defaults to nearest_cell, mirroring grace.py's policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("modis_lst")
class MODISLSTConnector(BaseObservationConnector):
    slug = "modis_lst"
    display_name = "NASA MODIS Land Surface Temperature (MOD11/MYD11)"
    kind = ObservationKind.LST
    structural_class = "gridded"
    base_url = "https://appeears.earthdatacloud.nasa.gov"
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
                "MODIS LST live fetch needs a NetCDF path (config 'nc_path' or 'path') "
                "or an Earthdata/AppEEARS download (not yet wired). The reduction path "
                "is the proven part; supply a MOD11A1/MYD11A1 (or 8-day MOD11A2) LST "
                "NetCDF to reduce it.",
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
        """Open a MODIS LST NetCDF, reduce to the basin, canonicalize to Kelvin.

        Masks packed DN outside ``[7500, 65535]`` (fill / invalid) to NaN,
        reduces over the basin, then scales DN -> Kelvin (``* 0.02``) at the
        boundary so values are in the canonical ``lst`` unit ``"K"``.
        Window-trimmed to half-open UTC ``[start, end)``.
        """
        import numpy as np
        import xarray as xr

        band = str(self.config.get("band", "day")).lower()
        reduction = self._choose_reduction(spec)

        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_lst_variable(ds, band)
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds.coords else ("y" if "y" in ds.coords else "lat")
            lon_name = "lon" if "lon" in ds.coords else ("x" if "x" in ds.coords else "lon")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        # Valid-range mask: keep packed DN in [7500, 65535], everything else
        # (fill 0, below-threshold, out-of-range) -> NaN so it reduces to MISSING.
        values = self._mask_invalid(values)

        # DN -> Kelvin at the boundary (native: val * LST_SCALE_FACTOR). The mean
        # of (DN * 0.02) equals (mean DN) * 0.02, so scaling before the reduction
        # gives the same basin-mean Kelvin the native handler produces.
        values = values * LST_SCALE_FACTOR

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MODIS LST basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MODIS LST nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # already Kelvin
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
            source_info={
                "source": "MODIS MOD11A1/MYD11A1 (LST)",
                "source_doi": "10.5067/MODIS/MOD11A1.061",
                "url": "https://lpdaac.usgs.gov/products/mod11a1v061/",
                "variable": var_name,
                "band": band,
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _mask_invalid(values):
        """Mask packed DN outside the valid range (fill / invalid) to NaN."""
        import numpy as np

        lo, hi = LST_VALID_RANGE
        out = values.astype("float64", copy=True)
        invalid = ~((out >= lo) & (out <= hi))
        out[invalid] = np.nan
        return out

    @staticmethod
    def _find_lst_variable(ds, band: str) -> str:
        """Find the LST variable for *band* ('day'/'night'), native priority order."""
        candidates = NIGHT_VARIABLES if band == "night" else DAY_VARIABLES
        for var in candidates:
            if var in ds.data_vars:
                return str(var)
        token = "night" if band == "night" else "day"
        suitable = [
            v for v in ds.data_vars
            if "lst" in str(v).lower() and token in str(v).lower()
        ]
        if suitable:
            return str(suitable[0])
        raise ConnectorError(
            "modis_lst",
            f"No {band}-band LST variable found in dataset. Available: {list(ds.data_vars)}",
        )

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"modis_lst:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"modis_lst:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MODIS LST over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# Re-export so a quality flag is importable alongside the connector for tests.
__all__ = ["MODISLSTConnector", "QualityFlag", "ObservationPoint"]
