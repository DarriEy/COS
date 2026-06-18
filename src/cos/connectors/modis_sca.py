# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""MODIS snow-cover-area connector (gridded, basin-reduced).

Ports SYMFLUENCE's native ``modis_snow`` / ``modis_sca`` / ``modis_snow_merged``
observation handler (``data/observation/handlers/modis_snow.py``) onto the COS
canonical contract. The native handler ingests MODIS MOD10A1 (Terra) / MYD10A1
(Aqua) NDSI Snow Cover rasters — optionally Terra+Aqua merged — and produces a
basin-mean fractional snow-cover time series.

Source semantics mirrored exactly from the native handler:

* **variable**: ``NDSI_Snow_Cover`` (priority order ``NDSI_Snow_Cover`` →
  ``snow_cover`` → ``SCA`` → ``sca``);
* **valid range**: NDSI snow cover percent in ``[0, 100]``. Every other byte
  value (200 missing, 201 no-decision, 211 night, 237 inland-water, 239 ocean,
  250 cloud, 254 saturated, 255 fill) is masked to NaN — these are
  ``QualityFlag.MISSING`` in the canonical series;
* **reduction**: spatial mean over the grid (basin-mean over the bbox here, the
  COS gridded path); ``nearest_cell`` for small basins, matching grace.py's
  size policy;
* **units**: source is NDSI snow cover *percent* (0–100); the native handler
  divides by 100 to a fraction (0–1). The canonical ``snow_cover`` unit is
  ``"fraction"``, so the percent→fraction conversion happens at the boundary.

The Earthdata download path is wired per-connector only where trivial; here, as
with grace.py, the proven part is the reduce + canonicalize path, exercised
hermetically against a supplied NetCDF (config ``nc_path`` / ``path``).
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

#: NDSI snow cover valid percent range; everything else is a flag byte.
VALID_SNOW_RANGE = (0.0, 100.0)
#: MODIS special / fill bytes (invalid for snow cover) -> masked to NaN.
MODIS_FILL_VALUES = frozenset({200, 201, 211, 237, 239, 250, 254, 255})
PERCENT_TO_FRACTION = 1.0 / 100.0
#: <= this area (km²) defaults to nearest_cell, mirroring grace.py's policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: snow-cover variable names, native-handler priority order.
SNOW_VARIABLES = ("NDSI_Snow_Cover", "snow_cover", "SCA", "sca")


@register("modis_sca")
class MODISSCAConnector(BaseObservationConnector):
    slug = "modis_sca"
    display_name = "NASA MODIS Snow Cover (MOD10A1/MYD10A1)"
    kind = ObservationKind.SNOW_COVER
    structural_class = "gridded"
    base_url = "https://n5eil01u.ecs.nsidc.org"
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
                "MODIS SCA live fetch needs a NetCDF path (config 'nc_path' or 'path') "
                "or an Earthdata download (not yet wired). The reduction path is the "
                "proven part; supply a MOD10A1/MYD10A1 (or merged) snow-cover NetCDF "
                "to reduce it.",
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
        """Open a MODIS SCA NetCDF, reduce to the basin, canonicalize to fraction.

        Quality-masks the NDSI byte flags to NaN, reduces over the basin, and
        converts percent→fraction at the boundary so values are in the canonical
        ``snow_cover`` unit. Window-trimmed to half-open UTC ``[start, end)``.
        """
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_snow_variable(ds)
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds.coords else ("y" if "y" in ds.coords else "lat")
            lon_name = "lon" if "lon" in ds.coords else ("x" if "x" in ds.coords else "lon")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

        # Quality filter: keep NDSI percent in [0, 100], mask all flag bytes to
        # NaN (mirrors native _apply_quality_filter + MODIS_FILL_VALUES). The
        # range filter already excludes every fill byte (all >= 200), so the
        # range mask is the single source of truth.
        values = self._mask_invalid(values)

        # Percent -> fraction at the boundary (native: df['sca'] / 100.0).
        values = values * PERCENT_TO_FRACTION

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MODIS SCA basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MODIS SCA nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,
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
                "source": "MODIS MOD10A1/MYD10A1",
                "source_doi": "10.5067/MODIS/MOD10A1.061",
                "url": "https://nsidc.org/data/mod10a1",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _mask_invalid(values):
        """Mask NDSI bytes outside [0, 100] (i.e. all MODIS flag/fill) to NaN."""
        import numpy as np

        lo, hi = VALID_SNOW_RANGE
        out = values.astype("float64", copy=True)
        invalid = ~((out >= lo) & (out <= hi))
        out[invalid] = np.nan
        return out

    @staticmethod
    def _find_snow_variable(ds) -> str:
        """Find the snow-cover variable, native-handler priority order."""
        for var in SNOW_VARIABLES:
            if var in ds.data_vars:
                return str(var)
        suitable = [v for v in ds.data_vars if "snow" in str(v).lower() or "ndsi" in str(v).lower()]
        if suitable:
            return str(suitable[0])
        raise ConnectorError(
            "modis_sca",
            f"No snow-cover variable found in dataset. Available: {list(ds.data_vars)}",
        )

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"modis_sca:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"modis_sca:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MODIS snow cover over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# Re-export so a quality flag is importable alongside the connector for tests.
__all__ = ["MODISSCAConnector", "QualityFlag", "ObservationPoint"]
