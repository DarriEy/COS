# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""VIIRS snow-cover-area connector (gridded, basin-reduced).

Ports SYMFLUENCE's native ``viirs_snow`` / ``vnp10`` observation handler into the
canonical COS contract. VIIRS (Visible Infrared Imaging Radiometer Suite, the
MODIS successor) NDSI snow-cover products (VNP10A1 / VNP10A1F) are gridded NDSI
snow-cover-percent rasters served as NetCDF/HDF behind NASA Earthdata. This
connector:

1. opens a VIIRS snow NetCDF (a local cached file, or a downloaded one — Earthdata
   auth via ``.netrc`` / the resolved credential token);
2. extracts ``lat / lon / time`` and the NDSI snow-cover variable as numpy arrays;
3. masks VIIRS fill / sentinel codes (cloud, night, ocean, missing, ...) and any
   value outside the physical 0–100 % range to NaN, exactly mirroring the native
   handler's ``NDSI_FILL_VALUES`` + ``NDSI_VALID_RANGE`` masking;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the GRACE-style size policy, made
   explicit and configurable);
5. converts NDSI snow-cover **percent (0–100)** → **fraction (0–1)** (the canonical
   ``snow_cover`` unit), exactly mirroring the native ``sca / 100.0`` step.

The fetch path is exercised only with Earthdata credentials; the mask + reduce +
canonicalize path is hermetically tested with a synthetic in-memory NetCDF, so the
architecture-critical reduction logic is covered without network or auth.
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

PERCENT_TO_FRACTION = 100.0
#: VIIRS NDSI snow-cover valid data range (percent). Mirrors native NDSI_VALID_RANGE.
NDSI_VALID_RANGE = (0.0, 100.0)
#: VIIRS NDSI snow-cover fill / sentinel codes (cloud, night, ocean, missing, ...).
#: Mirrors the native handler's NDSI_FILL_VALUES exactly.
NDSI_FILL_VALUES = (200, 201, 211, 237, 239, 250, 251, 252, 253, 254, 255)
#: Candidate NDSI snow-cover variable names, in native handler priority order.
SCA_VARIABLE_CANDIDATES = (
    "CGF_NDSI_Snow_Cover",
    "NDSI_Snow_Cover",
    "snow_cover",
    "SCA",
    "sca",
)
LAT_CANDIDATES = ("lat", "latitude", "y")
LON_CANDIDATES = ("lon", "longitude", "x")
TIME_CANDIDATES = ("time", "date")
#: <= this area (km²) defaults to point sampling, mirroring native grace.py policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("viirs_sca")
class VIIRSSnowCoverConnector(BaseObservationConnector):
    slug = "viirs_sca"
    display_name = "NASA VIIRS Snow Cover (VNP10)"
    kind = ObservationKind.SNOW_COVER
    structural_class = "gridded"
    base_url = "https://nsidc.org"
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
                "VIIRS live fetch needs a NetCDF path (config 'nc_path'/'path') or "
                "Earthdata download (not yet wired). The reduction path is the proven "
                "part; supply a downloaded VIIRS VNP10 snow-cover NetCDF to reduce it.",
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
        """Open a VIIRS snow NetCDF, mask + reduce to the basin, canonicalize to fraction."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            sca_var = self._find_name(ds.data_vars, SCA_VARIABLE_CANDIDATES)
            if sca_var is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing an NDSI snow-cover variable (tried {SCA_VARIABLE_CANDIDATES})",
                )
            lat_name = self._find_name({**ds.coords, **ds.dims}, LAT_CANDIDATES)
            lon_name = self._find_name({**ds.coords, **ds.dims}, LON_CANDIDATES)
            time_name = self._find_name({**ds.coords, **ds.dims}, TIME_CANDIDATES)
            if lat_name is None or lon_name is None or time_name is None:
                raise ConnectorError(
                    self.slug, "NetCDF missing lat/lon/time coordinate(s)"
                )

            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds[time_name].values)
            values = np.asarray(ds[sca_var].values, dtype="float64")  # (time, lat, lon)

        values = self._mask_fill_and_range(values)

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("VIIRS basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("VIIRS nearest_cell requires spec.centroid")

        # percent (0-100) -> fraction (0-1) at the boundary, mirroring native sca/100.
        points = reduce_grid(
            lats, lons, times, values / PERCENT_TO_FRACTION,
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
                "source": "NASA VIIRS VNP10 (NDSI snow cover)",
                "url": "https://nsidc.org/data/vnp10a1f",
                "native_range_percent": "0-100",
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _mask_fill_and_range(values):
        """Mask VIIRS fill/sentinel codes and out-of-range values to NaN.

        Mirrors the native handler: drop NDSI_FILL_VALUES, then keep only
        physically-valid percentages in NDSI_VALID_RANGE. Masked cells become
        NaN, so reduce_grid records them as QualityFlag.MISSING.
        """
        import numpy as np

        out = values.copy()
        fill_mask = np.isin(out, np.asarray(NDSI_FILL_VALUES, dtype="float64"))
        lo, hi = NDSI_VALID_RANGE
        range_mask = (out < lo) | (out > hi)
        out[fill_mask | range_mask] = np.nan
        return out

    @staticmethod
    def _find_name(container, candidates) -> str | None:
        for name in candidates:
            if name in container:
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
            site_id = f"viirs_sca:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"viirs_sca:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"VIIRS snow cover over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["VIIRSSnowCoverConnector", "QualityFlag"]
