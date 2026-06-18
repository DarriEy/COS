# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GLDAS-2.1 Noah total water storage connector (gridded, basin-reduced).

A second instance of the **gridded spatial-reduction path** of the canonical
contract, modelled on :mod:`cos.connectors.grace`. Where GRACE serves a single
liquid-water-equivalent thickness grid, GLDAS-2.1 Noah monthly TWS is composed
from the model's storage components — exactly mirroring the native SYMFLUENCE
``GLDASAcquirer`` (``GLDAS_NOAH025_M`` v2.1 via Earthdata):

    TWS = SoilMoi0_10cm + SoilMoi10_40cm + SoilMoi40_100cm + SoilMoi100_200cm
          + SWE_inst + CanopInt_inst

Every component is delivered by GLDAS in **kg m-2 == mm** of water, which is
already the canonical ``tws`` unit (:data:`KIND_UNITS`), so the boundary
conversion is the identity — the only canonicalisation is summing the
components and (as for GRACE) subtracting an anomaly-baseline mean so the
series is a TWS anomaly comparable to GRACE.

This connector:

1. opens a GLDAS NetCDF (a local cached / downloaded file — Earthdata netrc
   auth is wired per-connector only where trivial; here we reduce a supplied
   file, the COS gridded pattern);
2. extracts ``lat / lon / time`` and the component variables as numpy arrays,
   summing them into a per-cell TWS field (mm);
3. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` for larger
   basins, ``nearest_cell`` for small ones (the native handler reduces by a
   plain ``mean`` over the bbox; cos-lat basin-mean is the documented COS
   refinement of that, parity tolerance-based);
4. subtracts the anomaly baseline mean (default 2004-01-01..2009-12-31,
   matching the native handler's ``baseline_start`` / ``baseline_end``).

The fetch path is exercised only with Earthdata credentials; the reduce +
canonicalise path is hermetically tested with a synthetic in-memory NetCDF so
the architecture-critical reduction logic is covered without network or auth.
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

#: GLDAS-2.1 Noah storage components (kg m-2 == mm) summed into TWS, mirroring
#: the native ``GLDASAcquirer.SM_VARS`` / ``SWE_VAR`` / ``CANOPY_VAR``.
SOIL_MOISTURE_VARS = (
    "SoilMoi0_10cm_inst",
    "SoilMoi10_40cm_inst",
    "SoilMoi40_100cm_inst",
    "SoilMoi100_200cm_inst",
)
SWE_VAR = "SWE_inst"
CANOPY_VAR = "CanopInt_inst"
COMPONENT_VARS = (*SOIL_MOISTURE_VARS, SWE_VAR, CANOPY_VAR)

#: Native default anomaly baseline window (``baseline_start`` / ``baseline_end``).
DEFAULT_BASELINE = ("2004-01-01", "2009-12-31")
#: <= this area (km²) defaults to point sampling, mirroring native grace.py.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("gldas_tws")
class GLDASTWSConnector(BaseObservationConnector):
    slug = "gldas_tws"
    display_name = "NASA GLDAS-2.1 Noah TWS"
    kind = ObservationKind.TWS
    structural_class = "gridded"
    base_url = "https://hydro1.gesdisc.eosdis.nasa.gov"
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
                "GLDAS live fetch needs a NetCDF path (config 'nc_path'/'path') or "
                "Earthdata download (not yet wired). The reduction path is the proven "
                "part; supply a downloaded GLDAS_NOAH025_M NetCDF to reduce it.",
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
        """Open a GLDAS NetCDF, sum components, reduce to basin, canonicalize mm."""
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            present = [v for v in COMPONENT_VARS if v in ds]
            if not present:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing all GLDAS TWS component variables {COMPONENT_VARS}",
                )
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            # Sum components into a per-cell TWS field (kg m-2 == mm). NaNs in any
            # component propagate (treat-as-missing), matching grid masking; the
            # native handler sums only finite components, so we fill NaN with 0
            # only when at least one component is finite at that cell-time.
            tws = self._sum_components(ds, present, np)  # (time, lat, lon), mm

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("GLDAS basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("GLDAS nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, tws,  # already mm == canonical tws unit (identity)
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim (half-open UTC [start, end)) then anomaly baseline.
        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]
        points = self._apply_baseline(points, spec)

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "GLDAS_NOAH025_M",
                "version": "2.1",
                "components": "+".join(COMPONENT_VARS),
                "url": "https://disc.gsfc.nasa.gov/datasets/GLDAS_NOAH025_M_2.1",
                "baseline": "-".join(spec.options.get("baseline", DEFAULT_BASELINE)),
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _sum_components(ds, present, np):
        """Sum the present GLDAS components into a (time, lat, lon) mm field.

        Each component is broadcast to (time, lat, lon). A cell-time is finite
        (and contributes its summed storage) iff at least one component is
        finite there — finite components add, NaN components count as 0, exactly
        like the native handler's NaN-skipping sum. A cell-time with no finite
        component stays NaN so :func:`reduce_grid` flags it MISSING.
        """
        import xarray as xr

        stacked = xr.concat(
            [ds[v].transpose("time", "lat", "lon") for v in present], dim="_component"
        )
        arr = np.asarray(stacked.values, dtype="float64")  # (n, time, lat, lon)
        any_finite = np.isfinite(arr).any(axis=0)          # (time, lat, lon)
        total = np.nansum(arr, axis=0)                      # NaN treated as 0
        total = np.where(any_finite, total, np.nan)
        return total

    def _apply_baseline(self, points: list, spec: ReductionSpec) -> list:
        """Subtract the baseline-window mean to make a TWS anomaly (mm)."""
        b_start, b_end = spec.options.get("baseline", DEFAULT_BASELINE)
        b0 = _utc(datetime.fromisoformat(b_start))
        b1 = _utc(datetime.fromisoformat(b_end))
        vals = [p.value for p in points if p.value is not None and b0 <= p.timestamp <= b1]
        if not vals:
            vals = [p.value for p in points if p.value is not None]
        if not vals:
            return points
        mean = sum(vals) / len(vals)
        for p in points:
            if p.value is not None:
                p.value = p.value - mean
        return points

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"gldas_tws:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"gldas_tws:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"GLDAS TWS over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
