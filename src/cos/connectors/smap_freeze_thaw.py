# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NASA SMAP L3 surface freeze/thaw-state connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
*categorical* frozen/thawed-state product. NASA's SMAP L3 Radiometer Global Daily
Freeze/Thaw State (**SPL3FTP**, NSIDC product SPL3FTP, ~9 km EASE-Grid 2.0) carries
a per-cell landscape freeze/thaw flag for the AM and PM half-orbits behind NASA
Earthdata:

    0 = thawed    1 = frozen    (anything else = fill / not-retrieved)

There is **no SYMFLUENCE native** for SMAP freeze/thaw, so parity is
*spec-validated*: this connector is validated against the published SPL3FTP product
spec (the 0=thawed / 1=frozen flag, the ``-9999`` fill sentinel, the categorical
range) rather than a native handler. The load-bearing facts encoded here are:

* **canonical unit**: ``KIND_UNITS[ObservationKind.FREEZE_THAW] == "1"`` — a
  dimensionless frozen *fraction* in ``[0, 1]``. The source is a categorical
  0/1 flag, so the connector reduces the binary frozen field (1=frozen) to a
  **basin frozen-fraction** = ``mean(flag == frozen)`` over the in-basin cells via
  :func:`cos.core.reduce.reduce_grid` (an arithmetic / cos-lat-weighted mean of the
  binary field). A single cell (``nearest_cell``) therefore yields exactly the
  raw 0/1 flag — the boundary scale is the identity ``SOURCE_FT_SCALE`` (= 1.0);
* **frozen / thawed codes**: the spec flag values are ``FROZEN_CODE`` (1) and
  ``THAWED_CODE`` (0). The reduction counts frozen over valid (frozen | thawed);
* **fill / missing**: the ``FILL_VALUE`` (``-9999``) and any value outside the
  ``{thawed, frozen}`` categorical set is treated as not-retrieved → masked to NaN
  so the cell is excluded from the fraction; a step with no valid in-basin cell
  surfaces as :class:`QualityFlag.MISSING`;
* **window**: the series is trimmed to the half-open UTC interval ``[start, end)``;
* **AM / PM**: SPL3FTP carries separate AM and PM half-orbit flags; the connector
  selects the configured overpass (``config['overpass']`` / ``spec.options['overpass']``,
  default AM), matching the standard ``freeze_thaw`` (AM) variable.

The fetch path is exercised only with Earthdata credentials; the architecture-
critical extract → mask → reduce → canonicalize path is hermetically tested via
:meth:`SMAPFreezeThawConnector.reduce_arrays` on a synthetic in-memory grid, with
no network and no auth.
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
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

logger = structlog.get_logger()

#: SMAP SPL3FTP native fill / not-retrieved sentinel for the freeze/thaw flag.
FILL_VALUE = -9999.0
#: Spec flag codes: the landscape freeze/thaw state is a binary categorical field.
THAWED_CODE = 0.0
FROZEN_CODE = 1.0
#: The reduced quantity is already a frozen *fraction* in [0, 1] == the canonical
#: ``freeze_thaw`` unit, so the boundary scale is the identity. Documented as a
#: constant so the spec-validated unit contract is explicit.
SOURCE_FT_SCALE = 1.0
#: Candidate freeze/thaw-flag variable names, AM half-orbit preferred (the standard
#: SPL3FTP ``freeze_thaw`` variable), in preference order.
FT_VARIABLES_AM = ("freeze_thaw", "freeze_thaw_am", "frozen_state", "ft_state", "ft")
FT_VARIABLES_PM = ("freeze_thaw_pm", "freeze_thaw_pm_state")
#: <= this area (km²) defaults to point sampling (nearest cell); larger uses
#: basin_mean, mirroring grace.py / smap_sm.py's size policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("smap_freeze_thaw")
class SMAPFreezeThawConnector(BaseObservationConnector):
    slug = "smap_freeze_thaw"
    display_name = "NASA SMAP L3 Freeze/Thaw State"
    kind = ObservationKind.FREEZE_THAW
    structural_class = "gridded"
    base_url = "https://n5eil01u.ecs.nsidc.org"
    auth = frozenset({"earthdata"})  # SPL3FTP download needs Earthdata

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """One reduced region: the basin frozen-fraction (or its centroid cell)."""
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
                "SMAP freeze/thaw live fetch needs a NetCDF path (config 'nc_path'/"
                "'path') or Earthdata download (not yet wired). The reduce + "
                "fraction path is the proven part; supply a downloaded SPL3FTP "
                "NetCDF to reduce it.",
            )
        return [self.reduce_file(Path(nc_path), spec, start, end)]

    # -- file reader (extract arrays, then defer to the pure core) -----------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open an SPL3FTP NetCDF, extract the FT flag, reduce to a frozen fraction."""
        import numpy as np
        import xarray as xr

        overpass = self._overpass(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds, overpass)
            if var_name is None:
                tried = FT_VARIABLES_PM if overpass == "pm" else FT_VARIABLES_AM
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a SMAP freeze/thaw flag variable (overpass="
                    f"{overpass!r}, tried {tried})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            flag = np.asarray(da.values, dtype="float64")  # (time, lat, lon)
        return self.reduce_arrays(lats, lons, times, flag, spec, start, end, variable=var_name)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats,
        lons,
        times,
        flag,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        variable: str = "freeze_thaw",
    ) -> ObservationSeries:
        """Mask fill, reduce the binary frozen field to a basin frozen-fraction.

        *flag* is shaped ``(time, lat, lon)`` carrying the categorical freeze/thaw
        state (``THAWED_CODE`` / ``FROZEN_CODE``). Spec-validated semantics (no
        native to mirror):

        * mask the ``FILL_VALUE`` sentinel and any value outside the
          ``{thawed, frozen}`` set → NaN (excluded from the fraction);
        * map the categorical flag to a binary frozen indicator (1.0 where frozen,
          0.0 where thawed), then reduce over the basin with
          :func:`cos.core.reduce.reduce_grid` — the mean of that binary field IS
          the frozen fraction in ``[0, 1]`` (canonical ``freeze_thaw`` unit, the
          identity boundary scale ``SOURCE_FT_SCALE``);
        * a step with no valid in-basin cell surfaces as MISSING;
        * window-trim to half-open UTC ``[start, end)``.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        flag = np.asarray(flag, dtype="float64")

        # Spec mask: keep only the categorical {thawed, frozen} codes; everything
        # else (the -9999 fill, not-retrieved, non-finite) is excluded.
        valid = np.isfinite(flag) & ((flag == THAWED_CODE) | (flag == FROZEN_CODE))
        # Binary frozen indicator: 1.0 frozen, 0.0 thawed, NaN where not valid.
        frozen = np.where(flag == FROZEN_CODE, 1.0, 0.0)
        frozen = np.where(valid, frozen, np.nan) * SOURCE_FT_SCALE

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("SMAP freeze/thaw basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("SMAP freeze/thaw nearest_cell requires spec.centroid")

        # The mean of the binary frozen field over the basin IS the frozen
        # fraction; reduce_grid emits canonical [0,1] points (GOOD/finite,
        # MISSING/NaN). A single cell (nearest_cell) returns the raw 0/1 flag.
        points = reduce_grid(
            lats, lons, times, frozen,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        # Window-trim, half-open UTC [start, end).
        start_u = _utc(start)
        end_u = _utc(end)
        points = self._trim(points, start_u, end_u)

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "NASA SMAP L3 Freeze/Thaw State",
                "product": "SPL3FTP",
                "source_doi": "10.5067/4DQ54OUIJ9DL",
                "url": "https://nsidc.org/data/spl3ftp",
                "variable": variable,
                "overpass": self._overpass(spec),
                "ft_definition": "mean(flag == frozen) over valid in-basin cells",
            },
            fetched_at=datetime.now(UTC),
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _trim(
        points: list[ObservationPoint], start_u: datetime, end_u: datetime,
    ) -> list[ObservationPoint]:
        return [p for p in points if start_u <= _utc(p.timestamp) < end_u]

    def _overpass(self, spec: ReductionSpec) -> str:
        raw = spec.options.get("overpass") or self.config.get("overpass") or "am"
        return str(raw).lower()

    def _find_variable(self, ds: object, overpass: str) -> str | None:
        """Pick the freeze/thaw flag variable for the requested overpass."""
        data_vars = {str(v) for v in getattr(ds, "data_vars", {})}
        candidates = FT_VARIABLES_PM if overpass == "pm" else FT_VARIABLES_AM
        for name in candidates:
            if name in data_vars:
                return name
        # Fall back to any variable whose name advertises freeze/thaw state.
        token = "_pm" if overpass == "pm" else ""
        for name in data_vars:
            lower = name.lower()
            if ("freeze" in lower or "frozen" in lower or lower == "ft" or "ft_" in lower) and (
                not token or token in lower
            ):
                return name
        return None

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"smap_freeze_thaw:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"smap_freeze_thaw:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"SMAP freeze/thaw over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["SMAPFreezeThawConnector", "FILL_VALUE", "FROZEN_CODE", "THAWED_CODE", "SOURCE_FT_SCALE"]
