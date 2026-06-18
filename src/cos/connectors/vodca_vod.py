# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""VODCA vegetation-optical-depth connector (gridded, basin-reduced).

Exercises the **gridded spatial-reduction path** of the canonical contract for a
dimensionless vegetation index. VODCA (the *Vegetation Optical Depth Climate
Archive*, Moesinger et al. 2020, TU Wien / Zenodo) is a merged multi-band
microwave VOD record distributed as global **daily** gridded NetCDF on a regular
0.25 deg lat/lon grid. This connector:

1. opens a VODCA NetCDF (a local cached file supplied via config ``nc_path`` /
   ``path`` — the Zenodo distribution is a large single archive of yearly /
   per-band files, so COS reduces a *supplied* file rather than downloading the
   archive; the live download is intentionally left unwired);
2. extracts ``lat / lon / time`` and the VOD variable (``vod`` and its band
   variants) as numpy arrays, decoding the product's packed representation at the
   boundary — CF ``scale_factor`` / ``add_offset`` are applied and the
   ``_FillValue`` is masked to NaN (the VODCA files store VOD as a scaled integer
   with a sentinel fill);
3. masks values outside the product's documented physical range
   (``0 <= vod <= VOD_MAX``) so fill / out-of-range cells become NaN and reduce to
   MISSING;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` (cos-lat
   weighted) for larger basins, ``nearest_cell`` for small ones (the size policy
   made explicit and configurable here);
5. emits the canonical ``vod`` unit ``"1"`` (dimensionless). VOD carries no unit
   conversion beyond the scale/offset unpacking, so the boundary conversion is the
   product's own packing factor and then the identity to canonical.

There is **no native SYMFLUENCE VODCA handler**: parity here is *spec-validated*.
The extract→decode→mask→reduce→canonicalize path is hermetically tested against
the published VODCA product spec (packed scale/offset, ``_FillValue`` sentinel,
valid range) on a synthetic in-memory NetCDF, with no network and no auth.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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

#: Candidate VOD variable names, in preference order, mirroring the VODCA band
#: naming (merged record first, then the single-band products).
VOD_VARIABLES = ("vod", "VOD", "vod_merged", "vod_ku", "vod_x", "vod_c")
#: Documented physical upper bound on VOD (dimensionless); values above this (or
#: below zero) are non-physical and are masked, matching the product spec's valid
#: range. VODCA VOD is bounded well below this in practice.
VOD_MAX = 7.0
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("vodca_vod")
class VODCAVODConnector(BaseObservationConnector):
    slug = "vodca_vod"
    display_name = "VODCA Vegetation Optical Depth"
    kind = ObservationKind.VOD
    structural_class = "gridded"
    base_url = "https://zenodo.org"
    auth = frozenset()  # VODCA on Zenodo is anonymous / open access

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
                "VODCA live fetch needs a cached NetCDF (config 'nc_path'/'path'). "
                "The Zenodo distribution is a large single archive, so COS reduces a "
                "SUPPLIED VODCA NetCDF; archive download is not wired. The reduce + "
                "decode + canonicalize path is the proven part.",
            )
        return [self.reduce_file(Path(path), spec, start, end)]

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_file(
        self,
        path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open a VODCA NetCDF, decode + mask, reduce to the basin (dimensionless)."""
        import numpy as np
        import xarray as xr

        # mask_and_scale=False so the connector — not xarray — applies the
        # product's scale_factor / add_offset / _FillValue at the boundary,
        # making the packed-spec conversion explicit and testable (times are
        # still CF-decoded). Equivalent to letting xarray unpack, but kept
        # in-connector for parity-by-construction.
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a VODCA VOD variable (tried {VOD_VARIABLES})",
                )
            da = ds[var_name]
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            raw = np.asarray(da.values, dtype="float64")  # (time, lat, lon), packed
            attrs = dict(da.attrs)

        values = self._decode_and_mask(raw, attrs)

        from cos.core.reduce import reduce_grid

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("VODCA basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("VODCA nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,  # already dimensionless after decode
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
                "source": "VODCA (Vegetation Optical Depth Climate Archive)",
                "source_doi": "10.5281/zenodo.2575599",
                "url": "https://zenodo.org/record/2575599",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    # -- PURE decode/mask helper (no I/O) ------------------------------------

    @staticmethod
    def _decode_and_mask(raw, attrs: dict):
        """Decode the VODCA packing and mask fill / out-of-range cells to NaN.

        *raw* is the packed array straight from the file (``decode_cf=False``).
        Applies, in the product's documented order:

        1. ``_FillValue`` / ``missing_value`` sentinel -> NaN (before unpacking,
           so the sentinel is matched against the stored integer, not a scaled
           float);
        2. ``scale_factor`` / ``add_offset`` unpacking: ``vod = raw*scale + offset``;
        3. the physical valid-range mask ``0 <= vod <= VOD_MAX`` -> NaN.

        Pure / numpy-only so the spec conversion is unit-testable in isolation.
        """
        import numpy as np

        out = np.array(raw, dtype="float64", copy=True)

        fill = attrs.get("_FillValue", attrs.get("missing_value"))
        if fill is not None:
            out = np.where(out == float(fill), np.nan, out)
        out = np.where(~np.isfinite(out), np.nan, out)

        scale = float(attrs.get("scale_factor", 1.0))
        offset = float(attrs.get("add_offset", 0.0))
        if scale != 1.0 or offset != 0.0:
            out = out * scale + offset

        # Physical valid-range mask (spec): non-physical VOD -> MISSING.
        invalid = ~np.isfinite(out) | (out < 0.0) | (out > VOD_MAX)
        out = np.where(invalid, np.nan, out)
        return cast("np.ndarray", out)

    def _find_variable(self, ds: object) -> str | None:
        """Pick the VOD variable, merged record preferred (VODCA band order)."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in VOD_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            if "vod" in str(name).lower():
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
            site_id = f"vodca_vod:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"vodca_vod:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"VODCA VOD over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
