# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""ECOSTRESS L3 ET PT-JPL connector (gridded, basin-reduced).

ECOSTRESS (ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station)
has **no** SYMFLUENCE native handler, so this connector is *spec-validated*: its
scale, valid range, fill semantics, and the latent-heat → water-flux conversion
reproduce the published LP DAAC ECO3ETPTJPL.001 product spec, and the hermetic
tests assert that contract on a synthetic inline fixture rather than against a
native reference series.

Product (ECOSTRESS L3 ET PT-JPL, ``ECO3ETPTJPL.001``, ~70 m, served as HDF5/GeoTIFF
behind LP DAAC / NASA Earthdata):

* The PT-JPL algorithm retrieves an **instantaneous latent heat flux** at the
  ISS overpass, distributed as ``ETinst`` in **W/m²** (energy flux), alongside an
  upscaled **daily** evapotranspiration ``ETdaily`` in **mm/day** (a water flux).
  COS's canonical ``et`` unit is ``mm/day`` (:data:`cos.core.models.KIND_UNITS`).
* Boundary conversion (documented, applied here and nowhere else):
    - a source already in **mm/day** (``ETdaily``) is the identity
      (:data:`SOURCE_MM_PER_DAY` path);
    - a source in **W/m²** latent heat (``ETinst``) is converted to an
      equivalent water depth per day by dividing by the latent heat of
      vaporization and the density of water, then scaling to a day:

          mm/day = LE [W/m²] / (λ [J/kg] · ρ_w [kg/m³]) · 86400 [s/day] · 1000 [mm/m]

      With λ = :data:`LATENT_HEAT_VAPORIZATION` J/kg and ρ_w =
      :data:`WATER_DENSITY` kg/m³, the combined factor is
      :data:`WM2_TO_MM_PER_DAY` mm/day per W/m² (≈ 0.0353). This is the standard
      W/m² → mm/day latent-heat conversion used for instantaneous ET flux. Note
      ``ETinst`` is an instantaneous (overpass-time) flux, not a 24-h mean, so the
      W/m² path yields an *instantaneous-rate* mm/day; ``ETdaily`` is preferred and
      auto-picked whenever present. When an instantaneous flux
      (:data:`_INSTANTANEOUS_ET_VARS`) is the only ET signal, the conversion is
      still applied but a warning is logged and ``source_info["instantaneous_scaled"]``
      is set ``"true"`` so consumers can down-weight the (over-)estimate.
* The no-retrieval fill is ``-9999`` (:data:`ET_FILL_VALUE`); cells equal to the
  fill, non-finite, negative, or outside the physical valid band
  (:data:`VALID_ET_RANGE_MM_DAY`, mm/day, applied **after** conversion) are masked
  to NaN so they reduce to :class:`~cos.core.models.QualityFlag.MISSING`.

This connector:

1. opens an ECOSTRESS ET NetCDF (a local cached file supplied via config
   ``nc_path`` / ``path`` — LP DAAC / Earthdata download is not wired here, the
   reduce + canonicalize path is the proven part);
2. extracts ``lat / lon / time`` and the ET variable as numpy arrays, normalizing
   any dim ordering to ``(time, lat, lon)`` (the real product is tiled and may be
   served ``(lat, lon, time)`` or carry 2-D swath coordinates);
3. converts source units → canonical ``mm/day`` at the boundary, then masks
   fill / out-of-range cells;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` (cos-lat
   weighted) for larger basins, ``nearest_cell`` for small ones.

The architecture-critical extract→convert→mask→reduce→canonicalize path is
hermetically tested via :meth:`ECOSTRESSETConnector.reduce_arrays` on a synthetic
in-memory grid, with no network, no auth, and no NetCDF dependency. 2-D-coordinate
(swath) grids take a dedicated reduction path, mirroring the AMSR2 SWE connector.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import xarray as xr

logger = structlog.get_logger()

#: Published ECOSTRESS L3 ET PT-JPL fill / no-retrieval sentinel.
ET_FILL_VALUE = -9999.0
#: Latent heat of vaporization of water (J/kg) at ~20 °C; the standard constant
#: used for the latent-heat-flux → water-depth conversion.
LATENT_HEAT_VAPORIZATION = 2.45e6
#: Density of liquid water (kg/m³).
WATER_DENSITY = 1000.0
#: Seconds per day (W/m² is per-second; canonical ET is per-day).
SECONDS_PER_DAY = 86400.0
#: Combined factor: mm/day of water flux per W/m² of latent heat flux.
#: = (86400 s/day) / (λ J/kg · ρ_w kg/m³) · 1000 mm/m  ≈ 0.03526 mm/day per W/m².
WM2_TO_MM_PER_DAY = SECONDS_PER_DAY / (LATENT_HEAT_VAPORIZATION * WATER_DENSITY) * 1000.0
#: Identity factor for a source already in mm/day.
SOURCE_MM_PER_DAY = 1.0
#: Physical-plausibility band for daily ET (mm/day), applied AFTER conversion.
#: Open-water/irrigated peaks rarely exceed ~15 mm/day; the upper bound also
#: catches unconverted fill that slipped the sentinel test.
VALID_ET_RANGE_MM_DAY = (0.0, 30.0)
#: Source-unit strings (case/space-insensitive) that mean "W/m² latent heat
#: flux, convert via WM2_TO_MM_PER_DAY".
_WM2_UNITS = {"w/m2", "w/m^2", "wm-2", "w m-2", "watt/m2"}
#: Candidate ET variable names, in preference order. Daily mm/day (``ETdaily``)
#: is preferred over the instantaneous W/m² ``ETinst`` because it is a true daily
#: total in the canonical unit.
ET_VARIABLES = ("ETdaily", "ETcanopy", "ET", "et", "ETinst", "et_inst", "ETPTJPL")
#: ET variable names that are an INSTANTANEOUS (overpass-time) W/m² latent-heat
#: flux. Scaling these to mm/day by 86400 s assumes the overpass rate is sustained
#: 24 h, which OVERESTIMATES the daily total — so when one of these is the only ET
#: signal present (``ETdaily`` is preferred and would be auto-picked if present),
#: the conversion is still applied but a warning is emitted and provenance is
#: stamped (``source_info["instantaneous_scaled"]``) so consumers can down-weight.
_INSTANTANEOUS_ET_VARS = frozenset({
    "ETinst", "et_inst", "PTJPLSMinst", "STICinst", "BESSinst", "MOD16inst",
})
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("ecostress_et")
class ECOSTRESSETConnector(BaseObservationConnector):
    slug = "ecostress_et"
    display_name = "ECOSTRESS L3 ET PT-JPL"
    kind = ObservationKind.ET
    structural_class = "gridded"
    base_url = "https://e4ftl01.cr.usgs.gov"
    auth = frozenset({"earthdata"})  # LP DAAC ECO3ETPTJPL download needs Earthdata

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
                "ECOSTRESS ET live fetch needs a cached NetCDF (config 'nc_path'/'path') "
                "or LP DAAC / Earthdata download (not yet wired). The convert + reduce + "
                "canonicalize path is the proven part; supply an ECO3ETPTJPL.001 NetCDF.",
            )
        return [self.reduce_file(Path(path), spec, start, end)]

    # -- file reader (extract arrays, then defer to the pure core) -----------

    def reduce_file(
        self,
        path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open an ECOSTRESS ET NetCDF, extract arrays, convert + reduce + canonicalize."""
        import numpy as np
        import xarray as xr

        with xr.open_dataset(path) as ds:
            var_name = self._find_variable(ds)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing an ECOSTRESS ET variable (tried {ET_VARIABLES})",
                )
            da = ds[var_name]
            source_units = str(self.config.get("source_units") or da.attrs.get("units", "")).strip()
            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            time_name = "time" if "time" in ds else _coord_like(ds, "time")
            # The real product is tiled and may be served (lat, lon, time) or
            # (time, lat, lon); reduce_grid / basin_mean require (time, lat, lon).
            # Transpose by the dataset's own dim names so any ordering normalizes.
            da = _to_time_lat_lon(da, time_name, lat_name, lon_name)
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds[time_name].values)
            values = np.asarray(da.values, dtype="float64")
        return self.reduce_arrays(
            lats, lons, times, values, spec, start, end,
            var_name=var_name, source_units=source_units,
        )

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats,
        lons,
        times,
        et,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        var_name: str = "ETdaily",
        source_units: str = "",
    ) -> ObservationSeries:
        """Convert source→mm/day, mask fill/out-of-range, basin-reduce, window-trim.

        *et* is shaped ``(time, lat, lon)`` ET in the source unit (W/m² latent heat
        for ``ETinst``, or mm/day for ``ETdaily``). The boundary conversion is
        applied first (W/m² → mm/day via :data:`WM2_TO_MM_PER_DAY`, else identity),
        then cells equal to :data:`ET_FILL_VALUE`, non-finite, negative, or outside
        :data:`VALID_ET_RANGE_MM_DAY` become NaN and surface as MISSING.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        values = np.asarray(et, dtype="float64")

        # Detect fill BEFORE the multiplicative conversion (the -9999 sentinel is
        # in source units; scaling it would move the sentinel and break the test).
        is_fill = (values == ET_FILL_VALUE) | ~np.isfinite(values)

        factor, resolved_units = self._conversion_factor(source_units)
        # An instantaneous overpass-time W/m² flux scaled to mm/day overestimates
        # the daily total (ETdaily is preferred and auto-picked whenever present).
        instantaneous = resolved_units == "W/m2" and var_name in _INSTANTANEOUS_ET_VARS
        if instantaneous:
            logger.warning(
                "ecostress_et.instantaneous_flux_scaled",
                variable=var_name,
                detail="instantaneous (overpass-time) W/m² latent-heat flux scaled to "
                       "mm/day assumes the rate holds 24 h and OVERESTIMATES daily ET; "
                       "supply ETdaily for a true daily total.",
            )
        values = values * factor  # source -> canonical mm/day at the boundary

        lo, hi = VALID_ET_RANGE_MM_DAY
        invalid = is_fill | (values < lo) | (values > hi)
        values = np.where(invalid, np.nan, values)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("ECOSTRESS ET basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("ECOSTRESS ET nearest_cell requires spec.centroid")

        if lats.ndim == 2 or lons.ndim == 2:
            # Real swath/tile product can carry 2-D lat/lon. reduce_grid assumes
            # 1-D coord vectors (it indexes lat/lon axes independently), which
            # IndexErrors on a 2-D grid -> reduce over a bbox cell-mask instead.
            points = self._reduce_grid_2d(lats, lons, times, values, reduction, bbox, point)
        else:
            points = reduce_grid(
                lats, lons, times, values,
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
                "source": "ECOSTRESS L3 ET PT-JPL",
                "product": "ECO3ETPTJPL.001",
                "source_doi": "10.5067/ECOSTRESS/ECO3ETPTJPL.001",
                "url": "https://lpdaac.usgs.gov/products/eco3etptjplv001/",
                "variable": var_name,
                "source_units": resolved_units,
                "instantaneous_scaled": str(instantaneous).lower(),
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _conversion_factor(source_units: str) -> tuple[float, str]:
        """Resolve the source→mm/day factor and the recognized source-unit label.

        W/m² latent-heat flux divides by λ·ρ_w and scales to a day; anything else
        (mm/day, mm day-1, unlabeled — the ECO3ETPTJPL daily ET default) is the
        identity. Returns ``(factor, resolved_units)``.
        """
        normalized = source_units.lower().replace(" ", "")
        if normalized in {u.replace(" ", "") for u in _WM2_UNITS}:
            return WM2_TO_MM_PER_DAY, "W/m2"
        return SOURCE_MM_PER_DAY, source_units or "mm/day"

    def _reduce_grid_2d(
        self,
        lats,
        lons,
        times,
        values,
        reduction: SpatialReduction,
        bbox: tuple[float, float, float, float] | None,
        point: tuple[float, float] | None,
    ) -> list[ObservationPoint]:
        """Reduce a 2-D-coordinate (swath/tile) product to canonical points.

        ``lats``/``lons`` are 2-D (ny, nx); ``values`` is (time, ny, nx). Off-grid
        cells carry non-finite lat/lon and drop out of the bbox mask. ``basin_mean``
        is the cos-lat-weighted mean over the bbox cells; ``nearest_cell`` is the
        nearest valid in-grid cell to the centroid.
        """
        import numpy as np

        from cos.core.reduce import _as_datetime

        lats = np.broadcast_to(np.asarray(lats, dtype="float64"), values.shape[1:])
        lons = np.broadcast_to(np.asarray(lons, dtype="float64"), values.shape[1:])
        finite_coord = np.isfinite(lats) & np.isfinite(lons)

        if reduction == SpatialReduction.BASIN_MEAN:
            if bbox is None:
                raise ReductionError("ECOSTRESS ET basin_mean requires spec.bbox")
            lat_min, lon_min, lat_max, lon_max = bbox
            cell_mask = (
                finite_coord
                & (lats >= lat_min) & (lats <= lat_max)
                & (lons >= lon_min) & (lons <= lon_max)
            )
            if not cell_mask.any():
                raise ReductionError(
                    f"No grid cells inside bbox {bbox} on the 2-D coordinate grid"
                )
            weights = np.cos(np.deg2rad(np.where(cell_mask, lats, 0.0)))
            series = np.full(values.shape[0], np.nan, dtype="float64")
            for t in range(values.shape[0]):
                layer = values[t]
                use = cell_mask & np.isfinite(layer)
                if not use.any():
                    continue
                wsum = float(np.sum(weights[use]))
                if wsum > 0:
                    series[t] = float(np.sum(layer[use] * weights[use]) / wsum)
        else:
            if point is None:
                raise ReductionError("ECOSTRESS ET nearest_cell requires spec.centroid")
            plat, plon = point
            dist = np.where(
                finite_coord,
                (lats - plat) ** 2 + (lons - plon) ** 2,
                np.inf,
            )
            flat_idx = int(np.argmin(dist))
            i, j = np.unravel_index(flat_idx, dist.shape)
            series = values[:, i, j].astype("float64")

        points: list[ObservationPoint] = []
        for t, v in zip(times, series):
            ts = t if isinstance(t, datetime) else _as_datetime(t)
            finite = v is not None and np.isfinite(v)
            points.append(
                ObservationPoint(
                    timestamp=ts,
                    value=float(v) if finite else None,
                    quality=QualityFlag.GOOD if finite else QualityFlag.MISSING,
                )
            )
        return points

    def _find_variable(self, ds: object) -> str | None:
        """Pick the ET variable in the published preference order (daily first)."""
        cfg_var = self.config.get("variable")
        data_vars = set(getattr(ds, "data_vars", {}))
        if cfg_var and cfg_var in data_vars:
            return str(cfg_var)
        for name in ET_VARIABLES:
            if name in data_vars:
                return name
        for name in data_vars:
            low = str(name).lower()
            if "et" in low and "qc" not in low and "quality" not in low:
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
            site_id = f"ecostress_et:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"ecostress_et:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"ECOSTRESS ET over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _coord_like(ds: object, want: str) -> str:
    for name in getattr(ds, "coords", {}):
        if want in str(name).lower():
            return str(name)
    return want


def _to_time_lat_lon(
    da: xr.DataArray, time_name: str, lat_name: str, lon_name: str
) -> xr.DataArray:
    """Transpose an ET DataArray to ``(time, lat, lon)`` by its own dim names.

    A tiled ECOSTRESS product may ship ``(lat, lon, time)`` while
    :func:`cos.core.reduce.basin_mean`/``nearest_cell`` index ``(time, lat, lon)``.

    When ``lat``/``lon`` are 2-D *coordinates* (the real tiled/swath products from
    GDAL/rioxarray ride them on dims named ``y``/``x``, not ``lat``/``lon``), the
    spatial axes to reorder are those underlying *dimensions* — otherwise ``y``/``x``
    look like unexpected leading dims and ``time`` gets pushed to the trailing axis,
    breaking the downstream ``(time, lat, lon)`` reduction. Reorder only the dims
    that exist (a 2-D single-time grid has no time dim).
    """
    dims = tuple(str(d) for d in da.dims)
    coords = getattr(da, "coords", {})
    spatial: list[str] = []
    for coord in (lat_name, lon_name):
        if coord in coords and getattr(da[coord], "ndim", 1) == 2:
            spatial.extend(str(d) for d in da[coord].dims)
        elif coord in dims:
            spatial.append(coord)
    seen: dict[str, None] = {}
    for d in spatial:
        seen.setdefault(d, None)
    wanted = ([time_name] if time_name in dims else []) + [d for d in seen if d in dims]
    if not wanted:
        return da
    leading = [d for d in dims if d not in wanted]
    order = leading + wanted
    if order == list(dims):
        return da
    return da.transpose(*order)
