# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""OCO-3 solar-induced fluorescence (SIF) connector (gridded, basin-reduced).

OCO-3 SIF has **no** SYMFLUENCE native handler, so this connector is
*spec-validated*: its scale, valid range, and fill semantics reproduce the
published OCO-3 Lite SIF product spec (``OCO3_L2_Lite_SIF``, served as NetCDF
behind NASA GES DISC / Earthdata), and the hermetic tests assert that contract on
a synthetic fixture rather than against a native reference series. It complements
the (also spec-validated) :mod:`cos.connectors.tropomi_sif` connector: TROPOMI SIF
is retrieved near 740/743 nm, OCO-3 reports SIF at the 757 nm and 771 nm Fraunhofer
windows, giving an orthogonal, finer-resolution SIF constraint over the same
canonical ``sif`` unit.

Product (OCO3_L2_Lite_SIF, GES DISC, NetCDF behind NASA Earthdata netrc):

* the per-sounding SIF fields are ``SIF_757nm`` and ``SIF_771nm``, reported in
  **W/m²/sr/µm** (radiance per unit wavelength), *not* the COS canonical ``sif``
  unit ``mW/m²/nm/sr`` (:data:`cos.core.models.KIND_UNITS`). The boundary scale
  converts source→canonical exactly once here:

      mW/m²/nm/sr = (W → mW : ×1000) × (per-µm → per-nm : ÷1000) × W/m²/sr/µm

  The two factors cancel numerically (:data:`SOURCE_SIF_SCALE` ``= 1.0``), but the
  conversion is documented and applied at the boundary so a future product with a
  different source unit has exactly one place to change it. To put the 757/771 nm
  windows onto a single comparable SIF magnitude the connector also applies the
  published daily-correction-free 740 nm reference combination
  ``SIF_740 ≈ 0.5 * (SIF_757 + 1.5 * SIF_771)`` (the OCO-2/3 linear 740 nm proxy),
  selectable via the ``sif_combine`` option (default), or a single named field.
* the no-retrieval fill is ``-999999`` (:data:`SIF_FILL_VALUE`); cells / soundings
  equal to the fill, non-finite, or outside the physical valid band
  (:data:`VALID_SIF_RANGE`, mW/m²/nm/sr) are masked to NaN so they reduce to
  :class:`~cos.core.models.QualityFlag.MISSING`.

This connector:

1. opens an OCO-3 Lite SIF NetCDF (a local cached file supplied via config
   ``nc_path`` / ``path`` — GES DISC/Earthdata download is not wired here; the
   reduce + canonicalize path is the proven part);
2. extracts ``lat / lon / time`` and the SIF field(s) as numpy arrays;
3. masks fill / out-of-range soundings, combines 757/771 nm onto a 740 nm-like
   magnitude, applies the (identity) source→canonical scale at the boundary;
4. reduces to the basin via :mod:`cos.core.reduce` — ``basin_mean`` (cos-lat
   weighted) for larger basins, ``nearest_cell`` for small ones — and emits the
   canonical ``sif`` unit ``mW/m2/nm/sr``.

The real OCO-3 Lite product stores SIF on a **1-D ``sounding_dim``** (orbit
footprints, not a regular raster), with ``Latitude``/``Longitude`` as *data_vars*
(no xarray coords, no ``time`` variable) and a ≈ -9e30 ``missing_value`` fill.
:meth:`reduce_file` detects that layout and takes :meth:`_reduce_sounding_file`,
which reshapes the soundings onto the dedicated 2-D-coordinate reduction path
(bbox cell-mask + cos-lat-weighted mean / nearest sounding). A pre-gridded file
with explicit 1-D/2-D ``lat``/``lon`` and the 757/771 nm windows is also supported
via the 1-D :func:`cos.core.reduce.reduce_grid` path (any ``(lat, lon, time)`` dim
order is normalized to ``(time, lat, lon)``), mirroring
:mod:`cos.connectors.tropomi_sif` / :mod:`cos.connectors.amsr_swe`.

The architecture-critical extract→mask→combine→scale→reduce→canonicalize path is
hermetically tested via :meth:`OCO3SIFConnector.reduce_arrays` on a synthetic
in-memory grid, with no network, no auth, and no NetCDF dependency.
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
    ObservationSeries,
    ReductionSpec,
    SiteRef,
    SpatialReduction,
)
from cos.core.registry import register

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr

logger = structlog.get_logger()

#: Published OCO-3 Lite SIF fill / no-retrieval sentinel.
SIF_FILL_VALUE = -999999.0
#: Source→canonical scale. Source SIF fields are W/m²/sr/µm; canonical ``sif`` is
#: mW/m²/nm/sr. (W→mW ×1000) × (per-µm→per-nm ÷1000) = 1.0, so the boundary scale
#: is numerically the identity; written explicitly as the single conversion site.
W_TO_MW = 1000.0
PER_UM_TO_PER_NM = 1.0 / 1000.0
SOURCE_SIF_SCALE = W_TO_MW * PER_UM_TO_PER_NM  # == 1.0
#: Physical-plausibility band for SIF radiance (mW/m²/nm/sr). OCO-3 757/771 nm SIF
#: spans roughly 0..3; a small negative tail is a legitimate retrieval artefact,
#: so the lower bound allows mildly negative values while masking gross outliers.
VALID_SIF_RANGE = (-2.0, 12.0)
#: 740 nm linear combination of the OCO 757/771 nm windows (the published OCO-2/3
#: proxy for a single comparable SIF magnitude): SIF_740 = 0.5*(SIF_757 + 1.5*SIF_771).
SIF_771_WEIGHT = 1.5
SIF_COMBINE_SCALE = 0.5
#: Candidate per-window SIF variable names, in preference order (OCO3_L2_Lite_SIF).
SIF_757_VARIABLES = ("SIF_757nm", "sif_757nm", "SIF_757", "Daily_SIF_757nm")
SIF_771_VARIABLES = ("SIF_771nm", "sif_771nm", "SIF_771", "Daily_SIF_771nm")
#: Candidate already-combined / single-field SIF names, in preference order. The
#: real OCO3_L2_Lite_SIF granule carries a combined ``SIF_740nm`` field; it is
#: preferred over re-combining the daily-corrected ``Daily_SIF_757/771nm`` windows.
SIF_SINGLE_VARIABLES = ("SIF_740nm", "sif_740nm", "SIF_740", "sif", "SIF")
#: Per-sounding coordinate variable names (the real Lite product stores these as
#: *data_vars*, capitalized ``Latitude``/``Longitude`` on a 1-D ``sounding_dim``,
#: with no xarray coords and no ``time`` variable).
SIF_LAT_VARIABLES = ("lat", "Latitude", "latitude")
SIF_LON_VARIABLES = ("lon", "Longitude", "longitude")
#: Real OCO-3 Lite fill sentinel, carried on the SIF variable's ``missing_value``
#: attr (≈ -9e30) — distinct from the gridded path's -999999 :data:`SIF_FILL_VALUE`.
SIF_FILL_MAGNITUDE = 1.0e29
#: <= this area (km²) defaults to nearest_cell; larger uses basin_mean.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0


@register("oco3_sif")
class OCO3SIFConnector(BaseObservationConnector):
    slug = "oco3_sif"
    display_name = "OCO-3 Solar-Induced Fluorescence (OCO3_L2_Lite_SIF)"
    kind = ObservationKind.SIF
    structural_class = "gridded"
    base_url = "https://oco2.gesdisc.eosdis.nasa.gov"
    auth = frozenset({"earthdata"})

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """One reduced region: the basin (or its centroid sounding)."""
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
                "OCO-3 SIF live fetch needs a cached NetCDF (config 'nc_path'/'path') — "
                "an OCO3_L2_Lite_SIF granule. GES DISC/Earthdata download is not yet "
                "wired; the combine + scale + reduce + canonicalize path is the proven part.",
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
        """Open an OCO-3 Lite SIF NetCDF, extract arrays, then combine/scale/reduce."""
        import numpy as np
        import xarray as xr

        combine = bool(spec.options.get("sif_combine", True))
        with xr.open_dataset(path, mask_and_scale=False) as ds:
            # Real OCO3_L2_Lite_SIF is 1-D per-sounding: Latitude/Longitude are
            # *data_vars* on a sounding_dim, with no xarray coords and no `time`.
            # Detect that layout and take the dedicated per-sounding reader (uses
            # the combined SIF_740nm, not a re-combine of the daily windows);
            # otherwise fall back to the gridded (time, lat, lon) reader.
            slat = _first_var(ds, SIF_LAT_VARIABLES)
            slon = _first_var(ds, SIF_LON_VARIABLES)
            if (
                slat is not None
                and slon is not None
                and ds[slat].ndim == 1
                and "sounding_dim" in tuple(str(d) for d in ds[slat].dims)
            ):
                return self._reduce_sounding_file(ds, slat, slon, spec, start, end)

            lat_name = "lat" if "lat" in ds else _coord_like(ds, "lat")
            lon_name = "lon" if "lon" in ds else _coord_like(ds, "lon")
            time_name = "time" if "time" in ds else _coord_like(ds, "time")

            v757 = _first_present(ds, SIF_757_VARIABLES)
            v771 = _first_present(ds, SIF_771_VARIABLES)
            vsingle = _first_present(ds, SIF_SINGLE_VARIABLES)

            if combine and v757 is not None and v771 is not None:
                da757 = _to_time_lat_lon(ds[v757], time_name, lat_name, lon_name)
                da771 = _to_time_lat_lon(ds[v771], time_name, lat_name, lon_name)
                sif_757 = np.asarray(da757.values, dtype="float64")
                sif_771 = np.asarray(da771.values, dtype="float64")
                values = self._combine_757_771(sif_757, sif_771)
                var_name = f"{v757}+{v771}->SIF_740"
            elif vsingle is not None:
                da = _to_time_lat_lon(ds[vsingle], time_name, lat_name, lon_name)
                values = np.asarray(da.values, dtype="float64")
                var_name = vsingle
            else:
                raise ConnectorError(
                    self.slug,
                    "NetCDF missing OCO-3 SIF variables (tried 757/771 pair "
                    f"{SIF_757_VARIABLES}/{SIF_771_VARIABLES} and single "
                    f"{SIF_SINGLE_VARIABLES})",
                )

            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds[time_name].values)
        return self.reduce_arrays(
            lats, lons, times, values, spec, start, end, var_name=var_name, already_combined=True
        )

    def _reduce_sounding_file(
        self,
        ds: object,
        lat_name: str,
        lon_name: str,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Read the real 1-D per-sounding OCO3_L2_Lite_SIF layout.

        The published product ships a single combined ``SIF_740nm`` field (do NOT
        re-combine the daily-corrected windows), with ``Latitude``/``Longitude`` as
        1-D ``(sounding_dim,)`` data_vars and a ``missing_value`` fill of ≈ -9e30.
        Soundings are reshaped onto the 2-D reduction path (one synthetic time row,
        ``(1, n)`` coords) so the bbox cell-mask selects the in-domain footprints.
        """
        import numpy as np

        vsingle = _first_var(ds, SIF_SINGLE_VARIABLES, in_data_vars=True)
        if vsingle is None:
            raise ConnectorError(
                self.slug,
                f"OCO-3 Lite NetCDF missing a combined SIF field {SIF_SINGLE_VARIABLES}",
            )
        da = ds[vsingle]  # type: ignore[index]
        values = np.asarray(da.values, dtype="float64")
        # Honour the product fill (missing_value / _FillValue attr; ≈ -9e30, not the
        # -999999 the gridded path expects), then a magnitude guard for robustness.
        for attr in ("missing_value", "_FillValue"):
            fv = da.attrs.get(attr)
            if fv is not None:
                values = np.where(values == float(fv), np.nan, values)
        values = np.where(np.abs(values) >= SIF_FILL_MAGNITUDE, np.nan, values)

        lats = np.asarray(ds[lat_name].values, dtype="float64").reshape(1, -1)  # type: ignore[index]
        lons = np.asarray(ds[lon_name].values, dtype="float64").reshape(1, -1)  # type: ignore[index]
        values = values.reshape(1, 1, -1)
        times = np.array([np.datetime64(_granule_day(ds, start))])
        return self.reduce_arrays(
            lats, lons, times, values, spec, start, end,
            var_name=vsingle, already_combined=True,
        )

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_arrays(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        times: np.ndarray,
        sif: np.ndarray,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
        *,
        var_name: str = "SIF_740",
        already_combined: bool = True,
    ) -> ObservationSeries:
        """Mask fill/out-of-range, scale source→canonical, reduce, window-trim.

        *sif* is shaped ``(time, lat, lon)`` SIF radiance in the source unit
        (W/m²/sr/µm), either an already-combined 740 nm-like field (the default;
        ``already_combined=True``) or a single window. Cells equal to
        :data:`SIF_FILL_VALUE`, non-finite, or outside :data:`VALID_SIF_RANGE`
        (evaluated in the canonical unit) become NaN and surface as MISSING; the
        rest are multiplied by :data:`SOURCE_SIF_SCALE` (numeric identity) so the
        canonical unit ``mW/m2/nm/sr`` is preserved inside the reduction.

        Coordinate shape is honoured: 1-D ``lat``/``lon`` vectors defer to
        :func:`cos.core.reduce.reduce_grid`; 2-D per-sounding lat/lon (the real
        OCO-3 Lite footprint layout) take a dedicated bbox-mask reduction path.
        """
        import numpy as np

        from cos.core.reduce import reduce_grid, reduce_grid_2d

        lats = np.asarray(lats, dtype="float64")
        lons = np.asarray(lons, dtype="float64")
        values = np.asarray(sif, dtype="float64")

        # Apply the source→canonical scale at the boundary (W/m²/sr/µm ->
        # mW/m²/nm/sr; numerically the identity). Mask the fill sentinel before the
        # finite/range test so -999999 never sneaks into the canonical band.
        values = np.where(values == SIF_FILL_VALUE, np.nan, values * SOURCE_SIF_SCALE)
        lo, hi = VALID_SIF_RANGE
        invalid = ~np.isfinite(values) | (values < lo) | (values > hi)
        values = np.where(invalid, np.nan, values)

        reduction = self._choose_reduction(spec)
        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("OCO-3 SIF basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("OCO-3 SIF nearest_cell requires spec.centroid")

        if lats.ndim == 2 or lons.ndim == 2:
            # Real OCO-3 Lite product: 2-D per-sounding lat/lon. reduce_grid assumes
            # 1-D coord vectors (it indexes lat/lon axes independently), which
            # IndexErrors on a 2-D footprint grid -> reduce over a bbox cell-mask.
            points = reduce_grid_2d(
                lats, lons, times, values,
                reduction=reduction, bbox=bbox, point=point, grid_label="OCO-3 sounding",
            )
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
                "source": "OCO-3 Lite SIF",
                "product": "OCO3_L2_Lite_SIF",
                "source_doi": "10.5067/NOD1DPPBCXSO",
                "url": "https://disc.gsfc.nasa.gov/datasets/OCO3_L2_Lite_SIF_11r",
                "variable": var_name,
                "complements": "tropomi_sif",
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _combine_757_771(sif_757: np.ndarray, sif_771: np.ndarray) -> np.ndarray:
        """740 nm-like SIF from the OCO 757/771 nm windows (source unit, pre-scale).

        ``SIF_740 = 0.5 * (SIF_757 + 1.5 * SIF_771)`` — the published OCO-2/3 linear
        proxy. The fill sentinel is preserved on either input so the downstream mask
        still drops it: if either window is fill/non-finite the combined value is the
        fill sentinel, which :meth:`reduce_arrays` then masks to MISSING.
        """
        import numpy as np

        bad = (
            (sif_757 == SIF_FILL_VALUE)
            | (sif_771 == SIF_FILL_VALUE)
            | ~np.isfinite(sif_757)
            | ~np.isfinite(sif_771)
        )
        combined = SIF_COMBINE_SCALE * (sif_757 + SIF_771_WEIGHT * sif_771)
        return np.where(bad, SIF_FILL_VALUE, combined)


    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"oco3_sif:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"oco3_sif:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"OCO-3 SIF over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _coord_like(ds: object, want: str) -> str:
    for name in getattr(ds, "coords", {}):
        if want in str(name).lower():
            return str(name)
    return want


def _first_present(ds: object, names: tuple[str, ...]) -> str | None:
    """First variable in *names* present in the dataset's data_vars, else None."""
    data_vars = set(getattr(ds, "data_vars", {}))
    for name in names:
        if name in data_vars:
            return name
    return None


def _first_var(ds: object, names: tuple[str, ...], *, in_data_vars: bool = False) -> str | None:
    """First name present among the dataset's variables (data_vars + coords), else None.

    Unlike :func:`_first_present`, this also looks at coords/variables so the real
    Lite product's capitalized ``Latitude``/``Longitude`` data_vars are found.
    """
    pool = set(getattr(ds, "data_vars", {}))
    if not in_data_vars:
        pool |= set(getattr(ds, "variables", {}))
    for name in names:
        if name in pool:
            return name
    return None


def _granule_day(ds: object, fallback: datetime) -> str:
    """Granule day ``YYYY-MM-DD`` for the per-granule daily series timestamp.

    Prefer a product start-date global attr; fall back to the request start (the
    OCO-3 Lite SIF product is daily, so a single per-granule day is sufficient).
    """
    attrs = getattr(ds, "attrs", {})
    for attr in ("RangeBeginningDate", "time_coverage_start"):
        val = attrs.get(attr)
        if isinstance(val, str) and len(val) >= 10 and val[4] == "-":
            return val[:10]
    return fallback.strftime("%Y-%m-%d")


def _to_time_lat_lon(
    da: xr.DataArray, time_name: str, lat_name: str, lon_name: str
) -> xr.DataArray:
    """Transpose a SIF DataArray to ``(time, lat, lon)`` by its own dim names.

    Some OCO-3 Lite distributions ship ``(lat, lon, time)`` while
    :func:`cos.core.reduce.reduce_grid`/``basin_mean`` index ``(time, lat, lon)``.
    Reorder only the dims that exist (a single-time 2-D grid has no time dim),
    keeping any unexpected leading dims ahead of the canonical trailing axes.
    """
    dims = tuple(str(d) for d in da.dims)
    wanted = [d for d in (time_name, lat_name, lon_name) if d in dims]
    if not wanted:
        return da
    leading = [d for d in dims if d not in wanted]
    order = leading + wanted
    if order == list(dims):
        return da
    return da.transpose(*order)
