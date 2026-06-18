# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""MODIS FAPAR connector (gridded, basin-reduced).

Ports SYMFLUENCE's native ``modis_lai`` / ``mcd15`` observation handler
(``data/observation/handlers/modis_lai.py``) — the LAI+FAPAR handler — onto the
COS canonical contract for the FAPAR kind. The native handler ingests MODIS
MCD15A2H (combined Terra+Aqua) / MOD15A2H (Terra) / MYD15A2H (Aqua)
``Fpar_500m`` 8-day-composite rasters and produces a basin-mean Fraction of
Absorbed Photosynthetically Active Radiation time series.

Source semantics mirrored exactly from the native handler (native-parity):

* **variable**: ``Fpar_500m`` (priority order ``Fpar_500m`` → ``FPAR`` →
  ``fpar`` → ``Fpar``, the native ``_find_variable`` order);
* **valid range**: FAPAR digital number in ``[0, 100]`` (native
  ``FPAR_VALID_RANGE``). The fill byte (255) and every DN outside the valid
  range are masked to NaN — these surface as ``QualityFlag.MISSING`` in the
  canonical series;
* **scale factor**: source DN × ``0.01`` → FAPAR fraction in ``[0, 1]`` (the
  native ``FPAR_SCALE_FACTOR``). FAPAR is a dimensionless fraction; the
  canonical ``fapar`` unit is ``"1"``, so applying the 0.01 scale factor at the
  boundary delivers the canonical value;
* **QC**: the native ``_apply_qc_filter`` keeps only ``FparLai_QC`` algorithm
  paths in ``{0, 2}`` (bits 5-7: main=0, saturation=2). Applied here when a QC
  layer is supplied alongside the FAPAR grid;
* **reduction**: spatial mean over the grid (basin-mean over the bbox here, the
  COS gridded path); ``nearest_cell`` for small basins, matching grace.py's
  size policy.

As with grace.py / smap_sm.py / modis_lai.py, the Earthdata download path is not
wired here; the proven part is the reduce + canonicalize path, exercised
hermetically against a supplied NetCDF (config ``nc_path`` / ``path``).
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
    import numpy as np

logger = structlog.get_logger()

#: FAPAR digital-number valid range; everything else (incl. 255 fill) is masked.
VALID_FAPAR_RANGE = (0.0, 100.0)
#: MODIS FAPAR fill byte -> masked to NaN.
FAPAR_FILL_VALUE = 255.0
#: DN -> FAPAR fraction in [0, 1] (native FPAR_SCALE_FACTOR). FAPAR is the
#: dimensionless canonical "1".
FAPAR_SCALE_FACTOR = 0.01
#: FparLai_QC: bits 5-7 = algorithm path; accept main (0) and saturation (2).
QC_ALGORITHM_SHIFT = 5
QC_ALGORITHM_MASK = 0b111
QC_GOOD_ALGORITHMS = frozenset({0, 2})
#: <= this area (km²) defaults to nearest_cell, mirroring grace.py's policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: FAPAR variable names, native-handler priority order.
FAPAR_VARIABLES = ("Fpar_500m", "FPAR", "fpar", "Fpar")
#: QC variable names, native-handler priority order.
QC_VARIABLES = ("FparLai_QC", "QC", "qc")


@register("modis_fapar")
class MODISFAPARConnector(BaseObservationConnector):
    slug = "modis_fapar"
    display_name = "NASA MODIS FAPAR (MCD15A2H)"
    kind = ObservationKind.FAPAR
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
                "MODIS FAPAR live fetch needs a NetCDF path (config 'nc_path' or 'path') "
                "or an Earthdata download (not yet wired). The reduction path is the "
                "proven part; supply a MCD15A2H/MOD15A2H Fpar_500m NetCDF to reduce it.",
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
        """Open a MODIS FAPAR NetCDF, reduce to the basin, canonicalize to FAPAR ("1").

        Masks the fill byte and out-of-range DN to NaN, optionally applies the QC
        algorithm-path filter when a QC layer is present, reduces over the basin,
        and applies the 0.01 scale factor at the boundary so values are the
        canonical dimensionless FAPAR fraction in ``[0, 1]``. Window-trimmed to
        half-open UTC ``[start, end)``.
        """
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_variable(ds, FAPAR_VARIABLES)
            if var_name is None:
                raise ConnectorError(
                    self.slug,
                    f"NetCDF missing a MODIS FAPAR variable (tried {FAPAR_VARIABLES}). "
                    f"Available: {list(ds.data_vars)}",
                )
            da = ds[var_name]
            lat_name = "lat" if "lat" in ds.coords else ("y" if "y" in ds.coords else "lat")
            lon_name = "lon" if "lon" in ds.coords else ("x" if "x" in ds.coords else "lon")
            lats = np.asarray(ds[lat_name].values, dtype="float64")
            lons = np.asarray(ds[lon_name].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(da.values, dtype="float64")  # (time, lat, lon)

            qc_var = self._find_variable(ds, QC_VARIABLES)
            qc_values = (
                np.asarray(ds[qc_var].values) if qc_var is not None else None
            )

        # Mask fill (255) and out-of-range DN to NaN (native FPAR_VALID_RANGE +
        # FILL_VALUE). The range filter already excludes the fill byte.
        values = self._mask_invalid(values, qc_values)

        # DN -> FAPAR fraction in [0, 1] (≡ canonical dimensionless "1") at the
        # boundary.
        values = values * FAPAR_SCALE_FACTOR

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("MODIS FAPAR basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("MODIS FAPAR nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

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
                "source": "MODIS MCD15A2H/MOD15A2H",
                "source_doi": "10.5067/MODIS/MCD15A2H.061",
                "url": "https://lpdaac.usgs.gov/products/mcd15a2hv061/",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _mask_invalid(
        values: np.ndarray, qc_values: np.ndarray | None
    ) -> np.ndarray:
        """Mask FAPAR DN outside [0, 100] (incl. 255 fill); apply QC if present.

        Mirrors the native ``_extract_basin_mean`` valid-range filter + the
        ``_apply_qc_filter`` algorithm-path filter: out-of-range / fill DN ->
        NaN, then (when a QC layer is supplied) keep only algorithm paths in
        ``QC_GOOD_ALGORITHMS`` (main=0, saturation=2).
        """
        import numpy as np

        lo, hi = VALID_FAPAR_RANGE
        out = values.astype("float64", copy=True)
        invalid = ~((out >= lo) & (out <= hi))
        out[invalid] = np.nan

        if qc_values is not None:
            qc = np.asarray(qc_values)
            algorithm_bits = (qc.astype("int64") >> QC_ALGORITHM_SHIFT) & QC_ALGORITHM_MASK
            good = np.isin(algorithm_bits, list(QC_GOOD_ALGORITHMS))
            out[~good] = np.nan
        return out

    @staticmethod
    def _find_variable(ds: object, candidates: tuple[str, ...]) -> str | None:
        """Find a variable from candidates, native-handler priority order."""
        data_vars = set(getattr(ds, "data_vars", {}))
        for name in candidates:
            if name in data_vars:
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
            site_id = f"modis_fapar:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"modis_fapar:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"MODIS FAPAR over {spec.domain_name}",
        )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# Re-export so a quality flag is importable alongside the connector for tests.
__all__ = ["MODISFAPARConnector", "QualityFlag", "ObservationPoint"]
