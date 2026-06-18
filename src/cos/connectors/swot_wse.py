# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""SWOT river/lake water-surface-elevation connector (point network, anonymous).

The differentiation piece: SWOT (Surface Water and Ocean Topography) water-surface
elevation (WSE) per river *reach* (or lake), served by NASA's **Hydrocron**
time-series REST API. Hydrocron repackages the SWOT ``SWOT_L2_HR_RiverSP`` /
``LakeSP`` feature products into a tidy per-feature time series and is **anonymous**
(confirmed HTTP 200 with no Earthdata token):

    GET /hydrocron/v1/timeseries
        ?feature=Reach&feature_id=<reach_id>
        &start_time=<ISO>&end_time=<ISO>
        &output=csv&fields=reach_id,time_str,wse

There is **no SYMFLUENCE native** for SWOT WSE, so parity is *spec-validated*: the
connector is validated against the published Hydrocron product spec rather than a
native handler. The load-bearing facts encoded here are:

* **unit**: Hydrocron returns ``wse`` in **metres**, which is exactly the canonical
  ``water_level`` unit (``KIND_UNITS[ObservationKind.WATER_LEVEL] == "m"``). The
  conversion at the boundary is therefore the identity scale ``SOURCE_WSE_SCALE``
  (= 1.0); a non-metre ``wse_units`` column, if ever present, is rejected rather
  than silently mis-scaled;
* **fill / missing**: SWOT uses the sentinel ``-999999999999.0`` for "no
  observation"; such a value (and any blank / non-finite ``wse``, and the
  ``no_data`` ``time_str`` placeholder) maps to :class:`QualityFlag.MISSING`;
* **valid range**: a finite WSE outside the physical Earth-elevation band
  (``VALID_WSE_RANGE`` metres, generous bounds spanning the Dead Sea shore to
  above Everest) is treated as fill and masked to MISSING;
* **window**: ``time_str`` is ISO-8601 UTC (``2024-07-25T22:48:23Z``); the series
  is trimmed to the half-open UTC interval ``[start, end)``.

A **gridded** path is also provided for completeness (a supplied WSE raster /
NetCDF reduced over the basin via :mod:`cos.core.reduce` — ``basin_mean`` for
larger basins, ``nearest_cell`` for small ones), so the same canonical contract
covers both a per-reach REST series and a reduced WSE grid. The primary, proven
path is the per-reach REST fetch + pure CSV parse.
"""

from __future__ import annotations

import csv
import io
import math
from datetime import UTC, datetime
from pathlib import Path

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import ConnectorError, DataFormatError, ReductionError
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

#: SWOT / Hydrocron WSE is metres == the canonical ``water_level`` unit, so the
#: boundary conversion is the identity. Documented as a constant so the
#: spec-validated unit contract is explicit (and a future non-metre source would
#: change exactly this number).
SOURCE_WSE_SCALE = 1.0
#: SWOT no-observation sentinel (used for wse and other measurement fields).
SWOT_FILL_VALUE = -999999999999.0
#: Hydrocron emits this placeholder in ``time_str`` for a no-observation pass.
NO_DATA_TIME = "no_data"
#: Physically plausible Earth water-surface elevation band (metres, EGM-ish):
#: below the Dead Sea shore (~-430 m) to above the highest standing water. A
#: finite wse outside this is treated as fill / corrupt and masked to MISSING.
VALID_WSE_RANGE = (-500.0, 9000.0)
#: <= this area (km²) defaults to nearest_cell for the gridded path, mirroring
#: grace.py's size policy.
MEDIUM_BASIN_THRESHOLD_KM2 = 1000.0
#: WSE variable names tried in a supplied NetCDF (gridded path), common order.
WSE_VARIABLES = ("wse", "water_surface_elevation", "Band1", "band1", "height")


@register("swot_wse")
class SWOTWaterLevelConnector(BaseObservationConnector):
    slug = "swot_wse"
    display_name = "SWOT Water-Surface Elevation (Hydrocron)"
    kind = ObservationKind.WATER_LEVEL
    structural_class = "point_network"
    base_url = "https://soto.podaac.earthdatacloud.nasa.gov"
    auth = frozenset()  # anonymous (Hydrocron public REST)

    #: Hydrocron feature class; reaches by default, lakes via config/options.
    DEFAULT_FEATURE = "Reach"
    #: minimal field set the parser needs (units come back automatically).
    FIELDS = "reach_id,time_str,wse"

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Sites are the explicitly-requested SWOT reach (or lake) ids.

        SWOT feature discovery by bbox (against SWORD / PLD) is a separate
        (planned) call; today the domain selects features by explicit id
        (``spec.station_ids``), with a single ``feature_id`` / ``station`` config
        key honoured as a fallback.
        """
        feature = self._feature(spec)
        return [self._site(fid, spec, feature) for fid in self._feature_ids(spec)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        # Gridded path: reduce a supplied WSE raster / NetCDF if one is configured.
        nc_path = self.config.get("nc_path") or self.config.get("path")
        if nc_path:
            return [self.reduce_file(Path(nc_path), spec, start, end)]

        # Point path: per-reach Hydrocron REST fetch + pure CSV parse.
        feature = self._feature(spec)
        out: list[ObservationSeries] = []
        for feature_id in self._feature_ids(spec):
            text = await self._fetch_timeseries(feature, feature_id, start, end)
            points = self.parse_timeseries(text, start, end)
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=self._site(feature_id, spec, feature),
                    reduction=SpatialReduction.STATION,
                    unit=KIND_UNITS[self.kind],
                    points=points,
                    source_info={
                        "source": "NASA SWOT (Hydrocron)",
                        "source_doi": "10.5067/SWOT-RIVERSP-2.0",
                        "url": f"{self.base_url}/hydrocron/v1/timeseries",
                        "feature": feature,
                        "feature_id": feature_id,
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    async def _fetch_timeseries(
        self, feature: str, feature_id: str, start: datetime, end: datetime,
    ) -> str:
        params = {
            "feature": feature,
            "feature_id": feature_id,
            "start_time": _iso_z(start),
            "end_time": _iso_z(end),
            "output": "csv",
            "fields": self.FIELDS,
        }
        resp = await self._get("/hydrocron/v1/timeseries", params=params)
        return resp.text

    # -- pure parser (hermetically tested, network-free) ---------------------

    @staticmethod
    def parse_timeseries(text: str, start: datetime, end: datetime) -> list[ObservationPoint]:
        """Parse a Hydrocron CSV time series → canonical WSE points (metres).

        Hydrocron may wrap the CSV in a JSON envelope (``{"results": {"csv":
        "..."}}``) or return raw CSV; both are accepted. Columns are matched by
        header name (order-independent): ``time_str`` and ``wse`` are required.

        Spec-validated behaviour (no native to mirror):

        * ``wse`` is metres → canonical ``water_level`` unit; scaled by the
          identity ``SOURCE_WSE_SCALE``. A ``wse_units`` column other than metres
          raises :class:`DataFormatError` rather than silently mis-scaling;
        * the SWOT fill sentinel ``-999999999999.0``, a blank / non-finite
          ``wse``, or a wse outside ``VALID_WSE_RANGE`` → ``value=None`` with
          :class:`QualityFlag.MISSING`;
        * rows whose ``time_str`` is ``no_data`` / unparseable are skipped (no
          timestamp to anchor a point);
        * the series is trimmed to the half-open UTC interval ``[start, end)``.
        """
        csv_text = _unwrap_csv(text)
        reader = csv.reader(io.StringIO(csv_text))
        rows = [r for r in reader if any(cell.strip() for cell in r)]
        if len(rows) < 2:
            return []

        header = [h.strip().lower() for h in rows[0]]
        try:
            time_idx = header.index("time_str")
        except ValueError as exc:
            raise DataFormatError("swot_wse", f"Hydrocron CSV missing 'time_str' column: {header}") from exc
        try:
            wse_idx = header.index("wse")
        except ValueError as exc:
            raise DataFormatError("swot_wse", f"Hydrocron CSV missing 'wse' column: {header}") from exc
        units_idx = header.index("wse_units") if "wse_units" in header else None

        start_u = _utc(start)
        end_u = _utc(end)
        points: list[ObservationPoint] = []
        for row in rows[1:]:
            if len(row) <= max(time_idx, wse_idx):
                continue

            # Reject a non-metre unit at the boundary (spec contract guard).
            if units_idx is not None and len(row) > units_idx:
                unit = row[units_idx].strip().lower()
                if unit and unit not in {"m", "meter", "meters", "metre", "metres"}:
                    raise DataFormatError(
                        "swot_wse",
                        f"Hydrocron wse_units={row[units_idx]!r} is not metres; the "
                        "canonical water_level unit is 'm' — refusing to mis-scale.",
                    )

            ts = _parse_iso(row[time_idx].strip())
            if ts is None:  # 'no_data' placeholder or unparseable timestamp
                continue
            if not (start_u <= ts < end_u):
                continue
            points.append(_make_point(row[wse_idx].strip(), ts))

        points.sort(key=lambda p: p.timestamp)
        return points

    # -- gridded path (supplied WSE raster / NetCDF) -------------------------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open a WSE NetCDF, reduce to the basin, canonicalize to metres.

        The gridded counterpart to the per-reach path: extract ``wse`` (metres),
        mask the SWOT fill sentinel / out-of-range cells to NaN, reduce over the
        basin, and window-trim to half-open UTC ``[start, end)``. Values are
        already metres, so the boundary scale is the identity ``SOURCE_WSE_SCALE``.
        """
        import numpy as np
        import xarray as xr

        reduction = self._choose_reduction(spec)
        with xr.open_dataset(nc_path) as ds:
            var_name = self._find_wse_variable(ds)
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(ds[var_name].values, dtype="float64")  # (time, lat, lon)

        values = self._mask_invalid(values) * SOURCE_WSE_SCALE

        from cos.core.reduce import reduce_grid

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("SWOT WSE basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("SWOT WSE nearest_cell requires spec.centroid")

        points = reduce_grid(
            lats, lons, times, values,
            reduction=reduction, bbox=bbox, point=point,
            kind=self.kind, unit=KIND_UNITS[self.kind],
        )

        start_u = _utc(start)
        end_u = _utc(end)
        points = [p for p in points if start_u <= p.timestamp < end_u]

        return ObservationSeries(
            provider=self.slug,
            kind=self.kind,
            site=self._site_for_grid(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "NASA SWOT (gridded WSE)",
                "url": f"{self.base_url}/hydrocron/v1/timeseries",
                "variable": var_name,
            },
            fetched_at=datetime.now(UTC),
        )

    @staticmethod
    def _mask_invalid(values):
        """Mask the SWOT fill sentinel and out-of-range WSE cells to NaN."""
        import numpy as np

        lo, hi = VALID_WSE_RANGE
        out = np.asarray(values, dtype="float64").copy()
        invalid = ~np.isfinite(out) | (out == SWOT_FILL_VALUE) | (out < lo) | (out > hi)
        out[invalid] = np.nan
        return out

    @staticmethod
    def _find_wse_variable(ds) -> str:
        for var in WSE_VARIABLES:
            if var in ds.data_vars:
                return str(var)
        suitable = [
            v for v in ds.data_vars
            if "wse" in str(v).lower() or "elevation" in str(v).lower() or "height" in str(v).lower()
        ]
        if suitable:
            return str(suitable[0])
        raise ConnectorError(
            "swot_wse",
            f"No WSE / elevation variable found in dataset. Available: {list(ds.data_vars)}",
        )

    # -- helpers -------------------------------------------------------------

    def _feature(self, spec: ReductionSpec) -> str:
        feature = spec.options.get("feature") or self.config.get("feature") or self.DEFAULT_FEATURE
        return str(feature)

    def _feature_ids(self, spec: ReductionSpec) -> list[str]:
        ids = [s for s in spec.station_ids if s]
        if not ids:
            cfg = (
                self.config.get("station_ids")
                or self.config.get("feature_ids")
                or self.config.get("feature_id")
                or self.config.get("station")
            )
            if isinstance(cfg, str):
                ids = [cfg]
            elif isinstance(cfg, (list, tuple)):
                ids = list(cfg)
        # accept bare "78340600051" and namespaced "swot:78340600051".
        out: list[str] = []
        for s in ids:
            s = str(s)
            if s.lower().startswith("swot:"):
                s = s.split(":", 1)[1]
            out.append(s)
        return out

    def _site(self, feature_id: str, spec: ReductionSpec, feature: str) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"swot:{feature_id}",
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"SWOT {feature} {feature_id}",
            extra={"network": "SWOT", "feature": feature},
        )

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.area_km2 is not None and spec.area_km2 <= MEDIUM_BASIN_THRESHOLD_KM2:
            return SpatialReduction.NEAREST_CELL
        return SpatialReduction.BASIN_MEAN

    def _site_for_grid(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"swot:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"swot:cell:{clat:.3f}_{clon:.3f}"
        lat = spec.centroid[0] if spec.centroid else None
        lon = spec.centroid[1] if spec.centroid else None
        return SiteRef(
            kind="reduced_region", site_id=site_id, latitude=lat, longitude=lon,
            name=f"SWOT WSE over {spec.domain_name}",
        )


def _make_point(raw_val: str, ts_utc: datetime) -> ObservationPoint:
    """One CSV cell → a canonical WSE point; fill / out-of-range → MISSING."""
    if raw_val == "":
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    try:
        val = float(raw_val)
    except ValueError:
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    lo, hi = VALID_WSE_RANGE
    if val == SWOT_FILL_VALUE or not math.isfinite(val) or val < lo or val > hi:
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    return ObservationPoint(timestamp=ts_utc, value=val * SOURCE_WSE_SCALE, quality=QualityFlag.GOOD)


def _unwrap_csv(text: str) -> str:
    """Return raw CSV from a Hydrocron body that is raw CSV or a JSON envelope.

    Hydrocron's ``output=csv`` may return the CSV either directly or nested as
    ``{"results": {"csv": "..."}}`` (or ``{"csv": "..."}``). Anything that is not
    JSON is treated as raw CSV.
    """
    import json

    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return text
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, dict) and isinstance(results.get("csv"), str):
            return str(results["csv"])
        if isinstance(data.get("csv"), str):
            return str(data["csv"])
    return text


def _parse_iso(raw: str) -> datetime | None:
    """Parse a Hydrocron ``time_str`` (ISO-8601 UTC) → aware UTC datetime.

    Handles the trailing-``Z`` form ``2024-07-25T22:48:23Z`` and offset forms.
    The ``no_data`` placeholder and anything unparseable return ``None``.
    """
    if not raw or raw.lower() == NO_DATA_TIME:
        return None
    candidate = raw[:-1] + "+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return _utc(dt)


def _iso_z(value: datetime) -> str:
    """Format a datetime as a Hydrocron-friendly ``...Z`` UTC ISO-8601 string."""
    return _utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["SWOTWaterLevelConnector", "QualityFlag", "ObservationPoint"]
