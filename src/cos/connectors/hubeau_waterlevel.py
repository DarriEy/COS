# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""French Hub'Eau water-level connector (point network, anonymous).

Serves the ``water_level`` kind from the French government open-data Hub'Eau
hydrometry API (``hydrometrie/observations_tr``), which is anonymous and needs
no key. This is the **point-network / station-selection path**: water level is
a per-station (per-reach) observation, one canonical series per requested
``code_station``.

Parity with the native SYMFLUENCE ``hubeau_waterlevel`` handler
(``data/observation/handlers/hubeau.py::HubEauWaterLevelHandler``):

* selects the water-level (``hauteur d'eau``) series via the real-time
  observations endpoint with ``grandeur_hydro='H'`` — exactly as the native
  handler;
* the canonical ``water_level`` unit is **m** (``KIND_UNITS``); Hub'Eau reports
  ``resultat_obs`` in **millimetres**, so values are converted **mm → metres
  (÷1000)** at the connector boundary. This mirrors the native handler's
  ``water_level_m = water_level_mm / 1000.0`` reduction. A series already in
  metres ("m", ``grandeur_hydro`` unit override) passes through;
* a ``null`` / blank ``resultat_obs`` maps to :class:`QualityFlag.MISSING` with
  ``value=None``;
* Hub'Eau timestamps (``date_obs``, ISO-8601, typically a UTC ``Z`` offset) are
  normalised to UTC; the window is the half-open interval ``[start, end)``.

A gridded path is also provided for the rare case of a supplied raster of water
level (config ``nc_path`` / ``path``): the grid is reduced to the evaluation
geometry via :func:`cos.core.reduce.reduce_grid` (basin-mean over ``spec.bbox``
or nearest-cell at ``spec.centroid``), converting source units to the canonical
``m`` at the boundary (the documented scale factor + fill mask).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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

#: Hub'Eau reports water level (``hauteur``) in millimetres; canonical is metres.
MM_TO_M = 1.0 / 1000.0
#: real-time hydrometry "grandeur" code for water level (Hauteur).
GRANDEUR_WATER_LEVEL = "H"
#: Hub'Eau API page-size cap (mirrors the native handler's MAX_RECORDS).
MAX_RECORDS_PER_REQUEST = 10000
#: source unit codes that are already in metres (pass-through, no scaling).
_METRE_UNITS = {"m", "metre", "meter", "metres", "meters"}


@register("hubeau_waterlevel")
class HubEauWaterLevelConnector(BaseObservationConnector):
    slug = "hubeau_waterlevel"
    display_name = "Hub'Eau Hydrometrie Water Level"
    kind = ObservationKind.WATER_LEVEL
    structural_class = "point_network"
    base_url = "https://hubeau.eaufrance.fr/api/v1/hydrometrie"
    auth = frozenset()  # anonymous

    #: real-time observations endpoint (sub-daily H series).
    _OBS_TR_PATH = "/observations_tr"

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Sites are the explicitly-requested Hub'Eau station codes.

        The native handler reads a single ``code_station`` from config; here the
        domain selects stations by explicit id (``spec.station_ids``), and a
        single ``station``/``station_ids`` config key is honoured as a fallback.
        """
        return [self._site(sid, spec) for sid in self._station_ids(spec)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        # Gridded override: reduce a supplied raster of water level if asked.
        nc_path = self.config.get("nc_path") or self.config.get("path")
        if nc_path:
            return [self.reduce_file(Path(str(nc_path)), spec, start, end)]

        out: list[ObservationSeries] = []
        for station_id in self._station_ids(spec):
            text = await self._fetch_station(station_id, start, end)
            points = self.parse_series(text, start, end)
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=self._site(station_id, spec),
                    reduction=SpatialReduction.STATION,
                    unit=KIND_UNITS[self.kind],
                    points=points,
                    source_info={
                        "source": "Hub'Eau Hydrometrie API",
                        "url": self.base_url,
                        "station": station_id,
                        "grandeur_hydro": GRANDEUR_WATER_LEVEL,
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    async def _fetch_station(self, station_id: str, start: datetime, end: datetime) -> str:
        """Fetch all real-time water-level pages for one station as a JSON envelope.

        Follows the Hub'Eau ``next`` cursor (mirrors the native handler's
        pagination loop) and returns a single ``{"data": [...]}`` body so the
        pure parser is hermetic and network-free.
        """
        all_data: list[dict] = []
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {
                "code_entite": station_id,
                "date_debut_obs": _utc(start).strftime("%Y-%m-%d"),
                "date_fin_obs": _utc(end).strftime("%Y-%m-%d"),
                "grandeur_hydro": GRANDEUR_WATER_LEVEL,
                "size": MAX_RECORDS_PER_REQUEST,
            }
            if cursor:
                params["cursor"] = cursor
            resp = await self._get(self._OBS_TR_PATH, params=params)
            try:
                result = resp.json()
            except ValueError as exc:
                raise DataFormatError(self.slug, f"Invalid Hub'Eau JSON: {exc}") from exc
            data = result.get("data", []) or []
            all_data.extend(data)
            cursor = result.get("next")
            if not cursor or not data:
                break
        return json.dumps({"data": all_data})

    # -- pure parser (hermetically tested, network-free) ---------------------

    @staticmethod
    def parse_series(text: str, start: datetime, end: datetime) -> list[ObservationPoint]:
        """Parse a Hub'Eau observations JSON envelope → canonical points (mm→m).

        Reads the ``data`` array of ``observations_tr`` records, converts
        ``resultat_obs`` from millimetres to metres (a record flagged in metres
        passes through), maps ``null``/blank values to MISSING, and trims to the
        half-open UTC window ``[start, end)``.
        """
        try:
            payload = json.loads(text) if text else {}
        except ValueError as exc:
            raise DataFormatError("hubeau_waterlevel", f"Invalid Hub'Eau JSON: {exc}") from exc

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            block = payload.get("data", [])
            records = block if isinstance(block, list) else []
        else:
            raise DataFormatError("hubeau_waterlevel", "Hub'Eau JSON is neither list nor object")

        start_u = _utc(start)
        end_u = _utc(end)
        points: list[ObservationPoint] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            raw_dt = rec.get("date_obs")
            if raw_dt is None:
                continue
            ts_utc = _parse_hubeau_datetime(str(raw_dt))
            if ts_utc is None or not (start_u <= ts_utc < end_u):
                continue
            points.append(_make_point(rec.get("resultat_obs"), _record_unit(rec), ts_utc))

        points.sort(key=lambda p: p.timestamp)
        return points

    # -- gridded path --------------------------------------------------------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> ObservationSeries:
        """Open a water-level raster, reduce to the basin, canonicalize to metres.

        Source units are converted to the canonical ``m`` at the boundary using
        a documented scale factor (``options['source_scale_to_m']``, default 1.0
        for a metre source; pass ``0.001`` for a millimetre source); a sentinel
        fill (``options['fill_value']``, default NaN) is masked to NaN so it maps
        to :class:`QualityFlag.MISSING`.
        """
        import numpy as np
        import xarray as xr

        from cos.core.reduce import reduce_grid

        reduction = self._choose_reduction(spec)
        var = str(self.config.get("variable") or spec.options.get("variable") or "water_level")
        scale = float(spec.options.get("source_scale_to_m", 1.0))
        fill = spec.options.get("fill_value")

        with xr.open_dataset(nc_path) as ds:
            if var not in ds:
                raise ConnectorError(self.slug, f"NetCDF missing '{var}' variable")
            lats = np.asarray(ds["lat"].values, dtype="float64")
            lons = np.asarray(ds["lon"].values, dtype="float64")
            times = np.asarray(ds["time"].values)
            values = np.asarray(ds[var].values, dtype="float64")  # (time, lat, lon)

        if fill is not None:
            values = np.where(values == float(fill), np.nan, values)
        values = values * scale  # source units -> metres at the boundary

        point = spec.centroid
        bbox = spec.bbox
        if reduction == SpatialReduction.BASIN_MEAN and bbox is None:
            raise ReductionError("hubeau_waterlevel basin_mean requires spec.bbox")
        if reduction != SpatialReduction.BASIN_MEAN and point is None:
            raise ReductionError("hubeau_waterlevel nearest_cell requires spec.centroid")

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
            site=self._grid_site(spec, reduction),
            reduction=reduction,
            unit=KIND_UNITS[self.kind],
            points=points,
            source_info={
                "source": "Hub'Eau water-level raster",
                "variable": var,
                "scale_to_m": str(scale),
            },
            fetched_at=datetime.now(UTC),
        )

    def _choose_reduction(self, spec: ReductionSpec) -> SpatialReduction:
        if spec.reduction is not None:
            return spec.reduction
        if spec.bbox is not None:
            return SpatialReduction.BASIN_MEAN
        return SpatialReduction.NEAREST_CELL

    # -- helpers -------------------------------------------------------------

    def _station_ids(self, spec: ReductionSpec) -> list[str]:
        ids = [s for s in spec.station_ids if s]
        if not ids:
            cfg = self.config.get("station_ids") or self.config.get("station")
            if isinstance(cfg, str):
                ids = [cfg]
            elif isinstance(cfg, (list, tuple)):
                ids = list(cfg)
        # accept bare "H5920010" and namespaced "hubeau:H5920010".
        out: list[str] = []
        for s in ids:
            s = str(s)
            if s.lower().startswith("hubeau:"):
                s = s.split(":", 1)[1]
            out.append(s)
        return out

    def _site(self, station_id: str, spec: ReductionSpec) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"hubeau:{station_id}",
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"Hub'Eau {station_id}",
            extra={"network": "Hub'Eau Hydrometrie", "grandeur_hydro": GRANDEUR_WATER_LEVEL},
        )

    def _grid_site(self, spec: ReductionSpec, reduction: SpatialReduction) -> SiteRef:
        if reduction == SpatialReduction.BASIN_MEAN:
            site_id = f"hubeau_waterlevel:domain:{spec.domain_name}"
        else:
            clat, clon = spec.centroid or (0.0, 0.0)
            site_id = f"hubeau_waterlevel:cell:{clat:.3f}_{clon:.3f}"
        return SiteRef(
            kind="reduced_region",
            site_id=site_id,
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"Hub'Eau water level over {spec.domain_name}",
        )


def _record_unit(rec: dict) -> str:
    """Source unit of one record; default mm (Hub'Eau convention)."""
    for key in ("grandeur_hydro_unite", "unite", "unit"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return "mm"


def _make_point(raw_val: object, unit: str, ts_utc: datetime) -> ObservationPoint:
    if raw_val is None or str(raw_val).strip() == "":
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    try:
        val = float(str(raw_val))
    except (TypeError, ValueError):
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    if unit not in _METRE_UNITS:
        val *= MM_TO_M  # mm -> m at the boundary
    return ObservationPoint(timestamp=ts_utc, value=val, quality=QualityFlag.GOOD)


def _parse_hubeau_datetime(raw: str) -> datetime | None:
    """Parse a Hub'Eau ISO-8601 timestamp (often a ``Z`` UTC offset) → UTC."""
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = cast(datetime, datetime.fromisoformat(s))
    except ValueError:
        return None
    return _utc(dt)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
