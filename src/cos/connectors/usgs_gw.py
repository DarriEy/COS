# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""USGS NWIS groundwater-level connector (point network, anonymous).

Proves the **point-network / station-selection path** for the ``groundwater``
kind. The source is the USGS NWIS water-services API; the ``gwlevels`` /
``iv`` / ``dv`` JSON endpoints are anonymous and need no key.

Parity with the native SYMFLUENCE ``usgs_gw`` handler
(``data/observation/handlers/usgs.py::USGSGroundwaterHandler``):

* selects the groundwater-level variable by NWIS parameter code **72019**
  ("Depth to water level, feet below land surface") or a "water level" /
  "depth to water level" variable-name match — exactly as the native handler;
* the canonical ``groundwater`` unit is **m** (``KIND_UNITS``); NWIS reports the
  72019 series in **feet**, so values are converted **feet → metres (×0.3048)**
  at the connector boundary. A value already in metres ("m") passes through.
  This mirrors the native handler's ``to_meters`` reduction;
* NWIS no-data fill (``-999999``) and blank values map to
  :class:`QualityFlag.MISSING` with ``value=None``;
* timestamps are NWIS ISO-8601 with an offset and are normalised to UTC; the
  window is the half-open interval ``[start, end)``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import DataFormatError
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

#: feet -> metres (matches SYMFLUENCE ``UnitConversion.FEET_TO_METERS``).
FEET_TO_METERS = 0.3048
#: NWIS parameter code for "Depth to water level, feet below land surface".
GW_PARAM_CODE = "72019"
#: NWIS sentinel for "no data" in iv/dv value blocks.
NWIS_NODATA = -999999.0


@register("usgs_gw")
class USGSGroundwaterConnector(BaseObservationConnector):
    slug = "usgs_gw"
    display_name = "USGS NWIS Groundwater Level"
    kind = ObservationKind.GROUNDWATER
    structural_class = "point_network"
    base_url = "https://waterservices.usgs.gov"
    auth = frozenset()  # anonymous (USGS NWIS)

    #: endpoints tried in order, mirroring the native handler's gwlevels→iv→dv
    #: fallback chain. gwlevels carries every parameter; iv/dv are filtered to
    #: 72019 server-side.
    _ENDPOINTS: tuple[tuple[str, dict[str, str]], ...] = (
        ("/nwis/gwlevels/", {}),
        ("/nwis/iv/", {"parameterCd": GW_PARAM_CODE}),
        ("/nwis/dv/", {"parameterCd": GW_PARAM_CODE}),
    )

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Sites are the explicitly-requested NWIS site numbers.

        The native handler reads a single ``USGS_STATION`` / ``STATION_ID`` from
        config; here the domain selects stations by explicit id
        (``spec.station_ids``), and a single ``station``/``station_ids`` config
        key is honoured as a fallback.
        """
        return [self._site(sid, spec) for sid in self._station_ids(spec)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        out: list[ObservationSeries] = []
        for station_id in self._station_ids(spec):
            text, endpoint = await self._fetch_first_nonempty(station_id)
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
                        "source": "USGS NWIS",
                        "url": self.base_url,
                        "station": station_id,
                        "endpoint": endpoint,
                        "parameter": GW_PARAM_CODE,
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    async def _fetch_first_nonempty(self, station_id: str) -> tuple[str, str]:
        """Try gwlevels→iv→dv until a non-empty timeSeries block is returned."""
        last_text = ""
        last_path = self._ENDPOINTS[0][0]
        for path, extra in self._ENDPOINTS:
            params = {
                "format": "json",
                "sites": station_id,
                "agencyCd": "USGS",
                "siteStatus": "all",
                **extra,
            }
            resp = await self._get(path, params=params)
            text = resp.text
            last_text, last_path = text, path
            try:
                data = json.loads(text)
            except ValueError:
                continue
            series = data.get("value", {}).get("timeSeries", [])
            if series:
                return text, path
        # nothing had a timeSeries; hand back the last body so the pure parser
        # yields an empty (rather than error) series — parity with the native
        # handler returning no records.
        return last_text, last_path

    # -- pure parser (hermetically tested, network-free) ---------------------

    @staticmethod
    def parse_series(text: str, start: datetime, end: datetime) -> list[ObservationPoint]:
        """Parse NWIS waterservices JSON → canonical groundwater points (feet→m).

        Selects the groundwater-level variable (param ``72019`` or a "water
        level" name match), converts feet→metres (a "m" unit passes through),
        maps the ``-999999`` fill / blank values to MISSING, and trims to the
        half-open UTC window ``[start, end)``.
        """
        try:
            data = json.loads(text) if text else {}
        except ValueError as exc:
            raise DataFormatError("usgs_gw", f"Invalid NWIS JSON: {exc}") from exc

        value_block = data.get("value", {})
        if not isinstance(value_block, dict):
            raise DataFormatError("usgs_gw", "NWIS JSON missing 'value' object")
        time_series = value_block.get("timeSeries", [])
        if not isinstance(time_series, list):
            raise DataFormatError("usgs_gw", "NWIS JSON 'timeSeries' is not a list")

        start_u = _utc(start)
        end_u = _utc(end)
        points: list[ObservationPoint] = []
        for ts in time_series:
            variable = ts.get("variable", {}) or {}
            param_code = str(variable.get("parameterCode", ""))
            param_name = str(variable.get("variableName", "")).lower()
            is_gw = (
                GW_PARAM_CODE in param_code
                or "depth to water level" in param_name
                or "water level" in param_name
            )
            if not is_gw:
                continue

            unit_code = str(
                (variable.get("unit", {}) or {}).get("unitCode", "")
            ).strip().lower()
            in_feet = unit_code in {"ft", "feet", "foot"}

            for container in ts.get("values", []) or []:
                for obj in container.get("value", []) or []:
                    raw_dt = obj.get("dateTime")
                    raw_val = obj.get("value")
                    if raw_dt is None:
                        continue
                    ts_utc = _parse_nwis_datetime(raw_dt)
                    if ts_utc is None or not (start_u <= ts_utc < end_u):
                        continue
                    point = _make_point(raw_val, in_feet, ts_utc)
                    points.append(point)

        points.sort(key=lambda p: p.timestamp)
        return points

    # -- helpers -------------------------------------------------------------

    def _station_ids(self, spec: ReductionSpec) -> list[str]:
        ids = [s for s in spec.station_ids if s]
        if not ids:
            cfg = self.config.get("station_ids") or self.config.get("station")
            if isinstance(cfg, str):
                ids = [cfg]
            elif isinstance(cfg, (list, tuple)):
                ids = list(cfg)
        # accept bare "385854121023801", namespaced "usgs:<id>", and
        # "USGS-<id>" (the native handler splits on '-' and keeps the tail).
        out: list[str] = []
        for s in ids:
            s = str(s)
            if s.lower().startswith("usgs:"):
                s = s.split(":", 1)[1]
            if "-" in s:
                s = s.split("-")[-1]
            out.append(s)
        return out

    def _site(self, station_id: str, spec: ReductionSpec) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"usgs:{station_id}",
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"USGS {station_id}",
            extra={"network": "NWIS", "parameter": GW_PARAM_CODE},
        )


def _make_point(raw_val: object, in_feet: bool, ts_utc: datetime) -> ObservationPoint:
    if raw_val is None or str(raw_val).strip() == "":
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    try:
        val = float(str(raw_val))
    except (TypeError, ValueError):
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    if val == NWIS_NODATA:
        return ObservationPoint(timestamp=ts_utc, value=None, quality=QualityFlag.MISSING)
    if in_feet:
        val *= FEET_TO_METERS
    return ObservationPoint(timestamp=ts_utc, value=val, quality=QualityFlag.GOOD)


def _parse_nwis_datetime(raw: str) -> datetime | None:
    """Parse an NWIS ISO-8601 timestamp (often with an offset) → UTC.

    Handles ``2020-01-01T00:00:00.000-05:00`` (iv/dv) and bare dates
    ``2020-01-01`` (gwlevels). Naive timestamps are treated as UTC.
    """
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError:
        return None
    return _utc(dt)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
