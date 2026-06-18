# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GGMN groundwater-level connector (point network, anonymous).

Proves the **point-network / station-selection path** for groundwater. GGMN is
IGRAC's Global Groundwater Monitoring Network; stations are discovered via the
GeoServer WFS (``Groundwater_Well`` features) and each station's well-level
record is fetched from the IGRAC ``WellLevelMeasurement/list`` JSON endpoint.
Both are anonymous, so this is a no-key connector.

Units: the native ``ggmn.py`` handler reads ``value_value`` from each
measurement's embedded HTML form and stores it as ``groundwater_level`` with **no
scaling** — GGMN reports groundwater level in **metres**, which is already the
canonical ``groundwater`` unit (``KIND_UNITS[GROUNDWATER] == "m"``). The
conversion at this connector boundary is therefore identity (m → m); we keep the
parse step explicit so the contract is visible rather than implicit.

Time: GGMN measurement timestamps are ISO-8601; we treat naive timestamps as UTC
and trim to the half-open window ``[start, end)``.
"""

from __future__ import annotations

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


@register("ggmn_gw")
class GGMNGroundwaterConnector(BaseObservationConnector):
    slug = "ggmn_gw"
    display_name = "IGRAC GGMN (Global Groundwater Monitoring Network)"
    kind = ObservationKind.GROUNDWATER
    structural_class = "point_network"
    base_url = "https://ggis.un-igrac.org"
    auth = frozenset()  # anonymous (GGMN/IGRAC)

    #: WFS feature discovery (GeoServer OWS) — mirrors native ``WFS_URL``.
    _WFS_PATH = "/geoserver/ows"
    _WFS_TYPENAME = "groundwater:Groundwater_Well"

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Return the GGMN wells COS would serve for *spec*.

        Explicit ``spec.station_ids`` short-circuit discovery (one site each);
        otherwise we query the WFS by bbox, exactly as the native handler builds
        its ``BBOX(location, ...) AND groundwater_level_data>0`` CQL filter.
        """
        explicit = self._station_ids(spec)
        if explicit:
            return [self._site(sid, None, None, None) for sid in explicit]
        features = await self._wfs_features(spec)
        return [self._site_from_feature(f) for f in features]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        out: list[ObservationSeries] = []
        sites = await self.list_sites(spec)
        for site in sites:
            gid = site.extra.get("ggmn_id") or site.site_id.split(":", 1)[-1]
            payload = await self._fetch_measurements(gid)
            points = self.parse_measurements(payload, start, end)
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=site,
                    reduction=SpatialReduction.STATION,
                    unit=KIND_UNITS[self.kind],
                    points=points,
                    source_info={
                        "source": "IGRAC GGMN",
                        "url": self.base_url,
                        "station": gid,
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    # -- network calls -------------------------------------------------------

    async def _wfs_features(self, spec: ReductionSpec) -> list[dict]:
        if not spec.bbox:
            raise DataFormatError(self.slug, "GGMN discovery needs a bbox or explicit station_ids")
        lat_min, lon_min, lat_max, lon_max = spec.bbox
        cql = (
            f"BBOX(location, {lon_min}, {lat_min}, {lon_max}, {lat_max}) "
            "AND groundwater_level_data>0"
        )
        params = {
            "service": "WFS",
            "version": "1.0.0",
            "request": "GetFeature",
            "typename": self._WFS_TYPENAME,
            "outputFormat": "application/json",
            "cql_filter": cql,
        }
        resp = await self._get(self._WFS_PATH, params=params)
        try:
            collection = resp.json()
        except ValueError as exc:
            raise DataFormatError(self.slug, f"WFS did not return JSON: {exc}") from exc
        return collection.get("features", []) or []

    async def _fetch_measurements(self, gid: str) -> dict:
        path = f"/groundwater/record/{gid}/WellLevelMeasurement/list"
        resp = await self._get(path, headers={"User-Agent": "Mozilla/5.0"})
        try:
            return resp.json()
        except ValueError as exc:
            raise DataFormatError(self.slug, f"station {gid}: list endpoint not JSON: {exc}") from exc

    # -- pure parser (hermetically tested) -----------------------------------

    @staticmethod
    def parse_measurements(payload: dict, start: datetime, end: datetime) -> list[ObservationPoint]:
        """Parse a GGMN ``WellLevelMeasurement/list`` JSON → canonical points.

        The IGRAC list endpoint returns ``{"data": [{"html": "<form …>"}, …]}``
        where each row embeds an ``input[name=time]`` (ISO timestamp) and an
        ``input[name=value_value]`` (groundwater level in metres). The canonical
        ``groundwater`` unit is already metres, so the conversion is identity —
        we parse and trim to half-open UTC ``[start, end)``.

        Non-numeric / empty values map to :class:`QualityFlag.MISSING`.
        """
        if not isinstance(payload, dict):
            raise DataFormatError("ggmn_gw", f"expected mapping payload, got {type(payload).__name__}")
        rows = payload.get("data", [])
        start_u = _utc(start)
        end_u = _utc(end)
        points: list[ObservationPoint] = []
        for item in rows:
            html = (item or {}).get("html", "")
            if not html:
                continue
            time_raw = _extract_input(html, "time")
            val_raw = _extract_input(html, "value_value")
            if time_raw is None:
                continue
            ts = _parse_ts(time_raw)
            if ts is None or not (start_u <= ts < end_u):
                continue
            if val_raw is None or val_raw.strip() == "":
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            try:
                value_m = float(val_raw)  # GGMN metres -> canonical metres (identity)
            except ValueError:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            # Non-finite tokens ("NaN"/"inf") parse via float() but native coerces
            # them with pd.to_numeric + dropna, i.e. they are NOT valid levels.
            # Emit MISSING so a NaN never masquerades as a GOOD groundwater level.
            if value_m != value_m or value_m in (float("inf"), float("-inf")):
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            points.append(ObservationPoint(timestamp=ts, value=value_m, quality=QualityFlag.GOOD))
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
        out: list[str] = []
        for s in ids:
            out.append(s.split(":", 1)[1] if s.lower().startswith("ggmn_gw:") else s)
        return out

    def _site(
        self,
        gid: str,
        lat: float | None,
        lon: float | None,
        name: str | None,
    ) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"ggmn_gw:{gid}",
            latitude=lat,
            longitude=lon,
            name=name or f"GGMN well {gid}",
            extra={"network": "GGMN", "ggmn_id": str(gid)},
        )

    def _site_from_feature(self, feature: dict) -> SiteRef:
        props = feature.get("properties", {}) or {}
        gid = str(props.get("id"))
        lat = lon = None
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
        return self._site(gid, lat, lon, props.get("name"))


def _extract_input(html: str, name: str) -> str | None:
    """Extract ``value`` of ``<input name="{name}" value="…">`` without bs4.

    The native handler uses BeautifulSoup, but the relevant fragment is a flat
    set of hidden inputs; a dependency-light scan keeps the pure parser testable
    offline. Returns None if the named input is absent.
    """
    needle = f'name="{name}"'
    idx = html.find(needle)
    if idx == -1:
        return None
    # Find the enclosing <input ...> tag bounds.
    open_tag = html.rfind("<", 0, idx)
    close_tag = html.find(">", idx)
    if open_tag == -1 or close_tag == -1:
        return None
    tag = html[open_tag:close_tag]
    vkey = 'value="'
    vidx = tag.find(vkey)
    if vidx == -1:
        return None
    vstart = vidx + len(vkey)
    vend = tag.find('"', vstart)
    if vend == -1:
        return None
    return tag[vstart:vend]


def _parse_ts(raw: str) -> datetime | None:
    txt = raw.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(txt, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
