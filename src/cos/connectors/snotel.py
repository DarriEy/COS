# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""SNOTEL snow-water-equivalent connector (point network, anonymous).

Proves the **point-network / station-selection path**. SNOTEL is the USDA NRCS
Air-Water Database (AWDB); the report-generator CSV endpoint is anonymous and
needs no key, which makes this the live-smoke connector.

The native ``snotel.py`` handler keeps SWE *in inches* ("project convention");
COS converts to the canonical ``swe`` unit (**mm**, ×25.4) at the connector
boundary, so the canonical series is always mm. This is the documented unit
landmine (design §2 / §7) the canonical contract exists to neutralize.
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

INCH_TO_MM = 25.4


@register("snotel")
class SNOTELConnector(BaseObservationConnector):
    slug = "snotel"
    display_name = "NRCS SNOTEL (AWDB)"
    kind = ObservationKind.SWE
    structural_class = "point_network"
    base_url = "https://wcc.sc.egov.usda.gov"
    auth = frozenset()  # anonymous

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Return explicit stations or discover active SWE stations spatially."""
        explicit = self._station_ids(spec)
        if explicit:
            return [self._site(sid, spec) for sid in explicit]
        if not spec.bbox and not spec.centroid:
            raise DataFormatError(self.slug, "station discovery needs a bbox, centroid, or explicit station_ids")
        response = await self._get(
            "/awdbRestApi/services/v1/stations",
            params={
                "stationTriplets": "*:*:SNTL",
                "elements": "WTEQ",
                "durations": "DAILY",
                "activeOnly": "true",
            },
        )
        try:
            rows = response.json()
        except ValueError as exc:
            raise DataFormatError(self.slug, "AWDB station metadata was not valid JSON") from exc
        sites = self._sites_from_metadata(rows, spec)
        limit = int(spec.options.get("max_sites", self.config.get("max_sites", 25)))
        if limit < 1:
            raise DataFormatError(self.slug, "max_sites must be a positive integer")
        return sites[:limit]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        out: list[ObservationSeries] = []
        station_ids = self._station_ids(spec)
        if station_ids:
            selections = [(station_id, self._site(station_id, spec)) for station_id in station_ids]
        else:
            discovered = await self.list_sites(spec)
            fetch_limit = int(spec.options.get("max_fetch_sites", self.config.get("max_fetch_sites", 5)))
            if fetch_limit < 1:
                raise DataFormatError(self.slug, "max_fetch_sites must be a positive integer")
            selections = [(site.site_id.split(":", 1)[1], site) for site in discovered[:fetch_limit]]
        for station_id, site in selections:
            triplet = self._triplet(station_id, spec)
            text = await self._fetch_report(triplet)
            points = self.parse_report(text, start, end)
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=site,
                    reduction=SpatialReduction.STATION,
                    unit=KIND_UNITS[self.kind],
                    points=points,
                    source_info={"source": "NRCS SNOTEL", "url": self.base_url, "station": triplet},
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    async def _fetch_report(self, triplet: str) -> str:
        path = (
            "/reportGenerator/view_csv/customSingleStationReport/daily/"
            f"{triplet}%7Cid=%22%22%7Cname/"
            "POR_BEGIN,POR_END/WTEQ::value"
        )
        resp = await self._get(path)
        return resp.text

    # -- pure parser (hermetically tested) -----------------------------------

    @staticmethod
    def parse_report(text: str, start: datetime, end: datetime) -> list[ObservationPoint]:
        """Parse an NRCS daily report CSV → canonical SWE points (inches→mm).

        NRCS reports lead with ``#`` comment lines, then a header row, then
        ``Date,Snow Water Equivalent (in)`` rows. Trims to half-open UTC
        ``[start, end)``.
        """
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if len(lines) < 2:
            return []
        header = [h.strip() for h in lines[0].split(",")]
        date_idx = next((i for i, h in enumerate(header) if "date" in h.lower()), 0)
        swe_idx = next(
            (i for i, h in enumerate(header)
             if "snow water equivalent" in h.lower() or "wteq" in h.lower()),
            1 if len(header) >= 2 else None,
        )
        if swe_idx is None:
            raise DataFormatError("snotel", f"Could not find SWE column in header {header}")

        start_u = _utc(start)
        end_u = _utc(end)
        points: list[ObservationPoint] = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= max(date_idx, swe_idx):
                continue
            try:
                ts = datetime.fromisoformat(parts[date_idx].strip()).replace(tzinfo=UTC)
            except ValueError:
                continue
            if not (start_u <= ts < end_u):
                continue
            raw = parts[swe_idx].strip()
            if raw == "":
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            try:
                value_mm = float(raw) * INCH_TO_MM
            except ValueError:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            points.append(ObservationPoint(timestamp=ts, value=value_mm, quality=QualityFlag.GOOD))
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
        # accept bare "679", namespaced "snotel:679", AND full AWDB triplets
        # "679:WA:SNTL". Only strip a leading "snotel:" namespace — splitting on
        # any ":" mangles a full triplet (679:WA:SNTL -> WA:SNTL).
        out: list[str] = []
        for s in ids:
            if s.lower().startswith("snotel:"):
                out.append(s.split(":", 1)[1])
            else:
                out.append(s)
        return out

    def _triplet(self, station_id: str, spec: ReductionSpec) -> str:
        """AWDB station triplet ``<id>:<state>:SNTL``."""
        if station_id.count(":") >= 2:
            return station_id
        state = (
            spec.options.get("state")
            or self.config.get("state")
            or "WA"
        )
        return f"{station_id}:{state}:SNTL"

    def _site(self, station_id: str, spec: ReductionSpec) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"snotel:{station_id}",
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"SNOTEL {station_id}",
            extra={"network": "SNTL"},
        )

    @staticmethod
    def _sites_from_metadata(rows: object, spec: ReductionSpec) -> list[SiteRef]:
        """Filter AWDB StationDTO rows and rank them nearest the centroid."""
        if not isinstance(rows, list):
            raise DataFormatError("snotel", "AWDB station metadata must be a list")
        sites: list[SiteRef] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if spec.bbox:
                lat_min, lon_min, lat_max, lon_max = spec.bbox
                if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                    continue
            triplet = str(row.get("stationTriplet") or row.get("stationId") or "").strip()
            if not triplet:
                continue
            triplet_parts = triplet.split(":")
            state = str(row.get("stateCode") or (triplet_parts[1] if len(triplet_parts) >= 3 else ""))
            network = str(row.get("networkCode") or (triplet_parts[2] if len(triplet_parts) >= 3 else "SNTL"))
            sites.append(SiteRef(
                kind="station", site_id=f"snotel:{triplet}", latitude=lat, longitude=lon,
                name=str(row.get("name") or f"SNOTEL {triplet}"),
                extra={"network": network, "state": state},
            ))
        if spec.centroid:
            clat, clon = spec.centroid
            sites.sort(key=lambda s: (s.latitude - clat) ** 2 + (s.longitude - clon) ** 2)  # type: ignore[operator]
        else:
            sites.sort(key=lambda s: s.site_id)
        return sites


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
