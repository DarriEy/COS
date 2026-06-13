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
        """Sites are the explicitly-requested SNOTEL stations.

        AWDB station discovery by bbox is a separate (planned) call; today the
        domain selects stations by explicit id (``spec.station_ids``), exactly
        as the native handler reads ``SNOTEL_STATION`` from config.
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
            triplet = self._triplet(station_id, spec)
            text = await self._fetch_report(triplet)
            points = self.parse_report(text, start, end)
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=self._site(station_id, spec),
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
        # accept both bare "679" and namespaced "snotel:679"
        return [s.split(":", 1)[-1] for s in ids]

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


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
