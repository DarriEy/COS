# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""ISMN in-situ soil-moisture connector (point network, registered auth).

Exercises the **point-network / station-selection path** for volumetric soil
moisture. ISMN (International Soil Moisture Network, ismn.earth) serves in-situ
soil-moisture observations from globally distributed network stations, at one or
more sensor depths. Unlike SNOTEL (anonymous), ISMN requires a registered
account; the native SYMFLUENCE handler resolves ``ISMN_USERNAME`` /
``ISMN_PASSWORD`` from config / env, then falls back to a ``~/.netrc`` entry for
``ismn.earth``. This connector declares that requirement via ``auth`` and selects
stations by explicit id (the proven, parity-checked path), mirroring the native
handler's per-station CSV download.

Unit landmine (design §2 / §7): ISMN soil moisture is **volumetric m³/m³**, the
canonical ``soil_moisture`` unit, so the boundary conversion is the identity —
*except* when a source reports percent saturation (unit string contains
``* 100`` / ends in ``100``, or values exceed the physical 1.5 ceiling). In that
case the native handler divides by 100 to recover m³/m³; this connector applies
the exact same rule in its pure parser so the canonical series is always m³/m³.

The per-station fetch + parse is the architecture-critical path; it is split into
a network-free, hermetically-tested pure parser (:meth:`parse_station_csv`) so
the canonicalization (units, fill→MISSING, half-open UTC window-trim) is covered
without network or credentials.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import ConnectorError
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

#: Values above this (in the SM column) are read as percent saturation and
#: divided by 100 to recover m³/m³ — the exact threshold the native handler uses.
PERCENT_CEILING = 1.5


@register("ismn_sm")
class ISMNSoilMoistureConnector(BaseObservationConnector):
    slug = "ismn_sm"
    display_name = "ISMN In-Situ Soil Moisture"
    kind = ObservationKind.SOIL_MOISTURE
    structural_class = "point_network"
    base_url = "https://ismn.earth/dataviewer"
    #: ISMN requires a registered account (config ISMN_USERNAME/PASSWORD, env, or
    #: a ~/.netrc entry for ismn.earth) — declared here so the credential layer
    #: knows this connector is not anonymous.
    auth = frozenset({"ismn"})

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Sites are the explicitly-requested ISMN stations.

        ISMN station discovery by bbox (the native handler's metadata + Haversine
        ranking) is a separate planned call; today the domain selects stations by
        explicit id (``spec.station_ids`` / config ``station_ids``), the
        deterministic parity-checked path.
        """
        return [self._site(sid, spec) for sid in self._station_ids(spec)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        station_ids = self._station_ids(spec)
        if not station_ids:
            raise ConnectorError(
                self.slug,
                "ISMN fetch needs station ids (config 'station_ids'/'station' or "
                "spec.station_ids). bbox station discovery is not yet wired; the "
                "per-station parse path is the proven part.",
            )
        out: list[ObservationSeries] = []
        for station_id in station_ids:
            text = await self._fetch_station_csv(station_id, start, end)
            points = self.parse_station_csv(text, start, end)
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=self._site(station_id, spec),
                    reduction=SpatialReduction.STATION,
                    unit=KIND_UNITS[self.kind],
                    points=points,
                    source_info={
                        "source": "ISMN (International Soil Moisture Network)",
                        "url": "https://ismn.earth",
                        "station": station_id,
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    async def _fetch_station_csv(self, station_id: str, start: datetime, end: datetime) -> str:
        """Download one station's soil-moisture CSV (DateTime, soil_moisture).

        The native handler hits ``dataviewer_load_variable`` returning a JSON
        ``[dates, values]`` pair and writes a ``DateTime,soil_moisture`` CSV; the
        canonical contract here consumes that same two-column CSV layout.
        """
        path = (
            "/dataviewer_load_variable/"
            f"?station_id={station_id}"
            f"&start={_utc(start).strftime('%Y/%m/%d')}"
            f"&end={_utc(end).strftime('%Y/%m/%d')}"
        )
        resp = await self._get(path)
        return resp.text

    # -- pure parser (hermetically tested) -----------------------------------

    @staticmethod
    def parse_station_csv(text: str, start: datetime, end: datetime) -> list[ObservationPoint]:
        """Parse one ISMN station CSV → canonical SM points (m³/m³).

        Expects a header row with a date/time column and a soil-moisture column
        (``soil_moisture`` / ``vsm`` / ``theta`` / ``sm*``), then data rows.
        Performs the native handler's percent→fraction rule: if any finite value
        exceeds :data:`PERCENT_CEILING` (i.e. the series is percent saturation),
        the whole series is divided by 100 to recover m³/m³. Blank / unparseable
        values become MISSING. Trims to half-open UTC ``[start, end)``.
        """
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if len(lines) < 2:
            return []
        header = [h.strip() for h in lines[0].split(",")]
        date_idx = _find_col(header, ("timestamp", "datetime", "date", "time"))
        if date_idx is None:
            date_idx = 0
        sm_idx = _find_col(header, ("soil_moisture", "soilmoisture", "volumetric", "vsm", "theta"))
        if sm_idx is None:
            sm_idx = _find_sm_fallback(header)
        if sm_idx is None:
            raise ConnectorError("ismn_sm", f"Could not find soil-moisture column in header {header}")

        start_u = _utc(start)
        end_u = _utc(end)

        # First pass: collect (timestamp, raw_value) within the window, deciding
        # whether the series is percent saturation (any value > PERCENT_CEILING).
        rows: list[tuple[datetime, float | None]] = []
        is_percent = False
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) <= max(date_idx, sm_idx):
                continue
            ts = _parse_ts(parts[date_idx].strip())
            if ts is None or not (start_u <= ts < end_u):
                continue
            raw = parts[sm_idx].strip()
            if raw == "":
                rows.append((ts, None))
                continue
            try:
                val = float(raw)
            except ValueError:
                rows.append((ts, None))
                continue
            if val > PERCENT_CEILING:
                is_percent = True
            rows.append((ts, val))

        scale = 0.01 if is_percent else 1.0
        points: list[ObservationPoint] = []
        for ts, val in rows:
            if val is None:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
            else:
                points.append(
                    ObservationPoint(timestamp=ts, value=val * scale, quality=QualityFlag.GOOD)
                )
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
            # strip only a leading "ismn:" namespace; ISMN station ids may
            # themselves contain ":" (network:station), so do not split further.
            out.append(s.split(":", 1)[1] if s.lower().startswith("ismn:") else s)
        return out

    def _site(self, station_id: str, spec: ReductionSpec) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"ismn:{station_id}",
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"ISMN {station_id}",
            extra={"network": "ISMN"},
        )


def _find_col(header: list[str], terms: tuple[str, ...]) -> int | None:
    for i, h in enumerate(header):
        lower = h.lower()
        if any(term in lower for term in terms):
            return i
    return None


def _find_sm_fallback(header: list[str]) -> int | None:
    """Fallback: a column starting with 'sm' that is not a flag/qc column."""
    for i, h in enumerate(header):
        lower = h.lower()
        if lower.startswith("sm") and "flag" not in lower and "qc" not in lower:
            return i
    return None


def _parse_ts(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
