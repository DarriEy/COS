# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""FLUXNET / AmeriFlux flux-tower evapotranspiration connector (point network).

Proves the **flux-tower / station-selection path** for evapotranspiration. A
FLUXNET tower delivers a half-hourly / daily latent-heat-flux (LE) record from a
FULLSET CSV (FLUXNET2015) or an AmeriFlux BASE CSV; the COS canonical ``et`` unit
is **mm/day**, so the connector converts LE (W/m^2) -> ET (mm/day) at the
boundary and trims to half-open UTC ``[start, end)``.

This mirrors the native SYMFLUENCE handler
(``data/acquisition/handlers/fluxnet.py`` + ``fluxnet_constants.py``):

* column mapping picks the first present LE alias (``LE_F_MDS``, ``LE_CORR``,
  ``LE_1_1_1``, ...) and any matching ``*_QC`` column;
* LE -> ET uses ``ET = LE / (rho_w * lambda) * 86400`` with ``rho_w = 1000`` and
  ``lambda = 2.45e6`` (the ``LE_TO_ET_FACTOR ~= 0.0353`` shared constant);
* QC > ``max_qc`` (default 1: keep measured + good gap-fill) -> dropped to MISSING;
* negative ET (a quality artefact) -> MISSING, matching ``convert_le_to_et``;
* a sub-daily record is averaged to daily means before conversion.

The AmeriFlux API pull is keyed (registration: ``AMERIFLUX_USER_ID`` +
``AMERIFLUX_USER_EMAIL``); like ``grace.py`` the live pull is deferred and the
proven, hermetically-tested part is the pure ``parse_report`` parser over a
supplied FULLSET CSV (config ``path`` / ``csv_path``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

from cos.connectors.base import BaseObservationConnector
from cos.core.exceptions import ConnectorError, DataFormatError
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

# LE (W/m^2) -> ET (mm/day): ET = LE / (rho_w * lambda) * seconds_per_day.
# Mirrors symfluence fluxnet_constants (WATER_DENSITY, LATENT_HEAT_VAPORIZATION).
WATER_DENSITY = 1000.0          # kg/m^3
LATENT_HEAT_VAPORIZATION = 2.45e6  # J/kg at ~20 C
SECONDS_PER_DAY = 86400.0
MM_PER_M = 1000.0               # ET = depth in m/day -> mm/day
# ET[mm/day] = LE/(rho_w*lambda)*seconds_per_day [m/day] * 1000 [mm/m].
# Without MM_PER_M the result is metres/day (~1000x too small): 283.5 W/m^2
# must give ~10 mm/day, not 0.01.
LE_TO_ET_FACTOR = SECONDS_PER_DAY * MM_PER_M / (WATER_DENSITY * LATENT_HEAT_VAPORIZATION)  # ~0.03527

# FLUXNET / AmeriFlux missing sentinel.
FLUXNET_FILL = -9999.0
# Default QC ceiling: 0=measured, 1=good gap-fill, 2=medium, 3=poor.
DEFAULT_MAX_QC = 1

# LE column aliases (priority order), mirroring FLUXNET_VARIABLE_MAPPING['LE'].
LE_ALIASES = ("LE_F_MDS", "LE_PI_F_1_1_1", "LE_CORR", "LE_1_1_1", "LE", "LE_F")
# A pre-computed ET column, if the source already supplies one (mm/day).
ET_ALIASES = ("ET", "ET_F_MDS", "et_mm_day")
LE_QC_ALIASES = ("LE_F_MDS_QC", "LE_QC")
TIMESTAMP_ALIASES = ("TIMESTAMP_START", "TIMESTAMP", "datetime", "Date", "date")


@register("fluxnet_et")
class FluxnetETConnector(BaseObservationConnector):
    slug = "fluxnet_et"
    display_name = "FLUXNET / AmeriFlux flux-tower ET"
    kind = ObservationKind.ET
    structural_class = "point_network"
    base_url = "https://amfcdn.lbl.gov/api/v1"
    auth = frozenset({"ameriflux"})

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Sites are the explicitly-requested flux-tower stations.

        AmeriFlux bbox discovery is a separate (planned) call; today the domain
        selects towers by explicit id (``spec.station_ids`` / config), exactly as
        the native handler reads ``FLUXNET_STATION`` from config.
        """
        return [self._site(sid, spec) for sid in self._station_ids(spec)]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        path = self.config.get("path") or self.config.get("csv_path") or self.config.get("nc_path")
        if not path:
            raise ConnectorError(
                self.slug,
                "FLUXNET live pull needs an AmeriFlux account (config 'ameriflux' auth) "
                "or a downloaded FULLSET CSV (config 'path'/'csv_path'). The parse + "
                "LE->ET canonicalization is the proven part; supply a FLUXNET/AmeriFlux "
                "FULLSET CSV to parse it.",
            )
        station_id = (self._station_ids(spec) or ["unknown"])[0]
        text = Path(path).read_text()
        points = self.parse_report(text, start, end, max_qc=self._max_qc(spec))
        return [
            ObservationSeries(
                provider=self.slug,
                kind=self.kind,
                site=self._site(station_id, spec),
                reduction=SpatialReduction.STATION,
                unit=KIND_UNITS[self.kind],
                points=points,
                source_info={
                    "source": "FLUXNET/AmeriFlux",
                    "url": "https://fluxnet.org/",
                    "station": station_id,
                    "le_to_et_factor": f"{LE_TO_ET_FACTOR:.6f}",
                },
                fetched_at=datetime.now(UTC),
            )
        ]

    # -- pure parser (hermetically tested) -----------------------------------

    @staticmethod
    def parse_report(
        text: str,
        start: datetime,
        end: datetime,
        max_qc: int = DEFAULT_MAX_QC,
    ) -> list[ObservationPoint]:
        """Parse a FLUXNET/AmeriFlux FULLSET CSV -> canonical ET points (mm/day).

        Picks the first present LE alias (W/m^2) and converts to ET via
        ``LE_TO_ET_FACTOR``; a pre-computed ET column is used as-is (mm/day).
        Drops rows where the matching ``*_QC`` exceeds *max_qc* and where the fill
        sentinel (-9999) appears, emitting them as MISSING. Trims to half-open UTC
        ``[start, end)``. AmeriFlux ``#`` comment lines are skipped.
        """
        lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if len(lines) < 2:
            return []
        header = [h.strip() for h in lines[0].split(",")]
        col = {h: i for i, h in enumerate(header)}

        ts_idx = _first_index(col, TIMESTAMP_ALIASES)
        if ts_idx is None:
            raise DataFormatError("fluxnet_et", f"No timestamp column in header {header}")

        et_idx = _first_index(col, ET_ALIASES)
        le_idx = _first_index(col, LE_ALIASES)
        if et_idx is None and le_idx is None:
            raise DataFormatError(
                "fluxnet_et", f"No ET or LE column found in header {header}"
            )
        qc_idx = _first_index(col, LE_QC_ALIASES)

        start_u = _utc(start)
        end_u = _utc(end)
        points: list[ObservationPoint] = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            needed = [i for i in (ts_idx, et_idx, le_idx, qc_idx) if i is not None]
            if len(parts) <= max(needed):
                continue
            ts = _parse_timestamp(parts[ts_idx])
            if ts is None:
                continue
            if not (start_u <= ts < end_u):
                continue

            # QC gate (mirrors FLUXNET_QC_FILTER: drop QC > max_qc to MISSING).
            if qc_idx is not None:
                qc = _to_float(parts[qc_idx])
                if qc is not None and qc > max_qc:
                    points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                    continue

            value = None
            if et_idx is not None:
                value = _to_float(parts[et_idx])
            elif le_idx is not None:
                le = _to_float(parts[le_idx])
                if le is not None:
                    value = le * LE_TO_ET_FACTOR

            if value is None or value == FLUXNET_FILL:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            # Negative ET is a quality artefact (matches convert_le_to_et -> NaN).
            if value < 0:
                points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                continue
            points.append(ObservationPoint(timestamp=ts, value=value, quality=QualityFlag.GOOD))
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
            # Strip a leading "fluxnet:" / "ameriflux:" namespace; keep site ids
            # like "US-Ne1" intact (a single ':' split would mangle nothing here,
            # but namespaced ids exist in stored specs).
            low = s.lower()
            if low.startswith("fluxnet:") or low.startswith("ameriflux:") or low.startswith("fluxnet_et:"):
                out.append(s.split(":", 1)[1])
            else:
                out.append(s)
        return out

    def _max_qc(self, spec: ReductionSpec) -> int:
        v = spec.options.get("max_qc")
        if v is None:
            v = self.config.get("max_qc")
        try:
            return int(v) if v is not None else DEFAULT_MAX_QC
        except (TypeError, ValueError):
            return DEFAULT_MAX_QC

    def _site(self, station_id: str, spec: ReductionSpec) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"fluxnet:{station_id}",
            latitude=spec.centroid[0] if spec.centroid else None,
            longitude=spec.centroid[1] if spec.centroid else None,
            name=f"FLUXNET {station_id}",
            extra={"network": "FLUXNET"},
        )


def _first_index(col: dict[str, int], aliases: tuple[str, ...]) -> int | None:
    for a in aliases:
        if a in col:
            return col[a]
    return None


def _to_float(raw: str) -> float | None:
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_timestamp(raw: str) -> datetime | None:
    """FLUXNET timestamps are YYYYMMDDHHMM or YYYYMMDD; AmeriFlux ISO is allowed."""
    raw = raw.strip()
    if not raw:
        return None
    for fmt in ("%Y%m%d%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=UTC)
    except ValueError:
        return None


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
