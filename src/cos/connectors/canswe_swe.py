# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""CanSWE Canadian snow-water-equivalent connector (point network, anonymous).

Ports SYMFLUENCE's native ``canswe`` / ``canswe_swe`` observation handler
(``symfluence/data/observation/handlers/canswe.py``). CanSWE is the Canadian
historical in-situ SWE dataset (Vionnet et al., 2021, ESSD): a single anonymous
Zenodo NetCDF (``CanSWE-CanEEN_1928-2023_v6.nc``) holding ~2900 stations along a
``station`` dimension, each with ``lat`` / ``lon`` and a ``time × station`` SWE
field already in **mm** — which is exactly the canonical ``swe`` unit, so no
unit conversion is needed at the boundary (unlike SNOTEL's inches→mm).

Structural class: **point_network**. Like SNOTEL, COS emits one
:class:`ObservationSeries` per selected station. Station selection mirrors the
native handler: stations whose ``(lat, lon)`` fall inside ``spec.bbox`` (or, if
explicit ``spec.station_ids`` are given, only those ids), with a minimum
observation count filter (native ``CANSWE_MIN_OBSERVATIONS``, default 10).

Live download of the ~100 MB Zenodo NetCDF is not wired here; following the COS
pattern (cf. ``grace.py``), the file is supplied via config ``nc_path`` / ``path``
and the pure :meth:`reduce_file` helper opens, selects, canonicalizes, and
window-trims it with no network and no auth — so the architecture-critical
station-selection + canonicalize path is hermetically testable.
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

#: Source SWE is already mm == canonical swe unit; conversion is the identity.
#: Kept explicit so the unit-handling boundary mirrors snotel.py / grace.py.
SOURCE_TO_MM = 1.0
#: Native CANSWE_MIN_OBSERVATIONS default: drop stations with too few obs.
DEFAULT_MIN_OBSERVATIONS = 10

# Candidate variable / dimension names, mirroring the native handler's probing.
_LAT_NAMES = ("lat", "latitude", "Latitude", "LAT")
_LON_NAMES = ("lon", "longitude", "Longitude", "LON")
_SWE_NAMES = ("swe", "SWE", "snow_water_equivalent", "snw")
_STATION_DIMS = ("station", "site", "location", "obs")
_STATION_ID_NAMES = ("station_id", "site_id", "station_name", "id")


@register("canswe_swe")
class CanSWEConnector(BaseObservationConnector):
    slug = "canswe_swe"
    display_name = "CanSWE (Canadian Snow Water Equivalent)"
    kind = ObservationKind.SWE
    structural_class = "point_network"
    base_url = "https://zenodo.org"
    auth = frozenset()  # anonymous Zenodo NetCDF download

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Sites are the CanSWE stations selected for *spec*.

        Requires the CanSWE NetCDF (config ``nc_path`` / ``path``) so stations
        can be selected by bbox exactly as the native handler does. Without it
        we cannot enumerate the network, so raise.
        """
        nc_path = self._nc_path()
        if not nc_path:
            raise ConnectorError(
                self.slug,
                "CanSWE site listing needs the CanSWE NetCDF (config 'nc_path' / "
                "'path'); the ~100 MB Zenodo download is not wired here.",
            )
        selected = self._select_stations(Path(nc_path), spec)
        return [s.site for s in selected]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        nc_path = self._nc_path()
        if not nc_path:
            raise ConnectorError(
                self.slug,
                "CanSWE live fetch needs the CanSWE NetCDF (config 'nc_path' / "
                "'path') — the anonymous Zenodo download (~100 MB) is not yet "
                "wired. Supply a downloaded CanSWE NetCDF to extract stations.",
            )
        return self.reduce_file(Path(nc_path), spec, start, end)

    # -- the architecture-critical, hermetically-tested core -----------------

    def reduce_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        """Open the CanSWE NetCDF, select stations, canonicalize to mm/UTC.

        Returns one :class:`ObservationSeries` per selected station. Values are
        already in mm (canonical ``swe``); timestamps are coerced to UTC and
        window-trimmed to half-open ``[start, end)``. NaN / missing SWE become
        :attr:`QualityFlag.MISSING` points. Stations whose in-window valid-obs
        count is below ``min_observations`` are dropped (native filter).
        """
        selected = self._select_stations(nc_path, spec)
        min_obs = int(
            spec.options.get("min_observations", self.config.get("min_observations", DEFAULT_MIN_OBSERVATIONS))
        )
        start_u = _utc(start)
        end_u = _utc(end)

        out: list[ObservationSeries] = []
        for st in selected:
            points = _to_points(st.times, st.values, start_u, end_u)
            n_valid = sum(1 for p in points if p.value is not None)
            if n_valid < min_obs:
                continue
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=st.site,
                    reduction=SpatialReduction.STATION,
                    unit=KIND_UNITS[self.kind],
                    points=points,
                    source_info={
                        "source": "CanSWE",
                        "source_doi": "10.5194/essd-13-4603-2021",
                        "url": "https://zenodo.org/records/10835278",
                        "station_id": st.site.site_id,
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    # -- station selection (shared by list_sites + reduce_file) --------------

    def _select_stations(self, nc_path: Path, spec: ReductionSpec) -> list[_Station]:
        """Read the CanSWE NetCDF and return the stations matching *spec*.

        Selection mirrors the native handler: explicit ``spec.station_ids`` win
        (matched against the station-id variable); otherwise stations whose
        ``(lat, lon)`` lie inside ``spec.bbox`` are kept. With neither a bbox
        nor explicit ids, all stations are returned.
        """
        import numpy as np
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            lat_name = _first_present(ds, _LAT_NAMES)
            lon_name = _first_present(ds, _LON_NAMES)
            swe_name = _first_present(ds, _SWE_NAMES)
            if lat_name is None or lon_name is None:
                raise DataFormatError(self.slug, "CanSWE NetCDF missing lat/lon variables")
            if swe_name is None:
                raise DataFormatError(self.slug, "CanSWE NetCDF missing an SWE variable")

            station_dim = next((d for d in _STATION_DIMS if d in ds.dims), None)
            swe = ds[swe_name]
            if station_dim is None:
                # infer the non-time dim of a 2-D SWE field
                if swe.ndim == 2 and "time" in swe.dims:
                    station_dim = str(next(d for d in swe.dims if d != "time"))
                else:
                    raise DataFormatError(self.slug, "Cannot find a station dimension in CanSWE NetCDF")

            n_stations = ds.sizes[station_dim]
            lats = np.asarray(ds[lat_name].values, dtype="float64").reshape(-1)
            lons = np.asarray(ds[lon_name].values, dtype="float64").reshape(-1)
            times = np.asarray(ds["time"].values)

            id_name = _first_present(ds, _STATION_ID_NAMES)
            station_ids = _decode_ids(ds[id_name].values, n_stations) if id_name else [
                f"station_{i}" for i in range(n_stations)
            ]

            wanted = _normalize_ids(spec.station_ids)
            bbox = spec.bbox  # (lat_min, lon_min, lat_max, lon_max)

            stations: list[_Station] = []
            for idx in range(n_stations):
                lat = float(lats[idx]) if idx < lats.size else float("nan")
                lon = float(lons[idx]) if idx < lons.size else float("nan")
                sid = station_ids[idx]

                if wanted:
                    if _strip_ns(sid) not in wanted and sid not in wanted:
                        continue
                elif bbox is not None:
                    lat_min, lon_min, lat_max, lon_max = (
                        min(bbox[0], bbox[2]), min(bbox[1], bbox[3]),
                        max(bbox[0], bbox[2]), max(bbox[1], bbox[3]),
                    )
                    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                        continue

                station_values = swe.isel({station_dim: idx}).values
                values = np.asarray(station_values, dtype="float64").reshape(-1) * SOURCE_TO_MM
                stations.append(
                    _Station(
                        site=SiteRef(
                            kind="station",
                            site_id=f"canswe:{sid}",
                            latitude=lat if lat == lat else None,
                            longitude=lon if lon == lon else None,
                            name=f"CanSWE {sid}",
                            extra={"network": "CanSWE"},
                        ),
                        times=times,
                        values=values,
                    )
                )
            return stations

    def _nc_path(self) -> str | None:
        return self.config.get("nc_path") or self.config.get("path")


# -- internal value object ---------------------------------------------------


class _Station:
    """A selected CanSWE station: its SiteRef and its raw (time, SWE-mm) arrays."""

    __slots__ = ("site", "times", "values")

    def __init__(self, site: SiteRef, times, values) -> None:
        self.site = site
        self.times = times
        self.values = values


# -- pure helpers (hermetically tested, no I/O) ------------------------------


def _to_points(times, values, start_u: datetime, end_u: datetime) -> list[ObservationPoint]:
    """Build canonical SWE points (mm, UTC), window-trimmed to [start, end).

    NaN values become :attr:`QualityFlag.MISSING`; finite values are GOOD.
    """
    import math

    import numpy as np
    import pandas as pd

    points: list[ObservationPoint] = []
    for t, v in zip(times, values):
        ts = pd.Timestamp(t)
        ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
        ts = ts.to_pydatetime()
        if not (start_u <= ts < end_u):
            continue
        val = float(v)
        if math.isnan(val) or (isinstance(v, float) and np.isnan(v)):
            points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
        else:
            points.append(ObservationPoint(timestamp=ts, value=val, quality=QualityFlag.GOOD))
    return points


def _first_present(ds, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in ds:
            return name
    return None


def _decode_ids(raw, n: int) -> list[str]:
    import numpy as np

    arr = np.asarray(raw).reshape(-1)
    out: list[str] = []
    for i in range(n):
        if i < arr.size:
            v = arr[i]
            if isinstance(v, bytes):
                v = v.decode("utf-8", "replace")
            out.append(str(v).strip())
        else:
            out.append(f"station_{i}")
    return out


def _normalize_ids(ids: tuple[str, ...]) -> set[str]:
    """Lower-friction id matching: accept bare and ``canswe:``-namespaced ids."""
    return {_strip_ns(s) for s in ids if s}


def _strip_ns(sid: str) -> str:
    return sid.split(":", 1)[1] if sid.lower().startswith("canswe:") else sid


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
