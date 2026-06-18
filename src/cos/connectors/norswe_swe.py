# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""NorSWE snow-water-equivalent connector (point network, anonymous).

Ports SYMFLUENCE's ``NorSWEHandler`` (registry keys ``norswe`` / ``norswe_swe``,
``src/symfluence/data/observation/handlers/canswe.py``). NorSWE is the
Northern-Hemisphere extension of CanSWE (Vionnet et al., 2021): in-situ SWE
station observations served as a single NetCDF on Zenodo (record ``15263370``,
no authentication). It follows the CanSWE format, so the file carries:

* ``swe`` — snow water equivalent, **mm** (already the canonical ``swe`` unit);
* per-station ``lat`` / ``lon`` coordinates;
* a ``station_id`` identifier and a shared ``time`` axis.

Structurally this is a **point network** (one :class:`ObservationSeries` per
station, ``reduction = station``), exactly like SNOTEL — but the source is a
single bundled NetCDF rather than a per-station REST report. So, mirroring the
GRACE pattern, the live Zenodo download is not wired here: the connector reduces
a *supplied* NetCDF (config ``nc_path`` / ``path``), and that parse-and-select
core is hermetically unit-tested with a synthetic in-memory NetCDF.

Native parity:

* **units** — source ``swe`` is mm and canonical ``swe`` is mm, so values pass
  through unchanged (no scale factor), unlike the SNOTEL inches→mm landmine;
* **station selection** — stations whose (lat, lon) fall inside ``spec.bbox``
  ``(lat_min, lon_min, lat_max, lon_max)``, matching the handler's
  ``_find_stations_in_bbox`` inclusive bounding-box test;
* **missing data** — NaN SWE samples become ``QualityFlag.MISSING`` points (the
  native handler simply drops NaNs; COS preserves the timestamp as MISSING);
* **time** — half-open UTC ``[start, end)`` window trim.

The native handler additionally aggregates stations to a daily basin mean
(``station_mean``). COS keeps the *per-station* series (the canonical
point-network shape); a downstream consumer can take the cross-station mean to
reproduce the native aggregate. See ``parity_method`` in the port report.
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

#: Candidate variable / coordinate names in CanSWE/NorSWE NetCDFs (the native
#: handler probes the same alternatives).
_SWE_NAMES = ("swe", "SWE", "snow_water_equivalent", "snw")
_LAT_NAMES = ("lat", "latitude", "Latitude", "LAT")
_LON_NAMES = ("lon", "longitude", "Longitude", "LON")
_STATION_DIMS = ("station", "site", "location", "obs")
_STATION_ID_NAMES = ("station_id", "site_id", "station_name", "id")


@register("norswe_swe")
class NorSWEConnector(BaseObservationConnector):
    slug = "norswe_swe"
    display_name = "NorSWE Northern-Hemisphere SWE"
    kind = ObservationKind.SWE
    structural_class = "point_network"
    base_url = "https://zenodo.org"
    auth = frozenset()  # anonymous (Zenodo direct download)

    #: Zenodo record id for the bundled NorSWE NetCDF (parity with native handler).
    ZENODO_RECORD = "15263370"

    async def list_sites(self, spec: ReductionSpec) -> list[SiteRef]:
        """Stations inside ``spec.bbox`` (or all stations when no bbox given).

        Reads the supplied NetCDF (``nc_path``/``path``) and applies the same
        inclusive bounding-box selection as the native ``_find_stations_in_bbox``.
        """
        nc_path = self._nc_path()
        series = self.parse_file(nc_path, spec, datetime.min.replace(tzinfo=UTC), _MAX_DT)
        return [s.site for s in series]

    async def fetch_series(
        self,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        return self.parse_file(self._nc_path(), spec, start, end)

    # -- the architecture-critical, hermetically-tested core -----------------

    def parse_file(
        self,
        nc_path: Path,
        spec: ReductionSpec,
        start: datetime,
        end: datetime,
    ) -> list[ObservationSeries]:
        """Open a NorSWE/CanSWE NetCDF, select stations, build canonical series.

        Returns one :class:`ObservationSeries` per selected station, SWE in mm
        (pass-through; source is already mm), timestamps UTC, window-trimmed to
        half-open ``[start, end)``. NaN samples become ``MISSING`` points.
        """
        import numpy as np
        import xarray as xr

        with xr.open_dataset(nc_path) as ds:
            swe_var = _first_present(ds, _SWE_NAMES) or _first_matching(ds, ("swe", "snow"))
            if swe_var is None:
                raise DataFormatError(self.slug, f"No SWE variable found in {nc_path.name}")
            lat_var = _first_present(ds, _LAT_NAMES)
            lon_var = _first_present(ds, _LON_NAMES)
            if lat_var is None or lon_var is None:
                raise DataFormatError(self.slug, f"No lat/lon coordinates found in {nc_path.name}")

            station_dim = next((d for d in _STATION_DIMS if d in ds.dims), None)
            if station_dim is None:
                # Infer the non-time dim of the 2-D SWE variable.
                swe_dims = ds[swe_var].dims
                if "time" in swe_dims and len(swe_dims) == 2:
                    station_dim = next(d for d in swe_dims if d != "time")
            if station_dim is None:
                raise DataFormatError(self.slug, f"Could not identify station dimension in {nc_path.name}")

            n_stations = ds.sizes[station_dim]
            lats = _per_station(np.asarray(ds[lat_var].values, dtype="float64"), n_stations, station_dim, ds)
            lons = _per_station(np.asarray(ds[lon_var].values, dtype="float64"), n_stations, station_dim, ds)

            id_var = _first_present(ds, _STATION_ID_NAMES)
            station_ids = _station_id_values(ds, id_var, station_dim, n_stations) if id_var else None

            times = np.asarray(ds["time"].values)
            swe = ds[swe_var]
            # Orient SWE as (time, station).
            if swe.dims[0] == station_dim:
                values = np.asarray(swe.values, dtype="float64").T
            else:
                values = np.asarray(swe.values, dtype="float64")

        sel = self._select_stations(lats, lons, spec)
        ts_utc = [_to_utc(t) for t in times]
        start_u = _to_utc_dt(start)
        end_u = _to_utc_dt(end)

        out: list[ObservationSeries] = []
        for idx in sel:
            sid = station_ids[idx] if station_ids is not None else f"station_{idx}"
            points: list[ObservationPoint] = []
            col = values[:, idx]
            for ts, raw in zip(ts_utc, col):
                if ts is None or not (start_u <= ts < end_u):
                    continue
                if raw is None or np.isnan(raw):
                    points.append(ObservationPoint(timestamp=ts, value=None, quality=QualityFlag.MISSING))
                else:
                    points.append(ObservationPoint(timestamp=ts, value=float(raw), quality=QualityFlag.GOOD))
            out.append(
                ObservationSeries(
                    provider=self.slug,
                    kind=self.kind,
                    site=self._site(sid, float(lats[idx]), float(lons[idx])),
                    reduction=SpatialReduction.STATION,
                    unit=KIND_UNITS[self.kind],  # mm == source mm (pass-through)
                    points=points,
                    source_info={
                        "source": "NorSWE",
                        "url": f"{self.base_url}/records/{self.ZENODO_RECORD}",
                        "station": str(sid),
                    },
                    fetched_at=datetime.now(UTC),
                )
            )
        return out

    # -- helpers -------------------------------------------------------------

    def _select_stations(self, lats, lons, spec: ReductionSpec) -> list[int]:
        """Indices of stations inside ``spec.bbox`` (inclusive); all if no bbox."""
        import numpy as np

        n = len(lats)
        if spec.bbox is None:
            return list(range(n))
        lat_min, lon_min, lat_max, lon_max = spec.bbox
        lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
        lon_lo, lon_hi = min(lon_min, lon_max), max(lon_min, lon_max)
        in_box = (
            (lats >= lat_lo) & (lats <= lat_hi) & (lons >= lon_lo) & (lons <= lon_hi)
        )
        return np.where(in_box)[0].tolist()

    def _nc_path(self) -> Path:
        raw = self.config.get("nc_path") or self.config.get("path")
        if not raw:
            raise ConnectorError(
                self.slug,
                "NorSWE needs a NetCDF path (config 'nc_path' or 'path'). The live "
                f"Zenodo download (record {self.ZENODO_RECORD}) is not wired here; "
                "supply a downloaded NorSWE/CanSWE NetCDF to select stations from.",
            )
        return Path(raw)

    def _site(self, station_id: str, lat: float, lon: float) -> SiteRef:
        return SiteRef(
            kind="station",
            site_id=f"norswe:{station_id}",
            latitude=lat,
            longitude=lon,
            name=f"NorSWE {station_id}",
            extra={"network": "NorSWE"},
        )


_MAX_DT = datetime(9999, 12, 31, tzinfo=UTC)


def _first_present(ds, names) -> str | None:
    for n in names:
        if n in ds:
            return n
    return None


def _first_matching(ds, needles) -> str | None:
    for var in ds.data_vars:
        low = str(var).lower()
        if any(nd in low for nd in needles):
            return str(var)
    return None


def _per_station(arr, n_stations: int, station_dim: str, ds):
    """Collapse a per-station coordinate to a 1-D (n_stations,) array."""
    import numpy as np

    if arr.ndim == 1:
        return arr
    # 2-D (time, station) or (station, time): take the first finite per station.
    if arr.shape[0] == n_stations:
        per = np.where(np.isfinite(arr), arr, np.nan)
        return np.nanmean(per, axis=1)
    return np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=0)


def _station_id_values(ds, id_var: str, station_dim: str, n_stations: int):
    vals = ds[id_var].values
    out = []
    for i in range(n_stations):
        try:
            v = vals[i]
        except (IndexError, TypeError):
            out.append(f"station_{i}")
            continue
        if isinstance(v, bytes):
            v = v.decode("utf-8", "ignore")
        out.append(str(v).strip())
    return out


def _to_utc(value) -> datetime | None:
    import numpy as np
    import pandas as pd

    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT or (isinstance(value, float) and np.isnan(value)):
        return None
    dt = ts.to_pydatetime()
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _to_utc_dt(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
