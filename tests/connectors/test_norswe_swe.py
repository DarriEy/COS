"""NorSWE connector — hermetic test of the point-network / station path.

Builds a synthetic in-memory NorSWE/CanSWE-like NetCDF (station-indexed SWE in
mm) and selects stations from it; no network, no auth. This proves the
architecture-critical NetCDF-station → canonical-series path, station bbox
selection, mm pass-through units, half-open window trim, and NaN -> MISSING.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.norswe_swe import NorSWEConnector
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


@pytest.fixture
def norswe_nc(tmp_path):
    """Synthetic NorSWE NetCDF: swe (mm) over (time, station), 3 stations."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    # station 0: inside bbox; station 1: inside bbox; station 2: far outside.
    lats = np.array([51.0, 51.5, 10.0])
    lons = np.array([-115.0, -114.5, 100.0])
    station_ids = np.array(["AAA", "BBB", "CCC"])
    # swe (mm), with a NaN gap at station 0 / t=2020-01-03.
    swe = np.array(
        [
            [100.0, 200.0, 9.0],
            [110.0, 210.0, 9.0],
            [np.nan, 220.0, 9.0],
            [50.0, 60.0, 9.0],
        ]
    )
    ds = xr.Dataset(
        {
            "swe": (("time", "station"), swe),
            "lat": (("station",), lats),
            "lon": (("station",), lons),
            "station_id": (("station",), station_ids),
        },
        coords={"time": times},
    )
    path = tmp_path / "norswe_synth.nc"
    ds.to_netcdf(path)
    return path


def test_parse_file_bbox_selection_and_mm_passthrough(norswe_nc):
    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),  # selects stations 0 and 1, not 2
    )
    series = conn.parse_file(
        norswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert len(series) == 2  # station CCC is outside the bbox
    by_id = {s.site.site_id: s for s in series}
    assert set(by_id) == {"norswe:AAA", "norswe:BBB"}

    s0 = by_id["norswe:AAA"]
    assert s0.kind == ObservationKind.SWE
    assert s0.unit == "mm"
    assert s0.reduction == SpatialReduction.STATION
    assert s0.site.kind == "station"
    # mm pass-through: 100 mm stays 100 mm (no inches conversion).
    by_date = {p.timestamp.date().isoformat(): p for p in s0.points}
    assert by_date["2020-01-01"].value == pytest.approx(100.0)
    assert by_date["2020-01-01"].quality.value == "good"
    # NaN sample -> MISSING (timestamp preserved).
    assert by_date["2020-01-03"].value is None
    assert by_date["2020-01-03"].quality.value == "missing"


def test_window_trim_half_open(norswe_nc):
    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0))
    # Half-open [2020-01-02, 2020-01-03): only the 01-02 obs.
    series = conn.parse_file(
        norswe_nc, spec,
        datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC),
    )
    s = next(x for x in series if x.site.site_id == "norswe:AAA")
    dates = {p.timestamp.date().isoformat() for p in s.points}
    assert dates == {"2020-01-02"}
    # the out-of-window 2021-06-01 row is never present
    assert "2021-06-01" not in dates


def test_no_bbox_selects_all_stations(norswe_nc):
    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(domain_name="world")  # no bbox
    series = conn.parse_file(
        norswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert {s.site.site_id for s in series} == {"norswe:AAA", "norswe:BBB", "norswe:CCC"}


def test_site_carries_station_coords(norswe_nc):
    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0))
    series = conn.parse_file(
        norswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    s = next(x for x in series if x.site.site_id == "norswe:AAA")
    assert s.site.latitude == pytest.approx(51.0)
    assert s.site.longitude == pytest.approx(-115.0)
    assert s.site.extra["network"] == "NorSWE"


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = NorSWEConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))


@pytest.mark.asyncio
async def test_list_sites_returns_selected_stations(norswe_nc):
    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"norswe:AAA", "norswe:BBB"}
    assert all(s.kind == "station" for s in sites)
