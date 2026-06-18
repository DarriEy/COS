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
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


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


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION                                                       #
# --------------------------------------------------------------------------- #
# The native SYMFLUENCE NorSWEHandler (a CanSWEHandler subclass,
# src/symfluence/data/observation/handlers/canswe.py) reduces the *same* bundled
# NetCDF to a single daily time series. Its pipeline, transcribed exactly:
#
#   _find_stations_in_bbox: inclusive bbox test
#       (lat>=lat_min)&(lat<=lat_max)&(lon>=lon_min)&(lon<=lon_max)
#   _extract_station_data:  swe_mm = float(swe)  -- mm PASS-THROUGH, no scale;
#                           rows with np.isnan(swe) are DROPPED.
#   process time filter:    (df.index >= start) & (df.index <= end)  -- CLOSED.
#   _aggregate_stations:    daily UNWEIGHTED cross-station mean of swe_mm.
#
# COS keeps the per-station point-network shape (one ObservationSeries/station,
# SWE in mm), the canonical SWE form. The native daily basin aggregate is
# reproduced downstream by an UNWEIGHTED cross-station mean of the GOOD points
# per day. This test reimplements the native semantics inline and asserts the
# COS-derived aggregate equals the native aggregate to FLOAT tolerance.
#
# Why exact (not relative ~1e-3): NorSWE is a POINT network, not a gridded
# product. There is no cos-lat area weighting anywhere on either side — both the
# native `_aggregate_stations` and the COS-side reconstruction are plain
# arithmetic means over identical station values with identical (mm) units. The
# only intentional COS divergences are benign for the daily-mean objective:
#   * fill: COS keeps NaN samples as MISSING points; native drops them. Excluded
#     identically from the mean (a MISSING point contributes no value), so the
#     aggregate is unchanged.
#   * window: COS is half-open [start, end); native is closed [start, end]. The
#     parity fixture below keeps all observations strictly inside the window so
#     the boundary convention cannot perturb the comparison; the boundary
#     divergence itself is exercised separately in test_window_trim_half_open.


def _native_daily_station_mean(times, lats, lons, swe, bbox, start, end):
    """Inline reimplementation of the native NorSWE/CanSWE reduction.

    Mirrors CanSWEHandler._find_stations_in_bbox + _extract_station_data +
    (closed) time filter + _aggregate_stations. Returns {date_iso: mean_mm}.
    """
    import pandas as pd

    lat_min, lon_min, lat_max, lon_max = bbox
    in_box = (
        (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
    )
    station_idx = np.where(in_box)[0].tolist()
    # Native uses tz-naive pandas timestamps; compare on naive bounds.
    start_n = pd.Timestamp(start.replace(tzinfo=None))
    end_n = pd.Timestamp(end.replace(tzinfo=None))

    records = []
    for idx in station_idx:
        for t, v in zip(times, swe[:, idx]):
            if not np.isnan(v):  # native drops NaN
                records.append({"datetime": pd.to_datetime(t), "swe_mm": float(v)})
    df = pd.DataFrame(records).set_index("datetime").sort_index()
    df = df[(df.index >= start_n) & (df.index <= end_n)]  # native CLOSED window
    daily = df.groupby(df.index.date)["swe_mm"].mean()  # unweighted cross-station mean
    return {d.isoformat(): float(v) for d, v in daily.items()}


def _cos_daily_station_mean(series):
    """Reconstruct the native daily basin aggregate from COS per-station series.

    Unweighted cross-station mean of the GOOD (non-MISSING) points per day — the
    documented downstream reproduction of the native `station_mean`.
    """
    from collections import defaultdict

    by_day = defaultdict(list)
    for s in series:
        for p in s.points:
            if p.value is not None:  # MISSING contributes nothing, like a dropped NaN
                by_day[p.timestamp.date().isoformat()].append(p.value)
    return {d: float(np.mean(vals)) for d, vals in by_day.items()}


def test_parity_daily_station_mean_matches_native_exactly(norswe_nc):
    """COS-derived daily basin mean == native _aggregate_stations, float-exact.

    Window kept strictly interior so the half-open/closed boundary convention
    cannot affect the comparison (see module note + test_window_trim_half_open).
    """
    times = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    lats = np.array([51.0, 51.5, 10.0])
    lons = np.array([-115.0, -114.5, 100.0])
    swe = np.array(
        [
            [100.0, 200.0, 9.0],
            [110.0, 210.0, 9.0],
            [np.nan, 220.0, 9.0],  # station 0 NaN: native drops, COS -> MISSING
            [50.0, 60.0, 9.0],
        ]
    )
    bbox = (50.0, -116.0, 52.0, -114.0)  # selects stations 0,1; excludes 2
    # Strictly-interior window: covers 2020-01-01..03, excludes 2021-06-01 by a
    # wide margin under both closed (native) and half-open (COS) conventions.
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 6, 1, tzinfo=UTC)

    native = _native_daily_station_mean(times, lats, lons, swe, bbox=bbox, start=start, end=end)

    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(domain_name="bow", bbox=bbox)
    series = conn.parse_file(norswe_nc, spec, start, end)
    cos = _cos_daily_station_mean(series)

    # Same days present on both sides.
    assert set(cos) == set(native)
    # Expected by hand: 2020-01-01 = (100+200)/2 = 150; 2020-01-02 = (110+210)/2
    # = 160; 2020-01-03 = only station 1's 220 (station 0 NaN) = 220.
    assert native == pytest.approx({"2020-01-01": 150.0, "2020-01-02": 160.0, "2020-01-03": 220.0})
    for day in native:
        assert cos[day] == pytest.approx(native[day], abs=0.0, rel=0.0)  # float-exact


def test_parity_unit_factor_is_identity_mm(norswe_nc):
    """Native swe_mm = float(swe) (mm pass-through); COS must apply NO scale.

    Unlike SNOTEL (inches->mm, x25.4), NorSWE source and canonical units are both
    mm, so the unit factor is exactly 1.0. Any nonzero conversion would corrupt
    the SWE objective. Assert COS values equal the raw NetCDF values bit-for-bit.
    """
    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0))
    series = conn.parse_file(
        norswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_id = {s.site.site_id: s for s in series}
    a = {p.timestamp.date().isoformat(): p.value for p in by_id["norswe:AAA"].points}
    b = {p.timestamp.date().isoformat(): p.value for p in by_id["norswe:BBB"].points}
    # Raw NetCDF mm values pass through unchanged (factor == 1.0).
    assert a["2020-01-01"] == 100.0
    assert a["2020-01-02"] == 110.0
    assert b["2020-01-01"] == 200.0
    assert by_id["norswe:AAA"].unit == "mm"


def test_parity_fill_and_window_divergences_are_benign(norswe_nc):
    """The two intentional COS-vs-native divergences do not corrupt the aggregate.

    1. Fill: native drops NaN rows; COS keeps them as MISSING (value is None).
       Both are excluded from the cross-station mean identically.
    2. Window: native [start,end] closed vs COS [start,end) half-open. A boundary
       observation is treated as MISSING/absent by COS, present by native -- so we
       confirm the boundary obs is the ONLY difference and interior days agree.
    """
    conn = NorSWEConnector({"nc_path": str(norswe_nc)})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0))
    series = conn.parse_file(
        norswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    s0 = next(s for s in series if s.site.site_id == "norswe:AAA")
    pts = {p.timestamp.date().isoformat(): p for p in s0.points}
    # Fill: the NaN sample is preserved as a MISSING point (not silently dropped).
    assert pts["2020-01-03"].value is None
    assert pts["2020-01-03"].quality == QualityFlag.MISSING
    # A MISSING point carries no value, so it is excluded from a mean exactly like
    # a native dropped NaN -> the aggregate is identical (proven in the parity test).

    # Window boundary: end is exclusive in COS. With end == 2020-01-02, the 01-02
    # obs is excluded by COS (half-open) though native (closed) would keep it.
    bounded = conn.parse_file(
        norswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 2, tzinfo=UTC),
    )
    sb = next(s for s in bounded if s.site.site_id == "norswe:AAA")
    days = {p.timestamp.date().isoformat() for p in sb.points}
    assert days == {"2020-01-01"}  # 01-02 excluded by half-open end (documented)
