"""CanSWE connector — hermetic test of the point-network / station path.

Builds a synthetic in-memory CanSWE-like NetCDF (``time × station`` SWE in mm,
per-station ``lat`` / ``lon`` / ``station_id``) and selects + canonicalizes it;
no network, no auth. SWE is already mm == canonical ``swe`` unit, so this also
asserts the identity unit handling (contrast with SNOTEL's inches→mm).
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from cos.connectors.canswe_swe import CanSWEConnector
from cos.core.exceptions import ConnectorError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


@pytest.fixture
def canswe_nc(tmp_path):
    """Synthetic CanSWE NetCDF: 3 stations, SWE (mm) along (time, station)."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    # station 0 inside bbox, station 1 inside bbox, station 2 OUTSIDE bbox.
    lats = np.array([51.0, 51.5, 60.0])
    lons = np.array([-115.0, -114.5, -100.0])
    station_id = np.array(["BOW1", "BOW2", "FAR3"], dtype=object)
    swe = np.array(
        [
            [100.0, 200.0, 999.0],   # 2020-01-01
            [110.0, np.nan, 999.0],  # 2020-01-02 (station1 missing)
            [120.0, 210.0, 999.0],   # 2020-01-03
            [50.0, 50.0, 50.0],      # 2021-06-01 (out of window)
        ]
    )
    ds = xr.Dataset(
        {
            "swe": (("time", "station"), swe),
            "lat": (("station",), lats),
            "lon": (("station",), lons),
            "station_id": (("station",), station_id),
        },
        coords={"time": times},
    )
    path = tmp_path / "canswe_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec_bbox(min_obs=1):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),  # (lat_min, lon_min, lat_max, lon_max)
        centroid=(51.25, -114.75),
        options={"min_observations": min_obs},
    )


def test_reduce_file_selects_bbox_stations_units_mm(canswe_nc):
    conn = CanSWEConnector()
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    # Only the two in-bbox stations (FAR3 excluded).
    ids = {s.site.site_id for s in series}
    assert ids == {"canswe:BOW1", "canswe:BOW2"}
    s0 = next(s for s in series if s.site.site_id == "canswe:BOW1")
    assert s0.kind == ObservationKind.SWE
    assert s0.unit == "mm"  # source mm -> canonical mm (identity)
    assert s0.reduction == SpatialReduction.STATION
    assert s0.site.kind == "station"
    # mm carries through unchanged
    by_date = {p.timestamp.date().isoformat(): p for p in s0.points}
    assert by_date["2020-01-01"].value == pytest.approx(100.0)
    assert by_date["2020-01-01"].quality.value == "good"


def test_window_trim_half_open(canswe_nc):
    conn = CanSWEConnector()
    # Half-open [2020-01-01, 2020-01-03): includes 01-01, 01-02; excludes 01-03.
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC),
    )
    s0 = next(s for s in series if s.site.site_id == "canswe:BOW1")
    dates = {p.timestamp.date().isoformat() for p in s0.points}
    assert "2020-01-01" in dates
    assert "2020-01-02" in dates
    assert "2020-01-03" not in dates
    assert "2021-06-01" not in dates


def test_nan_swe_becomes_missing(canswe_nc):
    conn = CanSWEConnector()
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    s1 = next(s for s in series if s.site.site_id == "canswe:BOW2")
    by_date = {p.timestamp.date().isoformat(): p for p in s1.points}
    assert by_date["2020-01-02"].value is None
    assert by_date["2020-01-02"].quality.value == "missing"
    assert by_date["2020-01-03"].value == pytest.approx(210.0)


def test_min_observations_filter_drops_sparse_stations(canswe_nc):
    conn = CanSWEConnector()
    # BOW2 has only 2 valid (non-NaN) obs in window -> require 3 to drop it.
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(min_obs=3),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    ids = {s.site.site_id for s in series}
    assert ids == {"canswe:BOW1"}  # BOW1 has 3 valid obs, BOW2 only 2


def test_explicit_station_ids_select_one(canswe_nc):
    conn = CanSWEConnector()
    spec = ReductionSpec(
        domain_name="bow",
        station_ids=("canswe:BOW2",),
        options={"min_observations": 1},
    )
    series = conn.reduce_file(
        canswe_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert {s.site.site_id for s in series} == {"canswe:BOW2"}


def test_list_sites_returns_bbox_stations(canswe_nc):
    conn = CanSWEConnector(config={"nc_path": str(canswe_nc)})
    spec = _spec_bbox()
    import asyncio

    sites = asyncio.run(conn.list_sites(spec))
    assert {s.site_id for s in sites} == {"canswe:BOW1", "canswe:BOW2"}
    bow1 = next(s for s in sites if s.site_id == "canswe:BOW1")
    assert bow1.latitude == pytest.approx(51.0)
    assert bow1.longitude == pytest.approx(-115.0)


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION                                                        #
#                                                                              #
# Reimplements the SYMFLUENCE native CanSWEHandler semantics inline            #
# (symfluence/data/observation/handlers/canswe.py) and asserts the COS         #
# connector reproduces them on the SAME synthetic input.                       #
#                                                                              #
# Native semantics that matter for the SWE point-network objective:            #
#   * UNIT: SWE is read straight through as mm (`swe_mm = float(swe)`), no      #
#     conversion. COS uses SOURCE_TO_MM == 1.0. -> IDENTITY, exact.            #
#   * SELECTION: bbox is INCLUSIVE on all four edges                           #
#       (lats>=lat_min)&(lats<=lat_max)&(lons>=lon_min)&(lons<=lon_max).       #
#     COS clamps min/max of the bbox then applies the same inclusive test.     #
#   * FILL: native DROPS NaN rows (`if not np.isnan(swe): append`). COS keeps  #
#     the timestamp as a MISSING point (value=None). The set of FINITE values  #
#     is identical; only the missing-row representation differs (benign — the  #
#     min-obs filter and any downstream stat run over the finite values).      #
#   * MIN_OBS: native keeps stations whose non-NaN obs count >= min_obs        #
#     (groupby('station_id').size()). COS counts points with value not None,   #
#     i.e. the same non-NaN obs count. -> identical filter.                    #
#                                                                              #
# Documented divergence: WINDOW EDGE. Native trims CLOSED [start, end]         #
# (df.index >= start & df.index <= end); COS trims HALF-OPEN [start, end).     #
# Parity is asserted on the strictly-interior values (which both keep          #
# identically) and the divergence is exercised explicitly so it is on record.  #
# --------------------------------------------------------------------------- #


def _native_extract_mm(times, swe_station, station_lat, station_lon, bbox, start, end):
    """Reimplement the native per-station extract+filter on one station's array.

    Mirrors CanSWEHandler._find_stations_in_bbox + _extract_station_data +
    the closed-interval time filter in .process(). Returns the list of
    (timestamp_date_iso, swe_mm) the native handler would have RETAINED for
    this station, or None if the station is outside the bbox.
    """
    lat_min, lon_min, lat_max, lon_max = bbox
    lat_lo, lat_hi = min(lat_min, lat_max), max(lat_min, lat_max)
    lon_lo, lon_hi = min(lon_min, lon_max), max(lon_min, lon_max)
    # native: inclusive bbox on station lat/lon
    if not (lat_lo <= station_lat <= lat_hi and lon_lo <= station_lon <= lon_hi):
        return None
    kept = []
    for t, swe in zip(times, swe_station):
        ts = pd.Timestamp(t)
        # native unit handling: straight mm, no factor
        swe_mm = float(swe)
        # native fill rule: drop NaN rows entirely
        if np.isnan(swe_mm):
            continue
        # native time filter: CLOSED [start, end]
        if not (start <= ts.tz_localize("UTC") <= end):
            continue
        kept.append((ts.date().isoformat(), swe_mm))
    return kept


def test_parity_unit_and_values_identity(canswe_nc):
    """COS per-station finite SWE values == native finite SWE values (exact, mm)."""
    times = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    swe = np.array(
        [
            [100.0, 200.0, 999.0],
            [110.0, np.nan, 999.0],
            [120.0, 210.0, 999.0],
            [50.0, 50.0, 50.0],
        ]
    )
    lats = np.array([51.0, 51.5, 60.0])
    lons = np.array([-115.0, -114.5, -100.0])
    bbox = (50.0, -116.0, 52.0, -114.0)
    # Use a half-open window whose end excludes the out-of-range obs in BOTH
    # conventions, so the closed/half-open edge difference cannot confound the
    # value-identity assertion (window-edge divergence is tested separately).
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)

    conn = CanSWEConnector()
    series = conn.reduce_file(canswe_nc, _spec_bbox(min_obs=1), start, end)
    cos_by_station = {
        s.site.site_id.split(":", 1)[1]: {
            p.timestamp.date().isoformat(): p.value
            for p in s.points
            if p.value is not None  # COS finite values only
        }
        for s in series
    }

    station_ids = ["BOW1", "BOW2", "FAR3"]
    native_by_station = {}
    for j, sid in enumerate(station_ids):
        kept = _native_extract_mm(
            times, swe[:, j], lats[j], lons[j], bbox,
            pd.Timestamp(start), pd.Timestamp(end),
        )
        if kept is None:
            continue  # outside bbox (FAR3)
        native_by_station[sid] = dict(kept)

    # Same set of stations selected.
    assert set(cos_by_station) == set(native_by_station) == {"BOW1", "BOW2"}
    # Same finite (date -> mm) values, EXACTLY (identity unit handling).
    for sid in native_by_station:
        assert cos_by_station[sid] == native_by_station[sid]
        for date, val in native_by_station[sid].items():
            assert cos_by_station[sid][date] == pytest.approx(val, abs=0.0, rel=0.0)


def test_parity_min_obs_filter_matches_native(canswe_nc):
    """COS min-obs drop == native groupby-size drop on the same finite counts."""
    times = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
        dtype="datetime64[ns]",
    )
    swe = np.array(
        [[100.0, 200.0, 999.0], [110.0, np.nan, 999.0],
         [120.0, 210.0, 999.0], [50.0, 50.0, 50.0]]
    )
    lats = np.array([51.0, 51.5, 60.0])
    lons = np.array([-115.0, -114.5, -100.0])
    bbox = (50.0, -116.0, 52.0, -114.0)
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)
    min_obs = 3

    # Native: count non-NaN, in-window, in-bbox obs per station; keep >= min_obs.
    native_kept = set()
    for j, sid in enumerate(["BOW1", "BOW2", "FAR3"]):
        kept = _native_extract_mm(times, swe[:, j], lats[j], lons[j], bbox,
                                  pd.Timestamp(start), pd.Timestamp(end))
        if kept is not None and len(kept) >= min_obs:
            native_kept.add(sid)

    conn = CanSWEConnector()
    series = conn.reduce_file(canswe_nc, _spec_bbox(min_obs=min_obs), start, end)
    cos_kept = {s.site.site_id.split(":", 1)[1] for s in series}

    assert cos_kept == native_kept == {"BOW1"}


def test_parity_constant_single_cell_field_exact(tmp_path):
    """One station, all-constant finite SWE: COS == native to float tolerance.

    The 'single-cell / constant field' parity floor: with no NaN and no
    aggregation ambiguity, the COS canonical series must equal the native
    retained values bit-for-bit (both are just float(swe), mm).
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    swe = np.array([[42.0], [42.0], [42.0]])  # (time, station=1), constant mm
    ds = xr.Dataset(
        {
            "swe": (("time", "station"), swe),
            "lat": (("station",), np.array([51.0])),
            "lon": (("station",), np.array([-115.0])),
            "station_id": (("station",), np.array(["C1"], dtype=object)),
        },
        coords={"time": times},
    )
    path = tmp_path / "const.nc"
    ds.to_netcdf(path)

    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)
    conn = CanSWEConnector()
    series = conn.reduce_file(path, _spec_bbox(min_obs=1), start, end)
    assert len(series) == 1
    cos_vals = [p.value for p in series[0].points if p.value is not None]

    native_vals = [
        v for _, v in _native_extract_mm(
            times, swe[:, 0], 51.0, -115.0,
            (50.0, -116.0, 52.0, -114.0), pd.Timestamp(start), pd.Timestamp(end),
        )
    ]
    assert cos_vals == native_vals == [42.0, 42.0, 42.0]


def test_parity_fill_rule_divergence_is_benign(canswe_nc):
    """COS keeps NaN rows as MISSING; native drops them. Finite values agree.

    This pins the ONE representational divergence so it is explicit: COS emits
    a MISSING point where native emits no row. The finite-value content (what
    every downstream SWE statistic consumes) is identical, so the divergence
    does not corrupt the kind's objective.
    """
    conn = CanSWEConnector()
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(min_obs=1),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    bow2 = next(s for s in series if s.site.site_id == "canswe:BOW2")
    by_date = {p.timestamp.date().isoformat(): p for p in bow2.points}
    # COS keeps the NaN day as an explicit MISSING (native would have no row).
    assert by_date["2020-01-02"].value is None
    assert by_date["2020-01-02"].quality.value == "missing"
    # Finite content matches native exactly.
    cos_finite = {d: p.value for d, p in by_date.items() if p.value is not None}
    native_finite = dict(_native_extract_mm(
        np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
                 dtype="datetime64[ns]"),
        np.array([200.0, np.nan, 210.0, 50.0]),
        51.5, -114.5, (50.0, -116.0, 52.0, -114.0),
        pd.Timestamp(datetime(2020, 1, 1, tzinfo=UTC)),
        pd.Timestamp(datetime(2021, 1, 1, tzinfo=UTC)),
    ))
    assert cos_finite == native_finite


def test_window_edge_closed_vs_halfopen_divergence(canswe_nc):
    """Document COS half-open [start,end) vs native closed [start,end] at end.

    The obs at exactly `end` is KEPT by native (closed) but DROPPED by COS
    (half-open). This is the one window-semantics difference; it is recorded
    here so the parity grade can account for it (interior values are unaffected).
    """
    conn = CanSWEConnector()
    # end == 2020-01-03: COS half-open excludes the 01-03 obs.
    series = conn.reduce_file(
        canswe_nc, _spec_bbox(min_obs=1),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC),
    )
    bow1 = next(s for s in series if s.site.site_id == "canswe:BOW1")
    cos_dates = {p.timestamp.date().isoformat() for p in bow1.points if p.value is not None}
    assert "2020-01-03" not in cos_dates  # COS half-open drops the boundary

    # Native (closed) on the same input WOULD keep 01-03:
    native = dict(_native_extract_mm(
        np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2021-06-01"],
                 dtype="datetime64[ns]"),
        np.array([100.0, 110.0, 120.0, 50.0]),
        51.0, -115.0, (50.0, -116.0, 52.0, -114.0),
        pd.Timestamp(datetime(2020, 1, 1, tzinfo=UTC)),
        pd.Timestamp(datetime(2020, 1, 3, tzinfo=UTC)),
    ))
    assert "2020-01-03" in native  # native closed keeps the boundary
    # Strictly-interior values agree.
    assert {"2020-01-01", "2020-01-02"} <= set(cos_dates)
    assert {"2020-01-01", "2020-01-02"} <= set(native)


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = CanSWEConnector()
    spec = _spec_bbox()
    with pytest.raises(ConnectorError, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
async def test_fetch_series_with_ncpath_in_config(canswe_nc):
    conn = CanSWEConnector(config={"nc_path": str(canswe_nc), "min_observations": 1})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.25, -114.75))
    series = await conn.fetch_series(
        spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert {s.site.site_id for s in series} == {"canswe:BOW1", "canswe:BOW2"}
    assert all(s.unit == "mm" for s in series)
