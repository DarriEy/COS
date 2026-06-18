"""Hub'Eau water-level connector — hermetic tests of the point + gridded paths.

All offline tests use synthetic inline fixtures (no network, no auth). Parity is
asserted *by construction*: the native SYMFLUENCE ``hubeau_waterlevel`` reduction
(``resultat_obs`` mm → m via ÷1000, ``date_obs`` → UTC, null → missing) is
reimplemented inline on the SAME synthetic input and compared to the connector's
output. The point/unit/constant conversions are exact; basin-mean (cos-lat) is
tolerance-based against an unweighted native mean.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.hubeau_waterlevel import HubEauWaterLevelConnector
from cos.core.exceptions import DataFormatError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# A synthetic Hub'Eau observations_tr envelope: water level (hauteur) in mm.
# Two in-window goods, one null (-> MISSING), one out-of-window record.
MOCK_RECORDS = [
    {"date_obs": "2020-01-01T00:00:00Z", "resultat_obs": 1500.0, "code_station": "H5920010"},
    {"date_obs": "2020-01-01T06:00:00Z", "resultat_obs": 1600.0, "code_station": "H5920010"},
    {"date_obs": "2020-01-01T12:00:00Z", "resultat_obs": None, "code_station": "H5920010"},
    {"date_obs": "2021-06-01T00:00:00Z", "resultat_obs": 2000.0, "code_station": "H5920010"},
]
MOCK_BODY = json.dumps({"data": MOCK_RECORDS})

WINDOW_START = datetime(2020, 1, 1, tzinfo=UTC)
WINDOW_END = datetime(2021, 1, 1, tzinfo=UTC)


def _native_reduce(records: list[dict], start: datetime, end: datetime) -> dict[str, float | None]:
    """Inline reimplementation of the native ``HubEauWaterLevelHandler`` reduction.

    Mirrors ``process``: read ``date_obs``/``resultat_obs``, drop null values from
    the value series, divide mm by 1000 to metres. We additionally apply the same
    half-open [start, end) UTC window the connector uses so the comparison is
    apples-to-apples. Returns iso-timestamp -> metres (or None for null).
    """
    out: dict[str, float | None] = {}
    for rec in records:
        date_str = rec.get("date_obs")
        if date_str is None:
            continue
        s = str(date_str)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        ts = datetime.fromisoformat(s).astimezone(UTC)
        if not (start <= ts < end):
            continue
        value = rec.get("resultat_obs")
        if value is None:
            out[ts.isoformat()] = None  # native drops nulls; connector flags MISSING
        else:
            out[ts.isoformat()] = float(value) / 1000.0  # native: water_level_mm / 1000
    return out


def test_parse_series_mm_to_m_and_window():
    points = HubEauWaterLevelConnector.parse_series(MOCK_BODY, WINDOW_START, WINDOW_END)
    by_ts = {p.timestamp.isoformat(): p for p in points}
    # 2021-06-01 is outside [start, end).
    assert "2021-06-01T00:00:00+00:00" not in by_ts
    # mm -> m conversion (1500 mm -> 1.5 m).
    assert by_ts["2020-01-01T00:00:00+00:00"].value == pytest.approx(1.5)
    assert by_ts["2020-01-01T00:00:00+00:00"].quality.value == "good"
    assert by_ts["2020-01-01T06:00:00+00:00"].value == pytest.approx(1.6)
    # null resultat_obs -> MISSING.
    assert by_ts["2020-01-01T12:00:00+00:00"].value is None
    assert by_ts["2020-01-01T12:00:00+00:00"].quality.value == "missing"


def test_parity_by_construction_point():
    """Connector output equals the native reduction on the same synthetic input."""
    points = HubEauWaterLevelConnector.parse_series(MOCK_BODY, WINDOW_START, WINDOW_END)
    native = _native_reduce(MOCK_RECORDS, WINDOW_START, WINDOW_END)
    cos_vals = {p.timestamp.isoformat(): p.value for p in points}
    # Same set of in-window timestamps.
    assert set(cos_vals) == set(native)
    for ts, native_v in native.items():
        if native_v is None:
            assert cos_vals[ts] is None
        else:
            # mm->m conversion is exact (constant scale factor).
            assert cos_vals[ts] == pytest.approx(native_v, abs=1e-12)


def test_metre_unit_passes_through():
    body = json.dumps({"data": [
        {"date_obs": "2020-01-01T00:00:00Z", "resultat_obs": 2.5, "grandeur_hydro_unite": "m"},
    ]})
    points = HubEauWaterLevelConnector.parse_series(body, WINDOW_START, WINDOW_END)
    assert points[0].value == pytest.approx(2.5)  # already metres -> no ÷1000


def test_bare_list_payload_accepted():
    body = json.dumps(MOCK_RECORDS)  # raw list, not wrapped in {"data": ...}
    points = HubEauWaterLevelConnector.parse_series(body, WINDOW_START, WINDOW_END)
    assert len([p for p in points if p.value is not None]) == 2


def test_invalid_json_raises():
    with pytest.raises(DataFormatError):
        HubEauWaterLevelConnector.parse_series("{not json", WINDOW_START, WINDOW_END)


def test_empty_text_is_empty_series():
    assert HubEauWaterLevelConnector.parse_series("", WINDOW_START, WINDOW_END) == []


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://hubeau\.eaufrance\.fr/.*").mock(
        return_value=httpx.Response(200, json={"data": MOCK_RECORDS, "next": None})
    )
    conn = HubEauWaterLevelConnector()
    spec = ReductionSpec(domain_name="seine", station_ids=("hubeau:H5920010",))
    async with conn:
        series_list = await conn.fetch_series(spec, WINDOW_START, WINDOW_END)
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.WATER_LEVEL
    assert s.unit == "m"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "hubeau:H5920010"
    assert len([p for p in s.points if p.value is not None]) == 2


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = HubEauWaterLevelConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("H5920010", "hubeau:O5550010"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"hubeau:H5920010", "hubeau:O5550010"}


# -- gridded path -----------------------------------------------------------


@pytest.fixture
def waterlevel_nc(tmp_path):
    """A synthetic water-level raster (metres) over a small lat/lon grid."""
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-01-01", "2020-01-02", "2021-06-01"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([2.0, 3.0, 4.0])
    data = np.empty((3, 3, 3))
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 9.0
    ds = xr.Dataset(
        {"water_level": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "hubeau_wl_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_constant_field(waterlevel_nc):
    np = pytest.importorskip("numpy")
    conn = HubEauWaterLevelConnector()
    spec = ReductionSpec(
        domain_name="seine",
        bbox=(50.0, 2.0, 52.0, 4.0),
        centroid=(51.0, 3.0),
        reduction=SpatialReduction.BASIN_MEAN,
    )
    series = conn.reduce_file(waterlevel_nc, spec, WINDOW_START, WINDOW_END)
    assert series.kind == ObservationKind.WATER_LEVEL
    assert series.unit == "m"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    by_day = {p.timestamp.date().isoformat(): p.value for p in series.points}
    # Out-of-window day excluded (half-open).
    assert "2021-06-01" not in by_day
    # Parity by construction: a constant field reduces to the constant exactly,
    # regardless of cos-lat weighting (basin-mean == native unweighted mean here).
    assert by_day["2020-01-01"] == pytest.approx(1.0, abs=1e-9)
    assert by_day["2020-01-02"] == pytest.approx(2.0, abs=1e-9)
    # Cross-check against an explicit cos-lat weighted mean of the in-box layer.
    weights = np.cos(np.deg2rad(np.array([50.0, 51.0, 52.0])))
    expected = float(np.sum(np.full((3, 3), 1.0) * weights[:, None]) / np.sum(weights[:, None] * np.ones((3, 3))))
    assert by_day["2020-01-01"] == pytest.approx(expected, abs=1e-9)


def test_reduce_file_nearest_cell_and_window(waterlevel_nc):
    conn = HubEauWaterLevelConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        centroid=(51.0, 3.0),
        reduction=SpatialReduction.NEAREST_CELL,
    )
    series = conn.reduce_file(waterlevel_nc, spec, WINDOW_START, WINDOW_END)
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("hubeau_waterlevel:cell:")
    by_day = {p.timestamp.date().isoformat(): p.value for p in series.points}
    assert by_day["2020-01-01"] == pytest.approx(1.0)
    assert "2021-06-01" not in by_day


def test_reduce_file_fill_value_to_missing(waterlevel_nc):
    """A sentinel fill in source units masks to NaN -> QualityFlag.MISSING."""
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    # Rewrite the fixture's first timestep to the fill sentinel.
    with xr.open_dataset(waterlevel_nc) as ds:
        data = np.asarray(ds["water_level"].values, dtype="float64")
        lats = ds["lat"].values
        lons = ds["lon"].values
        times = ds["time"].values
    data[0] = -9999.0
    filled = waterlevel_nc.parent / "filled.nc"
    xr.Dataset(
        {"water_level": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    ).to_netcdf(filled)

    conn = HubEauWaterLevelConnector()
    spec = ReductionSpec(
        domain_name="seine",
        bbox=(50.0, 2.0, 52.0, 4.0),
        centroid=(51.0, 3.0),
        reduction=SpatialReduction.BASIN_MEAN,
        options={"fill_value": -9999.0},
    )
    series = conn.reduce_file(filled, spec, WINDOW_START, WINDOW_END)
    by_day = {p.timestamp.date().isoformat(): p for p in series.points}
    assert by_day["2020-01-01"].value is None
    assert by_day["2020-01-01"].quality == "missing"
    assert by_day["2020-01-02"].value == pytest.approx(2.0)
    assert by_day["2020-01-02"].quality == "good"


def test_reduce_file_source_scale_mm_to_m(tmp_path):
    """A millimetre-source raster is scaled to metres via source_scale_to_m."""
    np = pytest.importorskip("numpy")
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-01-01"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([2.0, 3.0])
    data = np.full((1, 2, 2), 1500.0)  # millimetres
    path = tmp_path / "wl_mm.nc"
    xr.Dataset(
        {"water_level": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    ).to_netcdf(path)

    conn = HubEauWaterLevelConnector()
    spec = ReductionSpec(
        domain_name="seine",
        centroid=(50.5, 2.5),
        reduction=SpatialReduction.NEAREST_CELL,
        options={"source_scale_to_m": 0.001},
    )
    series = conn.reduce_file(path, spec, WINDOW_START, WINDOW_END)
    assert series.points[0].value == pytest.approx(1.5)  # 1500 mm * 0.001 -> 1.5 m


# -- live smoke -------------------------------------------------------------


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_hubeau_waterlevel():
    """LIVE smoke against the real anonymous Hub'Eau endpoint (may be geo-fenced).

    Run with: pytest -m network tests/connectors/test_hubeau_waterlevel.py -k live
    """
    conn = HubEauWaterLevelConnector()
    spec = ReductionSpec(domain_name="seine", station_ids=("H5920010",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 8, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m"
