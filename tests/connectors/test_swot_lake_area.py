"""SWOT lake-area connector — hermetic test of the per-lake Hydrocron path.

SWOT lake area has NO SYMFLUENCE native, so this is *spec-validated*: the
assertions reproduce the published Hydrocron LakeSP product spec on a synthetic
inline fixture — the km² source unit normalized to the canonical SURFACE_WATER
``fraction`` by a reference extent, the -999999999999.0 fill sentinel, the
``no_data`` time_str placeholder, the non-negative area range, the half-open UTC
window, and the gridded reduction — with no network and no auth.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.swot_lake_area import (
    DEFAULT_REFERENCE_AREA_KM2,
    SWOT_FILL_VALUE,
    VALID_AREA_RANGE_KM2,
    SWOTLakeAreaConnector,
)
from cos.core.exceptions import DataFormatError
from cos.core.models import KIND_UNITS, ObservationKind, ReductionSpec, SpatialReduction

# Hydrocron lake CSV: header, valid rows (km²), a fill-sentinel row, a 'no_data'
# row, and an out-of-window row. area_total is in km².
MOCK_CSV = """\
lake_id,time_str,area_total,area_total_units
6350900223,2024-01-05T12:00:00Z,12.50,km2
6350900223,2024-01-12T12:00:00Z,15.00,km2
6350900223,2024-01-19T12:00:00Z,-999999999999.0,km2
6350900223,no_data,-999999999999.0,km2
6350900223,2025-06-01T12:00:00Z,20.00,km2
"""


def test_canonical_unit_is_fraction():
    """Spec: SURFACE_WATER canonical unit is the dimensionless 'fraction'."""
    assert KIND_UNITS[ObservationKind.SURFACE_WATER] == "fraction"
    assert SWOTLakeAreaConnector.kind == ObservationKind.SURFACE_WATER


def test_parse_identity_reference_and_window():
    """Spec: with reference 1.0 km², fraction == area in km²; window is half-open."""
    assert DEFAULT_REFERENCE_AREA_KM2 == 1.0
    points = SWOTLakeAreaConnector.parse_timeseries(
        MOCK_CSV,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # 2025-06-01 is outside [start, end); 'no_data' row is dropped (no timestamp).
    assert "2025-06-01" not in by_date
    assert "no_data" not in by_date
    # Identity reference (1.0 km²): fraction numerically equals km² area.
    assert by_date["2024-01-05"].value == pytest.approx(12.50)
    assert by_date["2024-01-05"].quality.value == "good"
    assert by_date["2024-01-12"].value == pytest.approx(15.00)


def test_reference_area_normalizes_to_fraction():
    """Spec: fraction = area_total_km2 / reference_area_km2 (the honest unit choice)."""
    # Reference 25 km² (the lake's max extent) -> 12.5 km² is 0.5 of full extent.
    points = SWOTLakeAreaConnector.parse_timeseries(
        MOCK_CSV,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        reference_area_km2=25.0,
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert by_date["2024-01-05"].value == pytest.approx(0.5)
    assert by_date["2024-01-12"].value == pytest.approx(0.6)


def test_fill_sentinel_maps_to_missing():
    """Spec: -999999999999.0 is the SWOT no-observation fill -> MISSING/None."""
    assert SWOT_FILL_VALUE == -999999999999.0
    points = SWOTLakeAreaConnector.parse_timeseries(
        MOCK_CSV,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert by_date["2024-01-19"].value is None
    assert by_date["2024-01-19"].quality.value == "missing"


def test_no_data_time_str_is_dropped():
    """A 'no_data' time_str placeholder has no anchor timestamp and is skipped."""
    points = SWOTLakeAreaConnector.parse_timeseries(
        MOCK_CSV,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    # 2 valid + 1 fill (still inside window) = 3 anchored points; 'no_data' and
    # the out-of-window row are absent.
    assert len(points) == 3


def test_out_of_range_area_masked_to_missing():
    """Spec: a finite source area outside the physical band -> MISSING."""
    lo, hi = VALID_AREA_RANGE_KM2
    csv_text = (
        "lake_id,time_str,area_total\n"
        f"1,2024-02-01T00:00:00Z,{hi * 10:.1f}\n"   # absurdly large -> out of range
        "1,2024-02-02T00:00:00Z,-5.0\n"             # negative area -> out of range
        "1,2024-02-03T00:00:00Z,8.0\n"              # plausible -> good
    )
    assert lo == 0.0
    points = SWOTLakeAreaConnector.parse_timeseries(
        csv_text, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert by_date["2024-02-01"].value is None
    assert by_date["2024-02-01"].quality.value == "missing"
    assert by_date["2024-02-02"].value is None
    assert by_date["2024-02-02"].quality.value == "missing"
    assert by_date["2024-02-03"].value == pytest.approx(8.0)


def test_non_km2_units_rejected():
    """Spec contract: an area_total_units other than km² must not be mis-scaled."""
    csv_text = (
        "lake_id,time_str,area_total,area_total_units\n"
        "1,2024-02-01T00:00:00Z,12.5,m2\n"
    )
    with pytest.raises(DataFormatError):
        SWOTLakeAreaConnector.parse_timeseries(
            csv_text, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_nonpositive_reference_rejected():
    """A reference extent <= 0 cannot normalize an area to a fraction."""
    with pytest.raises(DataFormatError):
        SWOTLakeAreaConnector.parse_timeseries(
            MOCK_CSV,
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
            reference_area_km2=0.0,
        )


def test_missing_required_column_raises():
    with pytest.raises(DataFormatError):
        SWOTLakeAreaConnector.parse_timeseries(
            "lake_id,area_total\n1,12.5\n",  # no time_str column
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_missing_area_column_raises():
    with pytest.raises(DataFormatError):
        SWOTLakeAreaConnector.parse_timeseries(
            "lake_id,time_str\n1,2024-02-01T00:00:00Z\n",  # no area_total column
            datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_json_envelope_is_unwrapped():
    """Hydrocron may wrap the CSV in {'results': {'csv': ...}}; parser unwraps it."""
    import json

    body = json.dumps({"status": "200 OK", "results": {"csv": MOCK_CSV}})
    points = SWOTLakeAreaConnector.parse_timeseries(
        body, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert len(points) == 3
    assert points[0].value == pytest.approx(12.50)


def test_column_order_independent():
    """Columns are matched by header name, not position."""
    csv_text = (
        "time_str,area_total,lake_id\n"
        "2024-03-01T00:00:00Z,7.5,1\n"
    )
    points = SWOTLakeAreaConnector.parse_timeseries(
        csv_text, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert len(points) == 1
    assert points[0].value == pytest.approx(7.5)


@pytest.mark.asyncio
async def test_fetch_series_builds_lake_series(monkeypatch):
    conn = SWOTLakeAreaConnector()

    async def _fake_fetch(self, feature, feature_id, start, end):  # noqa: ANN001
        assert feature == "PriorLake"
        assert feature_id == "6350900223"
        return MOCK_CSV

    monkeypatch.setattr(SWOTLakeAreaConnector, "_fetch_timeseries", _fake_fetch)
    spec = ReductionSpec(domain_name="lakes", station_ids=("swot:6350900223",))
    series_list = await conn.fetch_series(
        spec, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
    )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.SURFACE_WATER
    assert s.unit == "fraction"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "swot:6350900223"
    assert len([p for p in s.points if p.value is not None]) == 2


@pytest.mark.asyncio
async def test_fetch_series_honours_reference_option(monkeypatch):
    conn = SWOTLakeAreaConnector()

    async def _fake_fetch(self, feature, feature_id, start, end):  # noqa: ANN001
        return MOCK_CSV

    monkeypatch.setattr(SWOTLakeAreaConnector, "_fetch_timeseries", _fake_fetch)
    spec = ReductionSpec(
        domain_name="lakes",
        station_ids=("6350900223",),
        options={"reference_area_km2": 25.0},
    )
    series_list = await conn.fetch_series(
        spec, datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)
    )
    s = series_list[0]
    good = [p for p in s.points if p.value is not None]
    assert good[0].value == pytest.approx(0.5)  # 12.5 / 25
    assert s.source_info["reference_area_km2"] == "25"


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = SWOTLakeAreaConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("6350900223", "swot:9999999999"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"swot:6350900223", "swot:9999999999"}
    assert all(s.kind == "station" for s in sites)


# -- gridded path (synthetic area / inundation NetCDF) -----------------------


@pytest.fixture
def swot_area_nc(tmp_path):
    """A synthetic SWOT-like lake-area NetCDF (km²), with a fill cell to be masked."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2024-06-15", "2024-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.full((2, 3, 3), 10.0)  # km²
    data[0, 0, 0] = SWOT_FILL_VALUE  # one fill cell -> masked to NaN before mean
    ds = xr.Dataset(
        {"area_total": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "swot_area_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_fraction(swot_area_nc):
    conn = SWOTLakeAreaConnector({"reference_area_km2": 20.0})
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        swot_area_nc, spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SURFACE_WATER
    assert series.unit == "fraction"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Valid cells are 10 km² (fill cell masked) / reference 20 km² -> 0.5 fraction.
    for p in series.points:
        assert p.value == pytest.approx(0.5, abs=1e-6)
        assert p.quality.value == "good"


def test_reduce_file_window_trim_half_open(swot_area_nc):
    conn = SWOTLakeAreaConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2024-06-01, 2024-07-15): includes 06-15, excludes 07-15.
    series = conn.reduce_file(
        swot_area_nc, spec,
        datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 7, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2024, 6) in months
    assert (2024, 7) not in months


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_swot_lake():
    """LIVE smoke against the real anonymous Hydrocron endpoint.

    Run with: pytest -m network tests/connectors/test_swot_lake_area.py -k live
    """
    conn = SWOTLakeAreaConnector()
    spec = ReductionSpec(domain_name="lakes", station_ids=("6350900223",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "fraction"
