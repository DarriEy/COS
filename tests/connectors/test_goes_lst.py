"""GOES ABI LST connector — hermetic test of the gridded basin-reduction path.

GOES-R ABI L2 LST has NO SYMFLUENCE native, so this is *spec-validated*: the
assertions reproduce the published NOAA GOES-R ABI L2 LST product spec on a
synthetic inline fixture — the Kelvin canonical unit (identity scale), the
``DQF == 0`` good-quality gate (any non-zero DQF masked), the ``_FillValue``
sentinel, the physical valid Kelvin band, the half-open UTC window, and the
cos-lat basin reduction — with no network and no auth. Real-data shapes are
covered too: a (lat, lon, time)-ordered granule and a 2-D geolocated grid.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.goes_lst import (
    FILL_VALUE,
    GOOD_DQF,
    SOURCE_LST_SCALE,
    VALID_LST_RANGE,
    GOESLSTConnector,
)
from cos.core.models import KIND_UNITS, ObservationKind, ReductionSpec, SpatialReduction


def _spec(area_km2: float = 8000.0) -> ReductionSpec:
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


@pytest.fixture
def lst_arrays():
    """Synthetic ABI-like LST grid (Kelvin) with a companion DQF array.

    Uniform 300 K everywhere on the good layer; one fill-value cell and one
    non-zero-DQF (degraded) cell are present to be masked.
    """
    times = np.array(["2024-06-15T18:00", "2024-06-15T19:00"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    lst = np.full((2, 3, 3), 300.0)       # 300 K everywhere
    dqf = np.zeros((2, 3, 3))             # all good quality
    lst[0, 0, 0] = FILL_VALUE            # NetCDF fill -> masked
    dqf[0, 1, 1] = 1.0                   # degraded DQF -> masked (LST still 300 K)
    return lats, lons, times, lst, dqf


def test_canonical_unit_and_identity_scale_spec():
    """Spec contract: ABI LST is already Kelvin -> identity boundary scale."""
    assert SOURCE_LST_SCALE == 1.0
    assert GOOD_DQF == 0
    assert KIND_UNITS[ObservationKind.LST] == "K"


def test_reduce_arrays_basin_mean_identity_scale_and_masks(lst_arrays):
    """Spec: uniform 300 K passes through; fill and non-zero-DQF cells are masked out."""
    lats, lons, times, lst, dqf = lst_arrays
    conn = GOESLSTConnector()
    series = conn.reduce_arrays(
        lats, lons, times, lst, _spec(), dqf=dqf,
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.LST
    assert series.unit == "K"  # canonical LST unit
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Identity scale: remaining valid cells are all 300 K -> mean 300 K
    # (fill and DQF!=0 cells excluded; if they leaked the mean would not be clean).
    for p in series.points:
        assert p.value == pytest.approx(300.0, abs=1e-6)
        assert p.quality.value == "good"


def test_nonzero_dqf_masked_to_missing():
    """Spec: a timestep whose only valid cell is DQF!=0 -> None / MISSING."""
    conn = GOESLSTConnector()
    times = np.array(["2024-06-15T18:00"], dtype="datetime64[ns]")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    lst = np.array([[[300.0]]])
    dqf = np.array([[[2.0]]])  # invalid/degraded -> masked even though LST is plausible
    spec = ReductionSpec(
        domain_name="cell", centroid=(51.0, -115.0),
        reduction=SpatialReduction.NEAREST_CELL,
    )
    series = conn.reduce_arrays(
        lats, lons, times, lst, spec, dqf=dqf,
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_fill_only_layer_reduces_to_missing():
    """Spec: a timestep whose in-bbox cells are all fill -> None / MISSING."""
    conn = GOESLSTConnector()
    times = np.array(["2024-06-15T18:00"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    lst = np.full((1, 3, 3), FILL_VALUE)  # entire layer is fill
    series = conn.reduce_arrays(
        lats, lons, times, lst, _spec(),
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_out_of_range_masked_to_missing():
    """Spec: a finite LST outside VALID_LST_RANGE is implausible -> MISSING."""
    lo, hi = VALID_LST_RANGE
    conn = GOESLSTConnector()
    times = np.array(["2024-06-15T18:00"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    lst = np.full((1, 3, 3), hi + 100.0)  # absurdly hot -> all out of band
    series = conn.reduce_arrays(
        lats, lons, times, lst, _spec(),
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"
    assert 0 < lo < hi  # sanity: Kelvin band is strictly positive


def test_no_dqf_array_falls_back_to_fill_and_range_mask():
    """If a granule carries no DQF, fill/range masking alone still applies."""
    conn = GOESLSTConnector()
    times = np.array(["2024-06-15T18:00"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    lst = np.full((1, 3, 3), 295.0)
    lst[0, 0, 0] = FILL_VALUE
    series = conn.reduce_arrays(
        lats, lons, times, lst, _spec(),  # dqf=None
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value == pytest.approx(295.0, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_window_trim_half_open(lst_arrays):
    """Half-open [18:30, 19:00): excludes the 18:00 and 19:00 stamps."""
    lats, lons, times, lst, dqf = lst_arrays
    conn = GOESLSTConnector()
    series = conn.reduce_arrays(
        lats, lons, times, lst, _spec(), dqf=dqf,
        start=datetime(2024, 6, 15, 18, 30, tzinfo=UTC),
        end=datetime(2024, 6, 15, 19, 0, tzinfo=UTC),
    )
    # 18:00 is before the window, 19:00 is the exclusive upper bound -> empty.
    assert series.points == []

    # Half-open [18:00, 19:00) keeps only the 18:00 sub-hourly stamp.
    series2 = conn.reduce_arrays(
        lats, lons, times, lst, _spec(), dqf=dqf,
        start=datetime(2024, 6, 15, 18, 0, tzinfo=UTC),
        end=datetime(2024, 6, 15, 19, 0, tzinfo=UTC),
    )
    hours = {p.timestamp.hour for p in series2.points}
    assert hours == {18}


def test_small_basin_defaults_to_nearest_cell(lst_arrays):
    lats, lons, times, lst, dqf = lst_arrays
    conn = GOESLSTConnector()
    series = conn.reduce_arrays(
        lats, lons, times, lst, _spec(area_km2=500.0), dqf=dqf,
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("goes_lst:cell:")


def test_reduce_file_roundtrip_netcdf(tmp_path, lst_arrays):
    """End-to-end through the NetCDF reader: DQF mask + reduce on a real file."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats, lons, times, lst, dqf = lst_arrays
    ds = xr.Dataset(
        {
            "LST": (("time", "lat", "lon"), lst),
            "DQF": (("time", "lat", "lon"), dqf),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "goes_lst_synth.nc"
    ds.to_netcdf(path)

    conn = GOESLSTConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "K"
    assert all(p.value == pytest.approx(300.0, abs=1e-6) for p in series.points)
    assert series.source_info["variable"] == "LST"


def test_reduce_file_lat_lon_time_ordering(tmp_path):
    """Regression: a (lat, lon, time)-ordered granule must reduce without error.

    Against a reader that passed da.values straight through as if it were
    (time, lat, lon), basin_mean raises IndexError. With the transpose it yields
    one point per time step.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2024-06-15T18:00", "2024-06-15T19:00"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    lst = np.full((3, 3, 2), 300.0)   # (lat, lon, time) ordering
    dqf = np.zeros((3, 3, 2))
    lst[0, 0, 0] = FILL_VALUE
    ds = xr.Dataset(
        {
            "LST": (("lat", "lon", "time"), lst),
            "DQF": (("lat", "lon", "time"), dqf),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "goes_lst_lat_lon_time.nc"
    ds.to_netcdf(path)

    conn = GOESLSTConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "K"
    assert len(series.points) == 2  # one point per time step, not per lat
    for p in series.points:
        assert p.value == pytest.approx(300.0, abs=1e-6)
        assert p.quality.value == "good"


def test_two_dimensional_geostationary_coords_reduce():
    """Regression: ABI fixed-grid LST may carry 2-D lat/lon with off-disk fills.

    reduce_grid assumes 1-D coord vectors and raises IndexError on 2-D coords.
    The 2-D reduction path masks the bbox cells (and off-disk inf coords) and
    reduces over them instead.
    """
    times = np.array(["2024-06-15T18:00"], dtype="datetime64[ns]")
    lat2d, lon2d = np.meshgrid(
        np.array([49.5, 50.5, 51.5, 52.5]),
        np.array([-116.5, -115.5, -114.5, -113.5]),
        indexing="ij",
    )
    # Off-disk corner: non-finite coords (the real geostationary grid's edge).
    lat2d[0, 0] = np.inf
    lon2d[0, 0] = np.inf
    lst = np.full((1, 4, 4), 305.0)
    dqf = np.zeros((1, 4, 4))
    lst[0, 0, 0] = np.inf    # off-disk fill -> masked
    dqf[0, 3, 3] = 1.0       # degraded DQF -> masked

    conn = GOESLSTConnector()
    series = conn.reduce_arrays(
        lat2d, lon2d, times, lst, _spec(), dqf=dqf,
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "K"
    assert len(series.points) == 1
    assert series.points[0].value == pytest.approx(305.0, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_two_dimensional_coords_nearest_cell():
    """2-D coord nearest_cell picks the nearest valid in-grid cell to the centroid."""
    times = np.array(["2024-06-15T18:00"], dtype="datetime64[ns]")
    lat2d, lon2d = np.meshgrid(
        np.array([50.0, 51.0, 52.0]),
        np.array([-116.0, -115.0, -114.0]),
        indexing="ij",
    )
    lst = np.full((1, 3, 3), 280.0)
    lst[0, 1, 1] = 310.0  # the centroid (51, -115) cell
    conn = GOESLSTConnector()
    series = conn.reduce_arrays(
        lat2d, lon2d, times, lst, _spec(area_km2=500.0),
        start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.points[0].value == pytest.approx(310.0, abs=1e-6)


def test_native_abi_fixed_grid_geolocation(tmp_path):
    """Regression: real NODD ABI L2 granules ship NO lat/lon — only scan angles
    x/y (radians) and a goes_imager_projection. The connector must geolocate them
    (NOAA fixed-grid transform) instead of KeyError'ing on a missing 'lat'.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    # Fixed-grid stand-in near the GOES-16 sub-point (lon0 = -75). Uniform 300 K,
    # so basin_mean over whatever cells geolocate inside the bbox is exactly 300.
    x = np.array([-0.01, 0.0, 0.01])     # E-W scan angle, radians
    y = np.array([0.06, 0.05, 0.04])     # N-S elevation angle, radians
    lst = np.full((3, 3), 300.0, dtype="float32")
    dqf = np.zeros((3, 3), dtype="float32")
    proj = xr.DataArray(
        0,
        attrs=dict(
            grid_mapping_name="geostationary",
            perspective_point_height=35786023.0,
            semi_major_axis=6378137.0,
            semi_minor_axis=6356752.31414,
            longitude_of_projection_origin=-75.0,
            sweep_angle_axis="x",
        ),
    )
    ds = xr.Dataset(
        {"LST": (("y", "x"), lst), "DQF": (("y", "x"), dqf), "goes_imager_projection": proj},
        coords={"y": y, "x": x, "t": np.datetime64("2024-06-28T18:02:36")},
    )
    path = tmp_path / "OR_ABI-L2-LSTC-M6_G16.nc"
    ds.to_netcdf(path)

    spec = ReductionSpec(
        domain_name="conus", bbox=(-10.0, -85.0, 40.0, -65.0),
        centroid=(8.0, -75.0), area_km2=8000.0,
    )
    series = GOESLSTConnector().reduce_file(
        path, spec, datetime(2024, 6, 28, tzinfo=UTC), datetime(2024, 6, 29, tzinfo=UTC),
    )
    assert series.unit == "K"
    assert series.points and series.points[0].value == pytest.approx(300.0, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_anonymous_no_auth_spec():
    """Spec contract: AWS NODD is open data -> connector requires no auth provider."""
    conn = GOESLSTConnector()
    assert conn.auth == frozenset()
    assert conn.structural_class == "gridded"


@pytest.mark.asyncio
async def test_list_sites_returns_reduced_region():
    conn = GOESLSTConnector()
    sites = await conn.list_sites(_spec())
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "goes_lst:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = GOESLSTConnector()
    with pytest.raises(Exception, match="cached NetCDF"):
        await conn.fetch_series(
            _spec(), datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_goes_lst():
    """LIVE smoke against a real ABI L2 LST granule (requires a cached path).

    AWS NODD GOES is anonymous open data; supply a downloaded ABI-L2-LSTC/LSTF
    NetCDF. Run with: pytest -m network tests/connectors/test_goes_lst.py -k live
    """
    import os

    path = os.environ.get("GOES_LST_NC_PATH")
    if not path:
        pytest.skip("set GOES_LST_NC_PATH to a cached ABI-L2-LSTC/LSTF NetCDF")
    conn = GOESLSTConnector({"nc_path": path})
    async with conn:
        series_list = await conn.fetch_series(
            _spec(), datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC),
        )
    assert series_list and series_list[0].unit == "K"
