"""OCO-3 SIF connector — hermetic test of the gridded basin-reduction path.

OCO-3 SIF has NO SYMFLUENCE native, so this is *spec-validated*: the assertions
reproduce the published OCO3_L2_Lite_SIF product spec on a synthetic inline
fixture — the W/m²/sr/µm → mW/m²/nm/sr boundary conversion (numeric identity), the
757/771 nm → 740 nm linear combination, the -999999 fill sentinel, the physical
valid band, the half-open UTC window, and the cos-lat basin reduction — with no
network and no auth. The 2-D per-sounding coordinate path (the real Lite layout)
and the (lat, lon, time) dim-order safety are covered explicitly.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.oco3_sif import (
    SIF_771_WEIGHT,
    SIF_COMBINE_SCALE,
    SIF_FILL_VALUE,
    SOURCE_SIF_SCALE,
    VALID_SIF_RANGE,
    OCO3SIFConnector,
)
from cos.core.models import KIND_UNITS, ObservationKind, ReductionSpec, SpatialReduction

BOW_SPEC = dict(
    domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
    centroid=(51.0, -115.0), area_km2=8000.0,  # large -> basin_mean
)
YEAR_2020 = (datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))


def test_canonical_unit_and_identity_scale_spec():
    """Spec: W/m²/sr/µm -> mW/m²/nm/sr is numerically the identity (×1000 ÷1000)."""
    assert pytest.approx(1.0) == SOURCE_SIF_SCALE
    assert KIND_UNITS[ObservationKind.SIF] == "mW/m2/nm/sr"


def test_reduce_arrays_basin_mean_identity_scale_and_fill_mask():
    """Spec: uniform SIF passes through unchanged; the -999999 fill cell is masked out."""
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), 1.5)  # already-combined 740 nm-like, source unit
    values[0, 1, 1] = SIF_FILL_VALUE  # masked, must not pull the mean down
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_arrays(lats, lons, times, values, spec, *YEAR_2020)
    assert series.kind == ObservationKind.SIF
    assert series.unit == "mW/m2/nm/sr"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Identity scale: remaining valid cells are all 1.5 -> mean 1.5 (fill excluded).
    assert series.points[0].value == pytest.approx(1.5, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_757_771_combination_to_740_proxy():
    """Spec: SIF_740 = 0.5*(SIF_757 + 1.5*SIF_771); identity scale preserves magnitude."""
    sif_757 = np.full((1, 2, 2), 1.0)
    sif_771 = np.full((1, 2, 2), 2.0)
    expected = SIF_COMBINE_SCALE * (1.0 + SIF_771_WEIGHT * 2.0)  # 0.5*(1+3)=2.0
    combined = OCO3SIFConnector._combine_757_771(sif_757, sif_771)
    assert combined == pytest.approx(expected)
    # A fill on either window propagates the fill sentinel (later masked to MISSING).
    sif_757_fill = sif_757.copy()
    sif_757_fill[0, 0, 0] = SIF_FILL_VALUE
    out = OCO3SIFConnector._combine_757_771(sif_757_fill, sif_771)
    assert out[0, 0, 0] == SIF_FILL_VALUE


def test_combined_windows_full_path_reduces_to_proxy_magnitude():
    """End-to-end on uncombined inputs: reduce_arrays receives the 740 proxy field."""
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    sif_757 = np.full((1, 3, 3), 1.0)
    sif_771 = np.full((1, 3, 3), 2.0)
    combined = OCO3SIFConnector._combine_757_771(sif_757, sif_771)
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_arrays(lats, lons, times, combined, spec, *YEAR_2020)
    assert series.points[0].value == pytest.approx(2.0, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_fill_only_cell_reduces_to_missing():
    """Spec: a timestep whose in-bbox cells are all fill -> None / MISSING."""
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), SIF_FILL_VALUE)  # entire layer is fill
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_arrays(lats, lons, times, values, spec, *YEAR_2020)
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_out_of_range_masked_to_missing():
    """Spec: a finite SIF outside VALID_SIF_RANGE is treated as invalid -> MISSING."""
    lo, hi = VALID_SIF_RANGE
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), hi + 100.0)  # absurdly high -> all out of band
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_arrays(lats, lons, times, values, spec, *YEAR_2020)
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"
    assert lo < 0 < hi  # sanity: a mild negative SIF tail is allowed by the band


def test_window_trim_half_open():
    """Half-open [start, end): the end-boundary timestep is excluded."""
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15", "2020-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((2, 3, 3), 1.5)
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_arrays(
        lats, lons, times, values, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in months
    assert (2020, 7) not in months


def test_small_basin_defaults_to_nearest_cell():
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), 1.5)
    spec = ReductionSpec(
        domain_name="tiny", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_arrays(lats, lons, times, values, spec, *YEAR_2020)
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("oco3_sif:cell:")


def test_2d_coordinate_per_sounding_basin_mean():
    """Real OCO-3 Lite layout: 2-D per-sounding lat/lon; bbox cell-mask reduction.

    With 1-D-coord reduce_grid this footprint grid would IndexError; the 2-D path
    masks the in-bbox soundings and cos-lat-weights them.
    """
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lat2d = np.array([[50.0, 50.0, 50.0], [51.0, 51.0, 51.0], [52.0, 52.0, 52.0]])
    lon2d = np.array([[-116.0, -115.0, -114.0]] * 3)
    values = np.full((1, 3, 3), 1.5)
    # One sounding off-grid (non-finite coord) plus one fill value — both drop out.
    lat2d = lat2d.copy()
    lat2d[0, 0] = np.nan
    values[0, 2, 2] = SIF_FILL_VALUE
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_arrays(lat2d, lon2d, times, values, spec, *YEAR_2020)
    assert series.unit == "mW/m2/nm/sr"
    assert series.points[0].value == pytest.approx(1.5, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_2d_coordinate_nearest_sounding():
    conn = OCO3SIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lat2d = np.array([[50.0, 50.0, 50.0], [51.0, 51.0, 51.0], [52.0, 52.0, 52.0]])
    lon2d = np.array([[-116.0, -115.0, -114.0]] * 3)
    values = np.zeros((1, 3, 3))
    values[0, 1, 1] = 1.5  # only the centroid sounding carries signal
    spec = ReductionSpec(
        domain_name="tiny", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_arrays(lat2d, lon2d, times, values, spec, *YEAR_2020)
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.points[0].value == pytest.approx(1.5, abs=1e-6)


@pytest.fixture
def oco3_nc(tmp_path):
    """A synthetic OCO3_L2_Lite_SIF NetCDF (757/771 nm windows) with a fill cell."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15", "2020-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    sif757 = np.full((2, 3, 3), 1.0)
    sif771 = np.full((2, 3, 3), 2.0)
    sif757[0, 0, 0] = SIF_FILL_VALUE  # one fill sounding -> masked before mean
    ds = xr.Dataset(
        {
            "SIF_757nm": (("time", "lat", "lon"), sif757),
            "SIF_771nm": (("time", "lat", "lon"), sif771),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "oco3_sif_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_combines_and_canonicalizes(oco3_nc):
    conn = OCO3SIFConnector()
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_file(oco3_nc, spec, *YEAR_2020)
    assert series.unit == "mW/m2/nm/sr"
    assert len(series.points) == 2  # one per time step
    for p in series.points:
        # 0.5*(1 + 1.5*2) = 2.0, identity scale; fill sounding excluded from mean.
        assert p.value == pytest.approx(2.0, abs=1e-6)
        assert p.quality.value == "good"
    assert "SIF_757nm" in series.source_info["variable"]
    assert series.source_info["product"] == "OCO3_L2_Lite_SIF"


@pytest.fixture
def oco3_nc_lat_lon_time(tmp_path):
    """Real-data shape: a (lat, lon, time)-ordered OCO-3 SIF NetCDF.

    reduce_file must transpose to (time, lat, lon) before reducing — without the
    transpose, basin_mean indexes the wrong axes.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15", "2020-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    sif757 = np.full((3, 3, 2), 1.0)
    sif771 = np.full((3, 3, 2), 2.0)
    sif757[0, 0, 0] = SIF_FILL_VALUE
    ds = xr.Dataset(
        {
            "SIF_757nm": (("lat", "lon", "time"), sif757),
            "SIF_771nm": (("lat", "lon", "time"), sif771),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "oco3_sif_lat_lon_time.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_lat_lon_time_ordering(oco3_nc_lat_lon_time):
    """Regression: a (lat, lon, time)-ordered product must reduce without error."""
    conn = OCO3SIFConnector()
    spec = ReductionSpec(**BOW_SPEC)
    series = conn.reduce_file(oco3_nc_lat_lon_time, spec, *YEAR_2020)
    assert series.unit == "mW/m2/nm/sr"
    assert len(series.points) == 2  # one point per time step, not per lat
    for p in series.points:
        assert p.value == pytest.approx(2.0, abs=1e-6)
        assert p.quality.value == "good"


@pytest.mark.asyncio
async def test_list_sites_one_reduced_region():
    conn = OCO3SIFConnector()
    spec = ReductionSpec(**BOW_SPEC)
    sites = await conn.list_sites(spec)
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "oco3_sif:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = OCO3SIFConnector()
    spec = ReductionSpec(
        domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0), centroid=(51.0, -115.0),
    )
    with pytest.raises(Exception, match="OCO-3"):
        await conn.fetch_series(spec, *YEAR_2020)


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_oco3_sif():
    """LIVE smoke: requires a real Earthdata-downloaded OCO3_L2_Lite_SIF NetCDF.

    Run with: pytest -m network tests/connectors/test_oco3_sif.py -k live
    """
    import os

    nc_path = os.environ.get("OCO3_SIF_NC")
    if not nc_path:
        pytest.skip("set OCO3_SIF_NC to a real OCO3_L2_Lite_SIF NetCDF")
    conn = OCO3SIFConnector({"nc_path": nc_path})
    spec = ReductionSpec(**BOW_SPEC)
    async with conn:
        series_list = await conn.fetch_series(spec, *YEAR_2020)
    assert series_list and series_list[0].unit == "mW/m2/nm/sr"
