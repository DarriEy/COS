"""ECOSTRESS LST connector — hermetic test of the gridded basin-reduction path.

ECOSTRESS LST has NO SYMFLUENCE native, so this is *spec-validated*: the
assertions reproduce the published LP DAAC ECOSTRESS L2 LST product spec
(ECO2LSTE.001) on a synthetic inline fixture — the Kelvin unit, the DN*0.02
source scale, the DN 0 fill sentinel, the physical valid band, the half-open UTC
window, and the cos-lat basin reduction — with no network and no auth.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.ecostress_lst import (
    LST_FILL_VALUE,
    SOURCE_LST_SCALE,
    VALID_LST_RANGE,
    ECOSTRESSLSTConnector,
    _geo_grid_from_struct_metadata,
)
from cos.core.models import KIND_UNITS, ObservationKind, ReductionSpec, SpatialReduction

# A DN that scales to a plausible ~298 K surface temperature: 298 / 0.02 = 14900.
WARM_DN = 14900.0
WARM_K = WARM_DN * SOURCE_LST_SCALE  # 298.0 K


@pytest.fixture
def lst_nc(tmp_path):
    """A synthetic ECO2LSTE-like LST file (stored DN, scale 0.02) with a fill cell."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2021-06-15", "2021-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.full((2, 3, 3), WARM_DN)   # uniform ~298 K once scaled
    data[0, 0, 0] = LST_FILL_VALUE       # one fill cell -> masked to NaN before mean
    ds = xr.Dataset(
        {"LST": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ecostress_lst_synth.nc"
    ds.to_netcdf(path)
    return path


def test_canonical_unit_and_scale_spec():
    """Spec contract: ECO2LSTE stores DN; canonical Kelvin = DN * 0.02."""
    assert SOURCE_LST_SCALE == 0.02
    assert LST_FILL_VALUE == 0.0
    assert KIND_UNITS[ObservationKind.LST] == "K"


def test_reduce_arrays_basin_mean_scale_and_fill_mask():
    """Spec: uniform DN scales to Kelvin; the DN-0 fill cell is masked out of the mean."""
    conn = ECOSTRESSLSTConnector()
    times = np.array(["2021-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), WARM_DN)
    values[0, 1, 1] = LST_FILL_VALUE  # masked, must not pull the mean down
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_arrays(
        lats, lons, times, values, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.LST
    assert series.unit == "K"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Source scale: remaining valid cells all scale to 298 K (fill excluded).
    assert series.points[0].value == pytest.approx(WARM_K, abs=1e-6)
    assert series.points[0].quality.value == "good"


@pytest.fixture
def lst_nc_lat_lon_time(tmp_path):
    """Real-data shape: a granule may be dim-ordered (lat, lon, time).

    The synthetic fixture above is (time, lat, lon); reduce_file must transpose
    to (time, lat, lon) before reducing — without the transpose, basin_mean
    indexes the wrong axes.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2021-06-15", "2021-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.full((3, 3, 2), WARM_DN)  # (lat, lon, time) ordering
    data[0, 0, 0] = LST_FILL_VALUE
    ds = xr.Dataset(
        {"LST": (("lat", "lon", "time"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ecostress_lst_lat_lon_time.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_lat_lon_time_ordering(lst_nc_lat_lon_time):
    """Regression: a (lat, lon, time)-ordered product must reduce without error.

    Passing da.values straight through as if it were (time, lat, lon) makes
    basin_mean IndexError on the real grid; the transpose yields one scaled point
    per time step.
    """
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_file(
        lst_nc_lat_lon_time, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "K"
    assert len(series.points) == 2  # one point per time step, not per lat
    for p in series.points:
        assert p.value == pytest.approx(WARM_K, abs=1e-6)
        assert p.quality.value == "good"


@pytest.fixture
def lst_nc_2d_coords(tmp_path):
    """Real-data shape: high-res swath with 2-D geolocation lat/lon arrays."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2021-06-15"], dtype="datetime64[ns]")
    lat2d = np.array([[50.0, 50.0, 50.0], [51.0, 51.0, 51.0], [52.0, 52.0, 52.0]])
    lon2d = np.array([[-116.0, -115.0, -114.0]] * 3)
    data = np.full((1, 3, 3), WARM_DN)
    ds = xr.Dataset(
        {
            "LST": (("time", "y", "x"), data),
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        },
        coords={"time": times},
    )
    path = tmp_path / "ecostress_lst_2d.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_2d_geolocation(lst_nc_2d_coords):
    """Regression: 2-D geolocation lat/lon take the dedicated bbox-mask path."""
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_file(
        lst_nc_2d_coords, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "K"
    assert len(series.points) == 1
    assert series.points[0].value == pytest.approx(WARM_K, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_reduce_arrays_2d_nearest_cell():
    """Spec: small basin on a 2-D grid picks the nearest valid cell."""
    conn = ECOSTRESSLSTConnector()
    times = np.array(["2021-06-15"], dtype="datetime64[ns]")
    lat2d = np.array([[50.0, 50.0], [52.0, 52.0]])
    lon2d = np.array([[-116.0, -114.0], [-116.0, -114.0]])
    values = np.full((1, 2, 2), WARM_DN)
    values[0, 0, 0] = (290.0 / SOURCE_LST_SCALE)  # the SW cell scales to 290 K
    spec = ReductionSpec(
        domain_name="tiny", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(50.0, -116.0), area_km2=400.0,  # small -> nearest_cell at SW corner
    )
    series = conn.reduce_arrays(
        lat2d, lon2d, times, values, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.points[0].value == pytest.approx(290.0, abs=1e-6)


def test_fill_only_cell_reduces_to_missing():
    """Spec: a timestep whose in-bbox cells are all fill DN -> None / MISSING."""
    conn = ECOSTRESSLSTConnector()
    times = np.array(["2021-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), LST_FILL_VALUE)  # entire layer is fill DN 0
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_arrays(
        lats, lons, times, values, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_out_of_range_masked_to_missing():
    """Spec: a scaled LST outside VALID_LST_RANGE is treated as invalid -> MISSING."""
    lo, hi = VALID_LST_RANGE
    conn = ECOSTRESSLSTConnector()
    times = np.array(["2021-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), (hi + 100.0) / SOURCE_LST_SCALE)  # scales above band
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_arrays(
        lats, lons, times, values, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"
    assert lo > 0  # sanity: the valid band is strictly above absolute zero


def test_window_trim_half_open(lst_nc):
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    # Half-open [2021-06-01, 2021-07-15): includes 06-15, excludes 07-15.
    series = conn.reduce_file(
        lst_nc, spec,
        datetime(2021, 6, 1, tzinfo=UTC), datetime(2021, 7, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2021, 6) in months
    assert (2021, 7) not in months


def test_reduce_file_basin_mean(lst_nc):
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_file(
        lst_nc, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "K"
    for p in series.points:
        assert p.value == pytest.approx(WARM_K, abs=1e-6)
        assert p.quality.value == "good"
    assert series.source_info["variable"] == "LST"
    assert series.source_info["product"] == "ECO2LSTE.001"


def test_small_basin_defaults_to_nearest_cell(lst_nc):
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="tiny", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        lst_nc, spec,
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("ecostress_lst:cell:")


@pytest.mark.asyncio
async def test_list_sites_one_reduced_region():
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    sites = await conn.list_sites(spec)
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "ecostress_lst:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0), centroid=(51.0, -115.0),
    )
    with pytest.raises(Exception, match="ECO2LSTE"):
        await conn.fetch_series(
            spec, datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
        )


# --- gridded HDF-EOS5 (ECO_L2G_LSTE v002) path ----------------------------------

# StructMetadata corner bounds (HE5_GCTP_GEO microdegrees, (lon, lat)) for a small
# arid bbox: upper-left (-114.0, 36.0), lower-right (-112.0, 34.0). A 2x2 grid puts
# cell centres at lon {-113.5, -112.5}, lat {35.5, 34.5}.
_SM_UL = (-114.0e6, 36.0e6)
_SM_LR = (-112.0e6, 34.0e6)
_STRUCT_METADATA = (
    "GROUP=GridStructure\n"
    "\tGROUP=GRID_1\n"
    '\t\tGridName="ECO_L2G_LSTE_70m"\n'
    "\t\tXDim=2\n"
    "\t\tYDim=2\n"
    f"\t\tUpperLeftPointMtrs=({_SM_UL[0]:.6f},{_SM_UL[1]:.6f})\n"
    f"\t\tLowerRightMtrs=({_SM_LR[0]:.6f},{_SM_LR[1]:.6f})\n"
    "\t\tProjection=HE5_GCTP_GEO\n"
    "\tEND_GROUP=GRID_1\n"
    "END_GROUP=GridStructure\n"
)


def test_geo_grid_from_struct_metadata_centres():
    """Corner bounds + grid shape reconstruct descending-lat cell-centre vectors."""
    lats, lons = _geo_grid_from_struct_metadata(_STRUCT_METADATA, (2, 2))
    np.testing.assert_allclose(lons, [-113.5, -112.5])
    np.testing.assert_allclose(lats, [35.5, 34.5])  # lat descends from the upper-left


@pytest.fixture
def lst_hdfeos_grid(tmp_path):
    """A synthetic ECO_L2G_LSTE-like HDF-EOS5 file: nested GRID LST + StructMetadata.

    LST is already Kelvin (no DN scaling); geolocation must be rebuilt from the
    StructMetadata corner bounds, and the top-level dataset has no data variables.
    """
    h5py = pytest.importorskip("h5py")
    warm = 300.0  # plausible arid land-surface temperature, Kelvin
    lst = np.full((2, 2), warm, dtype="float32")
    lst[0, 0] = np.nan  # NaN fill cell -> masked out of the mean
    path = tmp_path / "ECO_L2G_LSTE_synth.h5"
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("HDFEOS/GRIDS/ECO_L2G_LSTE_70m/Data Fields")
        dset = grp.create_dataset("LST", data=lst)
        dset.attrs["scale_factor"] = 1.0
        dset.attrs["units"] = b"K"
        h5.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=_STRUCT_METADATA)
        md = "HDFEOS/ADDITIONAL/FILE_ATTRIBUTES/StandardMetadata/"
        h5.create_dataset(md + "RangeBeginningDate", data=b"2023-07-02")
        h5.create_dataset(md + "RangeBeginningTime", data=b"08:07:36.953583")
    return path, warm


def test_reduce_file_hdfeos_grid_geolocates_and_reduces(lst_hdfeos_grid):
    """Gridded HDF-EOS5: rebuilt geolocation + native-Kelvin scale reduce to 'K'.

    Reproduces the live ECO_L2G_LSTE path: top-level data_vars are empty, so the
    connector opens the nested GRID group, reads native-float Kelvin LST, and
    rebuilds lat/lon from StructMetadata — no DN*0.02 scaling.
    """
    path, warm = lst_hdfeos_grid
    conn = ECOSTRESSLSTConnector()
    spec = ReductionSpec(
        domain_name="az_sw", bbox=(34.0, -114.0, 36.0, -112.0),
        centroid=(35.0, -113.0), area_km2=8000.0,
    )
    series = conn.reduce_file(
        path, spec,
        datetime(2023, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "K"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert len(series.points) == 1  # one acquisition instant
    p = series.points[0]
    assert p.value == pytest.approx(warm, abs=1e-4)  # native Kelvin, NaN cell excluded
    assert p.quality.value == "good"
    assert p.timestamp.year == 2023 and p.timestamp.month == 7
    # The gridded path reports its own provenance, not the flat ECO2LSTE.001's.
    assert series.source_info["product"] == "ECO_L2G_LSTE.002"
    assert series.source_info["scale_k_per_count"] == "1"  # native Kelvin, no DN scaling
    assert "LST" in series.source_info["variable"]
    assert series.source_info["scale_k_per_count"] == "1"


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_ecostress_lst():
    """LIVE smoke: requires a real Earthdata/LP DAAC ECO2LSTE.001 LST granule.

    Run with: pytest -m network tests/connectors/test_ecostress_lst.py -k live
    """
    import os

    nc_path = os.environ.get("ECOSTRESS_LST_NC")
    if not nc_path:
        pytest.skip("set ECOSTRESS_LST_NC to a real ECO2LSTE.001 LST granule")
    conn = ECOSTRESSLSTConnector({"nc_path": nc_path})
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC),
        )
    assert series_list and series_list[0].unit == "K"
