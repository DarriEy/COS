"""ECOSTRESS L3 ET PT-JPL connector — hermetic test of the gridded reduction path.

ECOSTRESS has NO SYMFLUENCE native, so this is *spec-validated*: the assertions
reproduce the published LP DAAC ECO3ETPTJPL.001 product spec on a synthetic
inline fixture — the canonical mm/day unit, the W/m² latent-heat → mm/day
boundary conversion, the -9999 fill sentinel, the physical valid band, the
half-open UTC window, and the cos-lat basin reduction (including a
(lat, lon, time)-ordered grid and a 2-D-coordinate swath grid) — with no network
and no auth.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.ecostress_et import (
    ET_FILL_VALUE,
    LATENT_HEAT_VAPORIZATION,
    SOURCE_MM_PER_DAY,
    VALID_ET_RANGE_MM_DAY,
    WATER_DENSITY,
    WM2_TO_MM_PER_DAY,
    ECOSTRESSETConnector,
)
from cos.core.models import KIND_UNITS, ObservationKind, ReductionSpec, SpatialReduction

# A large basin -> basin_mean; a small one -> nearest_cell.
_BBOX = (50.0, -116.0, 52.0, -114.0)
_CENTROID = (51.0, -115.0)
_LATS = np.array([50.0, 51.0, 52.0])
_LONS = np.array([-116.0, -115.0, -114.0])


def _big_spec(domain="bow"):
    return ReductionSpec(
        domain_name=domain, bbox=_BBOX, centroid=_CENTROID, area_km2=8000.0,
    )


def _small_spec(domain="tiny"):
    return ReductionSpec(
        domain_name=domain, bbox=_BBOX, centroid=_CENTROID, area_km2=500.0,
    )


def test_canonical_unit_and_conversion_constants_spec():
    """Spec contract: canonical unit is mm/day; the W/m² factor matches λ·ρ_w."""
    assert KIND_UNITS[ObservationKind.ET] == "mm/day"
    assert SOURCE_MM_PER_DAY == 1.0
    # mm/day per W/m² = 86400 / (λ · ρ_w) · 1000.
    expected = 86400.0 / (LATENT_HEAT_VAPORIZATION * WATER_DENSITY) * 1000.0
    assert pytest.approx(expected) == WM2_TO_MM_PER_DAY
    assert pytest.approx(0.03526, abs=1e-4) == WM2_TO_MM_PER_DAY


def test_reduce_arrays_mm_day_identity_and_fill_mask():
    """Spec: a source already in mm/day passes through; the -9999 fill is masked out."""
    conn = ECOSTRESSETConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    values = np.full((1, 3, 3), 4.0)          # uniform 4.0 mm/day
    values[0, 1, 1] = ET_FILL_VALUE           # one fill cell -> must not pull the mean
    series = conn.reduce_arrays(
        _LATS, _LONS, times, values, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        var_name="ETdaily", source_units="mm/day",
    )
    assert series.kind == ObservationKind.ET
    assert series.unit == "mm/day"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.points[0].value == pytest.approx(4.0, abs=1e-6)
    assert series.points[0].quality.value == "good"
    assert series.source_info["variable"] == "ETdaily"
    assert series.source_info["source_units"] == "mm/day"


def test_reduce_arrays_wm2_latent_heat_conversion():
    """Spec: an instantaneous W/m² latent-heat source converts to mm/day at boundary.

    LE = 100 W/m² -> 100 · WM2_TO_MM_PER_DAY ≈ 3.526 mm/day (uniform field).
    """
    conn = ECOSTRESSETConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    le = np.full((1, 3, 3), 100.0)  # 100 W/m² latent heat flux
    series = conn.reduce_arrays(
        _LATS, _LONS, times, le, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        var_name="ETinst", source_units="W/m2",
    )
    assert series.unit == "mm/day"
    assert series.source_info["source_units"] == "W/m2"
    assert series.points[0].value == pytest.approx(100.0 * WM2_TO_MM_PER_DAY, abs=1e-6)
    assert series.points[0].quality.value == "good"
    # ETinst instantaneous flux scaled to mm/day -> provenance flag stamped.
    assert series.source_info["instantaneous_scaled"] == "true"


def test_wm2_unit_aliases_recognized():
    """Spec: case/space-insensitive W/m² spellings all trigger the conversion."""
    conn = ECOSTRESSETConnector()
    for unit in ("W/m2", "w m-2", "Wm-2", "W/m^2"):
        factor, resolved = conn._conversion_factor(unit)
        assert factor == pytest.approx(WM2_TO_MM_PER_DAY), unit
        assert resolved == "W/m2"
    # An mm/day-style or unlabeled source stays identity.
    assert conn._conversion_factor("mm/day")[0] == SOURCE_MM_PER_DAY
    assert conn._conversion_factor("")[0] == SOURCE_MM_PER_DAY


def test_fill_only_cell_reduces_to_missing():
    """Spec: a timestep whose in-bbox cells are all fill -> None / MISSING."""
    conn = ECOSTRESSETConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    values = np.full((1, 3, 3), ET_FILL_VALUE)  # entire layer is fill
    series = conn.reduce_arrays(
        _LATS, _LONS, times, values, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_out_of_range_masked_to_missing():
    """Spec: a finite ET above VALID_ET_RANGE_MM_DAY is invalid -> MISSING."""
    lo, hi = VALID_ET_RANGE_MM_DAY
    conn = ECOSTRESSETConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    values = np.full((1, 3, 3), hi + 100.0)  # absurdly high -> out of band
    series = conn.reduce_arrays(
        _LATS, _LONS, times, values, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        source_units="mm/day",
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"
    assert lo == 0.0  # negative ET is non-physical and also masked


def test_negative_et_masked_to_missing():
    """Spec: a negative (non-physical) ET is masked out -> MISSING."""
    conn = ECOSTRESSETConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    values = np.full((1, 3, 3), -2.0)
    series = conn.reduce_arrays(
        _LATS, _LONS, times, values, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        source_units="mm/day",
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_window_trim_half_open():
    """Half-open [2020-06-01, 2020-07-15): includes 06-15, excludes 07-15."""
    conn = ECOSTRESSETConnector()
    times = np.array(["2020-06-15", "2020-07-15"], dtype="datetime64[ns]")
    values = np.full((2, 3, 3), 3.0)
    series = conn.reduce_arrays(
        _LATS, _LONS, times, values, _big_spec(),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 15, tzinfo=UTC),
        source_units="mm/day",
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in months
    assert (2020, 7) not in months


def test_small_basin_defaults_to_nearest_cell():
    conn = ECOSTRESSETConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    values = np.full((1, 3, 3), 4.0)
    series = conn.reduce_arrays(
        _LATS, _LONS, times, values, _small_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        source_units="mm/day",
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("ecostress_et:cell:")


def test_reduce_arrays_2d_coordinate_swath():
    """Real-data shape: a tiled swath can carry 2-D lat/lon. reduce_arrays must
    take the dedicated 2-D path (reduce_grid would IndexError on 2-D coords)."""
    conn = ECOSTRESSETConnector()
    lat2d, lon2d = np.meshgrid(_LATS, _LONS, indexing="ij")  # (3, 3)
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    values = np.full((1, 3, 3), 5.0)
    values[0, 0, 0] = ET_FILL_VALUE  # masked, must not bias the 2-D mean
    series = conn.reduce_arrays(
        lat2d, lon2d, times, values, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        source_units="mm/day",
    )
    assert series.unit == "mm/day"
    assert series.points[0].value == pytest.approx(5.0, abs=1e-6)
    assert series.points[0].quality.value == "good"


@pytest.fixture
def ecostress_nc_lat_lon_time(tmp_path):
    """Real-data shape: a tiled ECOSTRESS grid served (lat, lon, time).

    reduce_file must transpose to (time, lat, lon) before reducing — without it,
    basin_mean indexes the wrong axes.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15", "2020-07-15"], dtype="datetime64[ns]")
    data = np.full((3, 3, 2), 4.0)  # (lat, lon, time), uniform 4.0 mm/day
    data[0, 0, 0] = ET_FILL_VALUE
    ds = xr.Dataset(
        {"ETdaily": (("lat", "lon", "time"), data, {"units": "mm/day"})},
        coords={"time": times, "lat": _LATS, "lon": _LONS},
    )
    path = tmp_path / "ecostress_lat_lon_time.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_lat_lon_time_ordering(ecostress_nc_lat_lon_time):
    """A (lat, lon, time)-ordered product must reduce to one point per time step."""
    conn = ECOSTRESSETConnector()
    series = conn.reduce_file(
        ecostress_nc_lat_lon_time, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mm/day"
    assert len(series.points) == 2  # one per time step, not per lat
    for p in series.points:
        assert p.value == pytest.approx(4.0, abs=1e-6)
        assert p.quality.value == "good"
    assert series.source_info["variable"] == "ETdaily"


@pytest.fixture
def ecostress_nc_2d_coords_yx(tmp_path):
    """Real-data shape: a tiled product with 2-D lat/lon *coordinates* riding on
    (y, x) dims and a (time, y, x) ET var — exactly what GDAL/rioxarray emit for
    the ECOSTRESS L3 Tiled (UTM) GeoTIFFs. reduce_file must reorder by the y/x
    dims (not push time to the trailing axis) or basin_mean broadcast-fails.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lat2d, lon2d = np.meshgrid(_LATS, _LONS, indexing="ij")  # (3, 3) on (y, x)
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    data = np.full((1, 3, 3), 4.0)  # (time, y, x), mm/day
    ds = xr.Dataset(
        {"ETdaily": (("time", "y", "x"), data, {"units": "mm/day"})},
        coords={"time": times, "lat": (("y", "x"), lat2d), "lon": (("y", "x"), lon2d)},
    )
    path = tmp_path / "ecostress_2d_yx.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_2d_coords_on_yx_dims(ecostress_nc_2d_coords_yx):
    """Regression: 2-D lat/lon coords on (y, x) dims must reduce to one point per
    time step. Previously _to_time_lat_lon reordered (time, y, x) -> (y, x, time)
    because it looked for 'lat'/'lon' among the dims and found only y/x."""
    conn = ECOSTRESSETConnector()
    series = conn.reduce_file(
        ecostress_nc_2d_coords_yx, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mm/day"
    assert len(series.points) == 1
    assert series.points[0].value == pytest.approx(4.0, abs=1e-6)
    assert series.source_info["variable"] == "ETdaily"


@pytest.fixture
def ecostress_nc_wm2(tmp_path):
    """A synthetic ECO3ETPTJPL-like NetCDF carrying instantaneous W/m² latent heat."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    data = np.full((1, 3, 3), 100.0)  # 100 W/m²
    ds = xr.Dataset(
        {"ETinst": (("time", "lat", "lon"), data, {"units": "W/m2"})},
        coords={"time": times, "lat": _LATS, "lon": _LONS},
    )
    path = tmp_path / "ecostress_wm2.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_wm2_units_from_metadata(ecostress_nc_wm2):
    """reduce_file reads the W/m² units attribute and converts at the boundary."""
    conn = ECOSTRESSETConnector()
    series = conn.reduce_file(
        ecostress_nc_wm2, _big_spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mm/day"
    assert series.source_info["source_units"] == "W/m2"
    assert series.points[0].value == pytest.approx(100.0 * WM2_TO_MM_PER_DAY, abs=1e-6)


@pytest.mark.asyncio
async def test_list_sites_one_reduced_region():
    conn = ECOSTRESSETConnector()
    sites = await conn.list_sites(_big_spec())
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "ecostress_et:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = ECOSTRESSETConnector()
    spec = ReductionSpec(domain_name="x", bbox=_BBOX, centroid=_CENTROID)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_ecostress_et():
    """LIVE smoke: requires a real LP DAAC ECO3ETPTJPL.001 NetCDF.

    Run with: pytest -m network tests/connectors/test_ecostress_et.py -k live
    """
    import os

    nc_path = os.environ.get("ECOSTRESS_ET_NC")
    if not nc_path:
        pytest.skip("set ECOSTRESS_ET_NC to a real ECO3ETPTJPL.001 NetCDF")
    conn = ECOSTRESSETConnector({"nc_path": nc_path})
    async with conn:
        series_list = await conn.fetch_series(
            _big_spec(), datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )
    assert series_list and series_list[0].unit == "mm/day"
