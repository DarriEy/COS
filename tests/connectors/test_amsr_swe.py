"""AMSR2 SWE connector — hermetic test of the gridded basin-reduction path.

AMSR2 daily SWE (NSIDC AU_DySno) has NO SYMFLUENCE native, so this is
*spec-validated*: the assertions reproduce the *published product spec* on a
synthetic inline fixture — the ``DN * 2`` mm scale factor, the ``0..240`` valid
DN range, the ``241..255`` flag/fill sentinels (and the NetCDF ``_FillValue``),
the canonical ``mm`` unit, the half-open UTC window, and the gridded reduction —
with no network and no auth.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.amsr_swe import (
    FILL_VALUE,
    MAX_VALID_DN,
    MAX_VALID_SWE_MM,
    SOURCE_SWE_SCALE,
    AMSR2SWEConnector,
)
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


def _spec(area_km2: float = 8000.0) -> ReductionSpec:
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


@pytest.fixture
def amsr_arrays():
    """Synthetic AMSR2-like SWE grid of *stored digital numbers* (DN).

    DN 50 -> 100 mm everywhere on the valid layer; one fill-value cell and one
    flag-sentinel cell (DN 252) are present to be masked.
    """
    times = np.array(["2024-06-15", "2024-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    dn = np.full((2, 3, 3), 50.0)         # 50 counts -> 100 mm
    dn[0, 0, 0] = FILL_VALUE              # NetCDF fill -> masked
    dn[0, 1, 1] = 252.0                   # flag sentinel (241..255) -> masked
    return lats, lons, times, dn


def test_published_spec_constants():
    """Spec contract: AU_DySno scale is 2 mm/count over a 0..240 valid DN range."""
    assert SOURCE_SWE_SCALE == 2.0
    assert MAX_VALID_DN == 240.0
    assert MAX_VALID_SWE_MM == 480.0  # 240 counts * 2 mm/count


def test_scale_conversion_dn_to_mm(amsr_arrays):
    """DN are scaled to mm at the boundary: 50 counts -> 100 mm w.e."""
    lats, lons, times, dn = amsr_arrays
    conn = AMSR2SWEConnector()
    series = conn.reduce_arrays(
        lats, lons, times, dn, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SWE
    assert series.unit == "mm"  # canonical SWE unit
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Every valid cell is 50 DN -> 100 mm; masked cells are dropped from the mean.
    for p in series.points:
        assert p.value == pytest.approx(100.0, abs=1e-6)
        assert p.quality.value == "good"


def test_flag_and_fill_cells_masked_before_mean(amsr_arrays):
    """Spec: fill value and flag sentinels (241..255) are not SWE -> excluded."""
    lats, lons, times, dn = amsr_arrays
    conn = AMSR2SWEConnector()
    # If the fill (-9999) or flag (252) leaked into the mean, the basin value
    # would not be a clean 100 mm. It is, so they were masked.
    series = conn.reduce_arrays(
        lats, lons, times, dn, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert all(p.value == pytest.approx(100.0, abs=1e-6) for p in series.points)


def test_all_fill_layer_reduces_to_missing():
    """A timestep with only flag/fill cells reduces to MISSING (value None)."""
    times = np.array(["2024-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    dn = np.full((1, 3, 3), 250.0)  # all flag sentinels -> all masked
    conn = AMSR2SWEConnector()
    series = conn.reduce_arrays(
        lats, lons, times, dn, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert len(series.points) == 1
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_max_valid_dn_is_kept_above_is_masked():
    """Spec boundary: DN 240 is valid (480 mm); DN 241 is the first flag -> masked."""
    times = np.array(["2024-06-15"], dtype="datetime64[ns]")
    lats = np.array([51.0])
    lons = np.array([-115.0])
    conn = AMSR2SWEConnector()
    point_spec = ReductionSpec(
        domain_name="cell", centroid=(51.0, -115.0),
        reduction=SpatialReduction.NEAREST_CELL,
    )

    valid = conn.reduce_arrays(
        lats, lons, times, np.array([[[MAX_VALID_DN]]]), point_spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert valid.points[0].value == pytest.approx(MAX_VALID_SWE_MM, abs=1e-6)

    flagged = conn.reduce_arrays(
        lats, lons, times, np.array([[[MAX_VALID_DN + 1.0]]]), point_spec,
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert flagged.points[0].value is None
    assert flagged.points[0].quality.value == "missing"


def test_window_trim_half_open(amsr_arrays):
    """Half-open [2024-06-01, 2024-07-15): includes 06-15, excludes 07-15."""
    lats, lons, times, dn = amsr_arrays
    conn = AMSR2SWEConnector()
    series = conn.reduce_arrays(
        lats, lons, times, dn, _spec(),
        datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 7, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2024, 6) in months
    assert (2024, 7) not in months


def test_small_basin_defaults_to_nearest_cell(amsr_arrays):
    lats, lons, times, dn = amsr_arrays
    conn = AMSR2SWEConnector()
    series = conn.reduce_arrays(
        lats, lons, times, dn, _spec(area_km2=500.0),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("amsr_swe:cell:")


@pytest.mark.asyncio
async def test_list_sites_returns_reduced_region():
    conn = AMSR2SWEConnector()
    sites = await conn.list_sites(_spec())
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "amsr_swe:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = AMSR2SWEConnector()
    with pytest.raises(Exception, match="cached file"):
        await conn.fetch_series(
            _spec(), datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_reduce_file_roundtrip_netcdf(tmp_path, amsr_arrays):
    """End-to-end through the NetCDF reader: scale + mask + reduce on a real file."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats, lons, times, dn = amsr_arrays
    ds = xr.Dataset(
        {"SWE": (("time", "lat", "lon"), dn)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "amsr_swe_synth.nc"
    ds.to_netcdf(path)

    conn = AMSR2SWEConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mm"
    assert all(p.value == pytest.approx(100.0, abs=1e-6) for p in series.points)


# -- regression tests for the real AU_DySno granule shape ---------------------
# The synthetic fixtures above used a 1-D lat/lon grid and assumed every layer is
# DN/2. The real AMSR2 AU_DySno product breaks both assumptions: it ships 2-D
# 721x721 EASE-Grid coords (with inf off-Earth fills) and a *hemisphere-dependent*
# scale (Northern daily is already mm, scale_factor=1.0; Southern is DN/2).


def test_northern_daily_scale_is_one_not_two(tmp_path):
    """REGRESSION: SWE_NorthernDaily metadata scale_factor=1.0 must be honored.

    The old connector hardcoded DN*2 for all data, doubling NH SWE (29.571 mm
    where the truth was 14.786 mm). Reading the variable's scale_factor fixes it.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2024-02-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    dn = np.full((1, 3, 3), 15.0)  # 15 counts; NH layer is already mm -> 15 mm
    ds = xr.Dataset(
        {"SWE_NorthernDaily": (("time", "lat", "lon"), dn)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds["SWE_NorthernDaily"].attrs["scale_factor"] = 1.0  # "0-240 SWE mm"
    path = tmp_path / "amsr_nh.nc"
    ds.to_netcdf(path)

    conn = AMSR2SWEConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
    )
    # With the bug (DN*2) this would be 30 mm; correct NH scale gives 15 mm.
    assert series.points[0].value == pytest.approx(15.0, abs=1e-6)
    assert series.source_info["scale_mm_per_count"] == "1"


def test_two_dimensional_ease_grid_coords_reduce(tmp_path):
    """REGRESSION: real AU_DySno stores 2-D (721x721) EASE-Grid lat/lon.

    reduce_grid assumes 1-D coord vectors and raises IndexError on 2-D coords.
    The 2-D reduction path masks the bbox cells (and off-Earth inf fills) and
    reduces over them instead.
    """
    times = np.array(["2024-02-15"], dtype="datetime64[ns]")
    # 4x4 EASE-Grid-like patch with 2-D curvilinear coords.
    lat2d, lon2d = np.meshgrid(
        np.array([49.5, 50.5, 51.5, 52.5]),
        np.array([-116.5, -115.5, -114.5, -113.5]),
        indexing="ij",
    )
    # Off-Earth corner: non-finite coords + inf value (the real product's fill).
    lat2d[0, 0] = np.inf
    lon2d[0, 0] = np.inf
    dn = np.full((1, 4, 4), 20.0)  # 20 counts
    dn[0, 0, 0] = np.inf           # off-Earth fill -> masked
    dn[0, 3, 3] = 252.0            # flag sentinel -> masked

    conn = AMSR2SWEConnector()
    # scale=1.0 (NH): valid cells -> 20 mm; basin_mean over the in-bbox finite cells.
    series = conn.reduce_arrays(
        lat2d, lon2d, times, dn, _spec(),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        scale=1.0,
    )
    assert series.unit == "mm"
    assert len(series.points) == 1
    assert series.points[0].value == pytest.approx(20.0, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_two_dimensional_coords_nearest_cell(tmp_path):
    """2-D coord nearest_cell picks the nearest valid in-grid cell to the centroid."""
    times = np.array(["2024-02-15"], dtype="datetime64[ns]")
    lat2d, lon2d = np.meshgrid(
        np.array([50.0, 51.0, 52.0]),
        np.array([-116.0, -115.0, -114.0]),
        indexing="ij",
    )
    dn = np.zeros((1, 3, 3))
    dn[0, 1, 1] = 30.0  # the centroid (51, -115) cell
    conn = AMSR2SWEConnector()
    series = conn.reduce_arrays(
        lat2d, lon2d, times, dn, _spec(area_km2=500.0),
        datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC),
        scale=1.0,
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.points[0].value == pytest.approx(30.0, abs=1e-6)


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_amsr_swe():
    """LIVE smoke against a real AU_DySno file (requires Earthdata + a cached path).

    Run with: pytest -m network tests/connectors/test_amsr_swe.py -k live
    """
    import os

    path = os.environ.get("AMSR_SWE_NC_PATH")
    if not path:
        pytest.skip("set AMSR_SWE_NC_PATH to a cached AU_DySno NetCDF for the live smoke")
    conn = AMSR2SWEConnector({"nc_path": path})
    async with conn:
        series_list = await conn.fetch_series(
            _spec(), datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC),
        )
    assert series_list and series_list[0].unit == "mm"
