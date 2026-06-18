"""MODIS NDVI connector — hermetic test of the gridded basin-reduction path.

Builds synthetic in-memory MOD13 scaled-integer (DN) grids and reduces them; no
network, no auth, no HDF dependency. There is no SYMFLUENCE native vegetation-
index handler, so these tests are SPEC-VALIDATED: they assert the connector
reproduces the published MOD13 product spec (scale factor 0.0001, valid range
-2000..10000, fill -3000) on the synthetic fixture.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_ndvi import (
    FILL_VALUE_DN,
    SCALE_FACTOR,
    VALID_MAX_DN,
    MODISNDVIConnector,
)
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


def _grid():
    """A small MOD13-like DN grid: (time, lat, lon), scaled 16-bit integers."""
    times = np.array(
        ["2020-06-01", "2020-06-17", "2020-07-03"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    # DN values: 5000 -> NDVI 0.5 ; 8000 -> 0.8 ; 2000 -> 0.2.
    data = np.empty((3, 3, 3), dtype="float64")
    data[0] = 5000.0
    data[1] = 8000.0
    data[2] = 2000.0
    return lats, lons, times, data


def _spec(area_km2=8000.0):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_scale_factor_dn_to_ratio_basin_mean():
    """DN * 0.0001 at the boundary; canonical unit is the dimensionless '1'."""
    conn = MODISNDVIConnector()
    lats, lons, times, data = _grid()
    series = conn.reduce_arrays(
        lats, lons, times, data, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.VEGETATION_INDEX
    assert series.unit == "1"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Uniform layers: basin-mean equals the per-layer scaled value.
    # The two June obs are DN 5000 -> 0.5 and DN 8000 -> 0.8.
    june = sorted(p.value for p in series.points if p.timestamp.month == 6)
    assert june == pytest.approx([0.5, 0.8], abs=1e-9)
    july = [p.value for p in series.points if p.timestamp.month == 7]
    assert july[0] == pytest.approx(2000.0 * SCALE_FACTOR, abs=1e-9)  # 0.2


def test_valid_range_upper_bound_kept():
    """A DN at the spec upper bound (10000 -> 1.0) survives the mask."""
    conn = MODISNDVIConnector()
    lats, lons, times, data = _grid()
    data[0] = VALID_MAX_DN  # 10000 -> 1.0, on the inclusive boundary
    series = conn.reduce_arrays(
        lats, lons, times, data, _spec(),
        datetime(2020, 5, 1, tzinfo=UTC), datetime(2020, 6, 2, tzinfo=UTC),
    )
    vals = [p.value for p in series.points]
    assert vals == [pytest.approx(VALID_MAX_DN * SCALE_FACTOR, abs=1e-9)]  # 1.0


def test_fill_value_maps_to_missing():
    """Fill DN (-3000) and out-of-range DN reduce to MISSING (None)."""
    conn = MODISNDVIConnector()
    lats, lons, times, data = _grid()
    data[0] = FILL_VALUE_DN          # whole layer is fill
    data[2] = VALID_MAX_DN + 5000.0  # whole layer out of valid range (too high)
    series = conn.reduce_arrays(
        lats, lons, times, data, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_month = {p.timestamp.month: p for p in series.points}
    # June layer 0 is all-fill -> MISSING.
    fill_pts = [p for p in series.points if p.timestamp == datetime(2020, 6, 1, tzinfo=UTC)]
    assert fill_pts[0].value is None
    assert fill_pts[0].quality == QualityFlag.MISSING
    # July out-of-range -> MISSING.
    assert by_month[7].value is None
    assert by_month[7].quality == QualityFlag.MISSING


def test_partial_fill_reduces_over_valid_cells_only():
    """Within a layer, fill cells are skipped; the mean is over valid cells."""
    conn = MODISNDVIConnector()
    lats, lons, times, data = _grid()
    data[1] = 8000.0
    data[1, 0, 0] = FILL_VALUE_DN  # one cell fill in the 2020-06-17 layer
    series = conn.reduce_arrays(
        lats, lons, times, data, _spec(),
        datetime(2020, 6, 10, tzinfo=UTC), datetime(2020, 6, 20, tzinfo=UTC),
    )
    vals = [p.value for p in series.points]
    # Remaining cells are all 8000 -> 0.8; the masked cell does not bias it.
    assert vals == [pytest.approx(0.8, abs=1e-9)]


def test_window_trim_half_open():
    conn = MODISNDVIConnector()
    lats, lons, times, data = _grid()
    # Half-open [2020-06-01, 2020-07-03): includes 06-01 & 06-17, excludes 07-03.
    series = conn.reduce_arrays(
        lats, lons, times, data, _spec(),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 3, tzinfo=UTC),
    )
    days = {(p.timestamp.month, p.timestamp.day) for p in series.points}
    assert (6, 1) in days
    assert (6, 17) in days
    assert (7, 3) not in days


def test_small_basin_defaults_to_nearest_cell():
    conn = MODISNDVIConnector()
    lats, lons, times, data = _grid()
    series = conn.reduce_arrays(
        lats, lons, times, data, _spec(area_km2=500.0),  # small -> nearest_cell
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_ndvi:cell:")


def test_reduce_file_netcdf_roundtrip(tmp_path):
    """reduce_file extracts a NetCDF NDVI variable and reproduces the spec."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    lats, lons, times, data = _grid()
    ds = xr.Dataset(
        {"NDVI": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "mod13_synth.nc"
    ds.to_netcdf(path)

    conn = MODISNDVIConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "1"
    july = [p.value for p in series.points if p.timestamp.month == 7]
    assert july[0] == pytest.approx(0.2, abs=1e-9)


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = MODISNDVIConnector()
    spec = _spec()
    with pytest.raises(Exception, match="cached file"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


@pytest.mark.network
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_lpdaac_fetch():  # pragma: no cover - requires Earthdata netrc
    pytest.skip("Live LP DAAC Earthdata download not yet wired")
