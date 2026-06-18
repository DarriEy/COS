"""VIIRS snow-cover connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory VIIRS-like NetCDF (NDSI snow-cover percent with fill
codes) and reduces it; no network, no auth. Proves the architecture-critical
gridded -> canonical-series path: fill masking, percent->fraction canonicalization,
basin_mean vs nearest_cell, and half-open UTC window-trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.viirs_sca import NDSI_FILL_VALUES, VIIRSSnowCoverConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def viirs_nc(tmp_path):
    """A synthetic VIIRS-like NetCDF: NDSI snow cover (percent) over a small grid.

    Layout (lat 50/51/52 x lon 244/245/246, 0-360 == -116..-114):
      t0 2020-01-15 : all 50 %  (-> fraction 0.5 everywhere)
      t1 2020-02-15 : all 100 % (-> fraction 1.0)
      t2 2020-03-15 : a mix of valid + fill codes (cloud 250, missing 255)
      t3 2020-04-15 : all 0 %   (-> fraction 0.0)
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-01-15", "2020-02-15", "2020-03-15", "2020-04-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    data = np.empty((4, 3, 3), dtype="float64")
    data[0] = 50.0
    data[1] = 100.0
    # t2: half the cells are valid 80 %, half are fill codes -> masked away.
    data[2] = 80.0
    data[2, 0, 0] = 250.0  # cloud fill
    data[2, 1, 1] = 255.0  # missing fill
    data[2, 2, 2] = 200.0  # no-decision fill
    data[3] = 0.0
    ds = xr.Dataset(
        {"CGF_NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "viirs_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2=8000.0, reduction=None):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
        reduction=reduction,
    )


def test_units_and_percent_to_fraction(viirs_nc):
    conn = VIIRSSnowCoverConnector()
    series = conn.reduce_file(
        viirs_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SNOW_COVER
    assert series.unit == "fraction"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"

    by_month = {p.timestamp.month: p.value for p in series.points}
    # 50 % -> 0.5, 100 % -> 1.0, 0 % -> 0.0 (basin mean of uniform layers).
    assert by_month[1] == pytest.approx(0.5, abs=1e-9)
    assert by_month[2] == pytest.approx(1.0, abs=1e-9)
    assert by_month[4] == pytest.approx(0.0, abs=1e-9)


def test_fill_codes_masked_then_mean_over_valid_only(viirs_nc):
    conn = VIIRSSnowCoverConnector()
    series = conn.reduce_file(
        viirs_nc, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_month = {p.timestamp.month: p.value for p in series.points}
    # t2: 3 cells are fill codes, the other 6 are 80 % -> fraction 0.8 each.
    # Masked cells drop out of the mean entirely, so the result is 0.8, not diluted.
    assert by_month[3] == pytest.approx(0.8, abs=1e-9)
    # Every fill value the native handler drops is in our mask set.
    for code in (250, 255, 200):
        assert code in NDSI_FILL_VALUES


def test_all_fill_layer_is_missing(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-05-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.full((1, 3, 3), 255.0)  # entirely missing
    ds = xr.Dataset(
        {"NDSI_Snow_Cover": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "allfill.nc"
    ds.to_netcdf(path)
    conn = VIIRSSnowCoverConnector()
    series = conn.reduce_file(
        path, _spec(),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert len(series.points) == 1
    p = series.points[0]
    assert p.value is None
    assert p.quality == QualityFlag.MISSING


def test_small_basin_defaults_to_nearest_cell(viirs_nc):
    conn = VIIRSSnowCoverConnector()
    series = conn.reduce_file(
        viirs_nc, _spec(area_km2=500.0),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("viirs_sca:cell:")
    # nearest cell to centroid is uniform per layer -> same fractions.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[1] == pytest.approx(0.5, abs=1e-9)
    assert by_month[2] == pytest.approx(1.0, abs=1e-9)


def test_window_trim_half_open(viirs_nc):
    conn = VIIRSSnowCoverConnector()
    # Half-open [2020-02-01, 2020-04-15): includes 02-15 and 03-15, excludes 04-15.
    series = conn.reduce_file(
        viirs_nc, _spec(),
        datetime(2020, 2, 1, tzinfo=UTC), datetime(2020, 4, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert 2 in months
    assert 3 in months
    assert 1 not in months  # before window
    assert 4 not in months  # == end, excluded (half-open)


def test_explicit_reduction_override(viirs_nc):
    conn = VIIRSSnowCoverConnector()
    series = conn.reduce_file(
        viirs_nc, _spec(area_km2=8000.0, reduction=SpatialReduction.NEAREST_CELL),
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    # Large basin but explicit override wins.
    assert series.reduction == SpatialReduction.NEAREST_CELL


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = VIIRSSnowCoverConnector()
    spec = _spec()
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
