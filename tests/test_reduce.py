"""Tests for the gridded spatial-reduction kernels (the gridded→series path)."""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.core.exceptions import ReductionError
from cos.core.models import ObservationKind, SpatialReduction
from cos.core.reduce import basin_mean, nearest_cell, reduce_grid


def _grid():
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    # (time=2, lat=3, lon=3); value = lat index for layer 0, +10 for layer 1.
    base = np.tile(np.arange(3.0)[:, None], (1, 3))
    values = np.stack([base, base + 10.0])
    times = [datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 2, 1, tzinfo=UTC)]
    return lats, lons, times, values


def test_basin_mean_area_weighted():
    lats, lons, _t, values = _grid()
    out = basin_mean(lats, lons, values, (50.0, -116.0, 52.0, -114.0))
    # Layer 0 mean of {0,1,2} cos-lat weighted ~ near 1.0; layer 1 ~ +10.
    assert out[0] == pytest.approx(1.0, abs=0.05)
    assert out[1] == pytest.approx(11.0, abs=0.05)


def test_basin_mean_no_cells_raises():
    lats, lons, _t, values = _grid()
    with pytest.raises(ReductionError):
        basin_mean(lats, lons, values, (10.0, 10.0, 11.0, 11.0))


def test_nearest_cell_picks_closest():
    lats, lons, _t, values = _grid()
    out = nearest_cell(lats, lons, values, (51.4, -115.1))  # nearest lat idx 1
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(11.0)


def test_basin_mean_handles_0_360_longitudes():
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 convention (=-116..-114)
    values = np.ones((1, 2, 3))
    out = basin_mean(lats, lons, values, (50.0, -116.0, 51.0, -114.0))
    assert out[0] == pytest.approx(1.0)


def test_reduce_grid_marks_nan_missing():
    lats, lons, times, values = _grid()
    values[0, :, :] = np.nan  # whole first layer missing
    points = reduce_grid(
        lats, lons, times, values,
        reduction=SpatialReduction.BASIN_MEAN,
        bbox=(50.0, -116.0, 52.0, -114.0), point=None,
        kind=ObservationKind.TWS, unit="mm",
    )
    assert points[0].value is None
    assert points[0].quality.value == "missing"
    assert points[1].value is not None
    assert points[1].timestamp.tzinfo is not None  # UTC-aware
