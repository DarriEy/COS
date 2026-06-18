"""Tests for the gridded spatial-reduction kernels (the gridded→series path)."""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.core.exceptions import ReductionError
from cos.core.models import ObservationKind, SpatialReduction
from cos.core.reduce import basin_mean, nearest_cell, reduce_grid, reduce_grid_2d


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


# -- reduce_grid_2d: the shared 2-D-coordinate kernel (swath/fixed-grid/EASE) ----


def test_reduce_grid_2d_basin_mean_masks_off_grid_inf():
    """2-D basin_mean: in-bbox finite cells averaged; off-grid inf cells dropped."""
    times = [datetime(2020, 6, 15, tzinfo=UTC)]
    lat2d, lon2d = np.meshgrid(np.array([50.0, 51.0]), np.array([-116.0, -115.0]), indexing="ij")
    lat2d[0, 0] = np.inf  # off-grid corner
    lon2d[0, 0] = np.inf
    values = np.full((1, 2, 2), 4.0)
    values[0, 1, 1] = 6.0
    pts = reduce_grid_2d(
        lat2d, lon2d, times, values,
        reduction=SpatialReduction.BASIN_MEAN,
        bbox=(49.0, -117.0, 52.0, -113.0), point=None,
    )
    # The inf corner is dropped; mean of the three finite 4/4/6 cells (cos-lat ~equal).
    assert pts[0].value == pytest.approx((4.0 + 4.0 + 6.0) / 3.0, abs=0.02)


def test_reduce_grid_2d_basin_mean_0_360_with_inf():
    """0-360 grid + a -180..180 bbox reduces; off-grid inf must not poison nanmax."""
    times = [datetime(2020, 6, 15, tzinfo=UTC)]
    lat2d, lon2d = np.meshgrid(np.array([50.0, 51.0]), np.array([244.0, 245.0, 246.0]), indexing="ij")
    lon2d[0, 0] = np.inf  # off-grid: would make nanmax inf and break 0-360 detection
    values = np.full((1, 2, 3), 7.0)
    pts = reduce_grid_2d(
        lat2d, lon2d, times, values,
        reduction=SpatialReduction.BASIN_MEAN,
        bbox=(50.0, -116.0, 51.0, -114.0), point=None,  # -116..-114 == 244..246
    )
    assert pts[0].value == pytest.approx(7.0)


def test_reduce_grid_2d_nearest_cell_great_circle_across_seam():
    """nearest_cell uses great-circle distance: a closer across-seam cell wins."""
    times = [datetime(2020, 6, 15, tzinfo=UTC)]
    lat2d, lon2d = np.meshgrid(np.array([51.0]), np.array([10.0, 355.0]), indexing="ij")
    values = np.full((1, 1, 2), 1.0)
    values[0, 0, 1] = 9.0  # the 355E (=-5, 5 deg from lon 0) cell — planar would miss it
    pts = reduce_grid_2d(
        lat2d, lon2d, times, values,
        reduction=SpatialReduction.NEAREST_CELL, bbox=None, point=(51.0, 0.0),
    )
    assert pts[0].value == pytest.approx(9.0)


def test_reduce_grid_2d_empty_bbox_raises_with_label():
    times = [datetime(2020, 6, 15, tzinfo=UTC)]
    lat2d, lon2d = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]), indexing="ij")
    values = np.zeros((1, 2, 2))
    with pytest.raises(ReductionError, match="EASE-Grid"):
        reduce_grid_2d(
            lat2d, lon2d, times, values,
            reduction=SpatialReduction.BASIN_MEAN,
            bbox=(40.0, 40.0, 41.0, 41.0), point=None, grid_label="EASE-Grid",
        )
