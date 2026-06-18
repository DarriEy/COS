# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""GLEAM ET connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory GLEAM-like NetCDF (variable ``E`` in mm/day) and
reduces it; no network, no GLEAM credentials. Proves the gridded -> canonical
ET-series path, the canonical mm/day unit, and half-open window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.gleam_et import GLEAMETConnector
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


@pytest.fixture
def gleam_nc(tmp_path):
    """A synthetic GLEAM-like NetCDF: E (mm/day) over a small 0-360 grid."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2015-01-15", "2015-02-15", "2015-03-15", "2015-04-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    # mm/day: constant per timestep so basin-mean == that constant.
    data = np.empty((4, 3, 3))
    data[0] = 1.0
    data[1] = 2.0
    data[2] = 3.0
    data[3] = 4.0
    ds = xr.Dataset(
        {"E": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "gleam_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_units_mm_per_day(gleam_nc):
    conn = GLEAMETConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.ET
    assert series.unit == "mm/day"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "gleam_et:domain:bow"
    # Constant-per-layer field -> basin-mean equals the layer constant.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[1] == pytest.approx(1.0, abs=1e-9)
    assert by_month[2] == pytest.approx(2.0, abs=1e-9)
    assert by_month[4] == pytest.approx(4.0, abs=1e-9)


def test_unit_conversion_multiplier(gleam_nc):
    # Mirrors native ET_UNIT_CONVERSION: apply a boundary multiplier.
    conn = GLEAMETConnector(config={"unit_conversion": 0.5})
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[4] == pytest.approx(2.0, abs=1e-9)  # 4.0 * 0.5


def test_small_basin_defaults_to_nearest_cell(gleam_nc):
    conn = GLEAMETConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("gleam_et:cell:")
    # Constant layers -> nearest-cell value equals the layer constant.
    by_month = {p.timestamp.month: p.value for p in series.points}
    assert by_month[3] == pytest.approx(3.0, abs=1e-9)


def test_window_trim_half_open(gleam_nc):
    conn = GLEAMETConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2015-02-01, 2015-04-15): includes 02-15 and 03-15, excludes 04-15.
    series = conn.reduce_file(
        gleam_nc, spec,
        datetime(2015, 2, 1, tzinfo=UTC), datetime(2015, 4, 15, tzinfo=UTC),
    )
    months = {p.timestamp.month for p in series.points}
    assert months == {2, 3}
    assert 4 not in months  # half-open excludes the exact end timestamp


def test_variable_autodetect_evaporation_name(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2015-01-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0])
    lons = np.array([244.0, 245.0])
    data = np.full((1, 2, 2), 2.5)
    ds = xr.Dataset(
        {"evaporation": (("time", "latitude", "longitude"), data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    path = tmp_path / "gleam_alt.nc"
    ds.to_netcdf(path)

    conn = GLEAMETConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 51.0, -115.0),
                         centroid=(50.5, -115.5), area_km2=8000.0)
    series = conn.reduce_file(
        path, spec,
        datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
    )
    assert series.source_info["variable"] == "evaporation"
    assert series.points[0].value == pytest.approx(2.5, abs=1e-9)


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = GLEAMETConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0))
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC),
        )
