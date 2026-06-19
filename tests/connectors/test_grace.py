"""GRACE connector — hermetic test of the gridded basin-reduction path.

Builds a synthetic in-memory GRACE NetCDF and reduces it; no network, no auth.
This proves the architecture-critical gridded → canonical-series path.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.grace import GRACEConnector
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction


@pytest.fixture
def grace_nc(tmp_path):
    """A synthetic GRACE-like NetCDF: lwe_thickness (cm) over a small grid."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2003-06-15", "2004-06-15", "2020-06-15", "2020-07-15"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    # cm values: baseline years ~2.0 cm, 2020 ~ 5.0 cm.
    data = np.empty((4, 3, 3))
    data[0] = 2.0
    data[1] = 2.0
    data[2] = 5.0
    data[3] = 5.0
    ds = xr.Dataset(
        {"lwe_thickness": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "grace_synth.nc"
    ds.to_netcdf(path)
    return path


def test_reduce_file_basin_mean_cm_to_mm_anomaly(grace_nc):
    conn = GRACEConnector()
    spec = ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_file(
        grace_nc, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.TWS
    assert series.unit == "mm"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Baseline (2003-2008) mean = 20 mm (2 cm). 2020 = 50 mm -> anomaly +30 mm.
    by_year = {p.timestamp.year: p.value for p in series.points}
    assert by_year[2003] == pytest.approx(0.0, abs=1e-6)
    assert by_year[2020] == pytest.approx(30.0, abs=1e-6)


def test_small_basin_defaults_to_nearest_cell(grace_nc):
    conn = GRACEConnector()
    spec = ReductionSpec(
        domain_name="tiny",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        grace_nc, spec,
        datetime(2003, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("grace:cell:")


def test_window_trim_half_open(grace_nc):
    conn = GRACEConnector()
    spec = ReductionSpec(domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
                         centroid=(51.0, -115.0), area_km2=8000.0)
    # Half-open [2020-06-01, 2020-07-15): includes the 06-15 obs, excludes 07-15.
    series = conn.reduce_file(
        grace_nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 15, tzinfo=UTC),
    )
    years_months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in years_months
    assert (2020, 7) not in years_months


@pytest.mark.asyncio
async def test_fetch_series_live_fetches_when_no_path(grace_nc, monkeypatch):
    """With no nc_path, fetch_series live-fetches the JPL mascon via Earthdata and
    reduces it. The earthaccess download is mocked to the synthetic fixture (no
    network); this asserts the wiring (no path -> _live_fetch -> earthaccess_granules
    -> reduce_file)."""
    from cos.core import fetch as cos_fetch

    seen = {}

    def fake_granules(short_name, version, temporal, bbox, dest_dir, **kw):
        seen["short_name"] = short_name
        return [grace_nc]
    monkeypatch.setattr(cos_fetch, "earthaccess_granules", fake_granules)

    conn = GRACEConnector()
    spec = ReductionSpec(domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0), centroid=(51.0, -115.0))
    series = await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
    assert seen["short_name"] == GRACEConnector.EARTHDATA_SHORTNAME
    assert series and series[0].unit == "mm" and series[0].kind.value == "tws"
