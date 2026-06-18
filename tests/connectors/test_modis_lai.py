"""MODIS LAI connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory MODIS-LAI-like NetCDF (digital numbers) and reduces
it; no network, no auth. Proves the architecture-critical gridded → canonical
path for Leaf Area Index: the 0.1 DN scale factor to the canonical dimensionless
unit ("1"), fill (255) / out-of-range masking, the QC algorithm-path filter,
basin-mean vs nearest-cell reduction, and half-open UTC window trim.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.modis_lai import LAI_FILL_VALUE, MODISLAIConnector
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def lai_nc(tmp_path):
    """A synthetic MODIS-LAI NetCDF: Lai_500m digital numbers over a small grid.

    Four 8-day timesteps on a 3x3 grid (0-360 longitudes, = -116..-114). Values
    are raw DN (pre-scale): layer 0 uniform DN 30 (-> LAI 3.0), layer 1 DN 50
    with one out-of-range cell (DN 200 > 100, masked), layer 2 DN 20, layer 3
    entirely fill (255) so it must reduce to MISSING.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-09", "2020-06-17", "2020-06-25", "2020-07-03"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    data = np.empty((4, 3, 3))
    data[0] = 30.0           # uniform DN 30 -> LAI 3.0
    data[1] = 50.0           # uniform DN 50 -> LAI 5.0
    data[1, 0, 0] = 200.0    # out-of-range DN -> masked, mean stays 5.0
    data[2] = 20.0           # uniform DN 20 -> LAI 2.0
    data[3] = LAI_FILL_VALUE  # entirely 255 fill -> MISSING
    ds = xr.Dataset(
        {"Lai_500m": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_lai_synth.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def lai_qc_nc(tmp_path):
    """A 1-timestep LAI NetCDF with a QC layer to exercise the algorithm filter.

    DN 40 everywhere (-> LAI 4.0). QC bits 5-7: one cell main (0 -> keep), one
    saturation (2 -> keep), the rest backup (1 -> drop). The two kept cells both
    hold DN 40, so the basin mean is LAI 4.0.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-09"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])
    data = np.full((1, 3, 3), 40.0)
    qc = np.full((1, 3, 3), 1 << 5, dtype="int16")  # algorithm path 1 (backup) -> drop
    qc[0, 0, 0] = 0 << 5      # main (0) -> keep
    qc[0, 1, 1] = 2 << 5      # saturation (2) -> keep
    ds = xr.Dataset(
        {
            "Lai_500m": (("time", "lat", "lon"), data),
            "FparLai_QC": (("time", "lat", "lon"), qc),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "modis_lai_qc_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_reduce_file_basin_mean_units_and_scale(lai_nc):
    conn = MODISLAIConnector()
    series = conn.reduce_file(
        lai_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.LAI
    assert series.unit == "1"  # canonical dimensionless LAI
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "modis_lai:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # DN 30 * 0.1 = LAI 3.0.
    assert by_day[9].value == pytest.approx(3.0, abs=1e-9)
    assert by_day[9].quality == QualityFlag.GOOD
    # Out-of-range DN 200 masked; remaining DN 50 * 0.1 = 5.0.
    assert by_day[17].value == pytest.approx(5.0, abs=1e-9)
    # DN 20 * 0.1 = 2.0.
    assert by_day[25].value == pytest.approx(2.0, abs=1e-9)


def test_fill_value_reduces_to_missing(lai_nc):
    conn = MODISLAIConnector()
    series = conn.reduce_file(
        lai_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # The all-fill (255) layer must surface as MISSING with no value.
    assert by_day[3].value is None
    assert by_day[3].quality == QualityFlag.MISSING


def test_qc_algorithm_filter(lai_qc_nc):
    conn = MODISLAIConnector()
    series = conn.reduce_file(
        lai_qc_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    # Only the main (0) + saturation (2) cells survive; both DN 40 -> LAI 4.0.
    assert series.points[0].value == pytest.approx(4.0, abs=1e-9)
    assert series.points[0].quality == QualityFlag.GOOD


def test_small_basin_defaults_to_nearest_cell(lai_nc):
    conn = MODISLAIConnector()
    series = conn.reduce_file(
        lai_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 8, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("modis_lai:cell:")
    # Nearest cell to centroid is uniform within each layer: DN 30 * 0.1 = 3.0.
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[9].value == pytest.approx(3.0, abs=1e-9)


def test_window_trim_half_open(lai_nc):
    conn = MODISLAIConnector()
    # Half-open [06-09, 06-25): includes 06-09 and 06-17, excludes 06-25.
    series = conn.reduce_file(
        lai_nc, _spec(8000.0),
        datetime(2020, 6, 9, tzinfo=UTC), datetime(2020, 6, 25, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {9, 17}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = MODISLAIConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.network
@pytest.mark.live
@pytest.mark.asyncio
async def test_live_earthdata_fetch_placeholder():
    """Live Earthdata fetch is not wired (parity is the reduce path); skip."""
    pytest.skip("MODIS LAI live Earthdata download is not wired in this connector")
