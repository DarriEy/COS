"""TROPOMI SIF connector — hermetic test of the gridded basin-reduction path.

TROPOMI SIF has NO SYMFLUENCE native, so this is *spec-validated*: the
assertions reproduce the published Caltech gridded TROPOMI SIF product spec on a
synthetic inline fixture — the mW/m²/nm/sr unit (identity scale), the -999 fill
sentinel, the physical valid band, the half-open UTC window, and the cos-lat
basin reduction — with no network and no auth.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.tropomi_sif import (
    SIF_FILL_VALUE,
    SOURCE_SIF_SCALE,
    VALID_SIF_RANGE,
    TROPOMISIFConnector,
)
from cos.core.models import KIND_UNITS, ObservationKind, ReductionSpec, SpatialReduction


@pytest.fixture
def sif_nc(tmp_path):
    """A synthetic Caltech-like TROPOMI SIF NetCDF (mW/m²/nm/sr) with a fill cell."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(["2020-06-15", "2020-07-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    data = np.full((2, 3, 3), 2.0)  # uniform 2.0 mW/m²/nm/sr
    data[0, 0, 0] = SIF_FILL_VALUE   # one fill cell -> masked to NaN before mean
    ds = xr.Dataset(
        {"sif": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "tropomi_sif_synth.nc"
    ds.to_netcdf(path)
    return path


def test_canonical_unit_and_identity_scale_spec():
    """Spec contract: the product is already mW/m²/nm/sr -> identity boundary scale."""
    assert SOURCE_SIF_SCALE == 1.0
    assert KIND_UNITS[ObservationKind.SIF] == "mW/m2/nm/sr"


def test_reduce_arrays_basin_mean_identity_scale_and_fill_mask():
    """Spec: uniform SIF passes through unchanged; the -999 fill cell is masked out."""
    conn = TROPOMISIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), 2.0)
    values[0, 1, 1] = SIF_FILL_VALUE  # masked, must not pull the mean down
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,  # large -> basin_mean
    )
    series = conn.reduce_arrays(
        lats, lons, times, values, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SIF
    assert series.unit == "mW/m2/nm/sr"
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    # Identity scale: remaining valid cells are all 2.0 -> mean 2.0 (fill excluded).
    assert series.points[0].value == pytest.approx(2.0, abs=1e-6)
    assert series.points[0].quality.value == "good"


def test_reduce_file_basin_mean(sif_nc):
    conn = TROPOMISIFConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_file(
        sif_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.unit == "mW/m2/nm/sr"
    for p in series.points:
        assert p.value == pytest.approx(2.0, abs=1e-6)
        assert p.quality.value == "good"
    assert series.source_info["variable"] == "sif"


def test_fill_only_cell_reduces_to_missing():
    """Spec: a timestep whose in-bbox cells are all fill -> None / MISSING."""
    conn = TROPOMISIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), SIF_FILL_VALUE)  # entire layer is fill
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_arrays(
        lats, lons, times, values, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"


def test_out_of_range_masked_to_missing():
    """Spec: a finite SIF outside VALID_SIF_RANGE is treated as invalid -> MISSING."""
    lo, hi = VALID_SIF_RANGE
    conn = TROPOMISIFConnector()
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([-116.0, -115.0, -114.0])
    values = np.full((1, 3, 3), hi + 100.0)  # absurdly high -> all out of band
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    series = conn.reduce_arrays(
        lats, lons, times, values, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.points[0].value is None
    assert series.points[0].quality.value == "missing"
    assert lo < 0 < hi  # sanity: a mild negative SIF tail is allowed by the band


def test_window_trim_half_open(sif_nc):
    conn = TROPOMISIFConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    # Half-open [2020-06-01, 2020-07-15): includes 06-15, excludes 07-15.
    series = conn.reduce_file(
        sif_nc, spec,
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 15, tzinfo=UTC),
    )
    months = {(p.timestamp.year, p.timestamp.month) for p in series.points}
    assert (2020, 6) in months
    assert (2020, 7) not in months


def test_small_basin_defaults_to_nearest_cell(sif_nc):
    conn = TROPOMISIFConnector()
    spec = ReductionSpec(
        domain_name="tiny", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=500.0,  # small -> nearest_cell
    )
    series = conn.reduce_file(
        sif_nc, spec,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("tropomi_sif:cell:")


@pytest.mark.asyncio
async def test_list_sites_one_reduced_region():
    conn = TROPOMISIFConnector()
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    sites = await conn.list_sites(spec)
    assert len(sites) == 1
    assert sites[0].kind == "reduced_region"
    assert sites[0].site_id == "tropomi_sif:domain:bow"


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = TROPOMISIFConnector()
    spec = ReductionSpec(
        domain_name="x", bbox=(50.0, -116.0, 52.0, -114.0), centroid=(51.0, -115.0),
    )
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_tropomi_sif():
    """LIVE smoke: requires a real Earthdata-downloaded Caltech TROPOMI SIF NetCDF.

    Run with: pytest -m network tests/connectors/test_tropomi_sif.py -k live
    """
    import os

    nc_path = os.environ.get("TROPOMI_SIF_NC")
    if not nc_path:
        pytest.skip("set TROPOMI_SIF_NC to a real Caltech TROPOMI SIF NetCDF")
    conn = TROPOMISIFConnector({"nc_path": nc_path})
    spec = ReductionSpec(
        domain_name="bow", bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0), area_km2=8000.0,
    )
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )
    assert series_list and series_list[0].unit == "mW/m2/nm/sr"
