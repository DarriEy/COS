"""ASCAT soil-moisture connector — hermetic test of the gridded reduction path.

Builds a synthetic in-memory ASCAT-like NetCDF (degree of saturation) and
reduces it; no network, no auth. This proves the architecture-critical gridded
-> canonical-series path for a C-band active-microwave product: the native
saturation -> volumetric conversion (percentage detection, porosity scaling),
physical-range masking -> MISSING, basin-mean vs nearest-cell reduction, and the
half-open UTC window trim. Canonical unit is m³/m³.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from cos.connectors.ascat_sm import (
    DEFAULT_POROSITY,
    ASCATSoilMoistureConnector,
    _canonicalize_saturation,
)
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction


@pytest.fixture
def ascat_nc(tmp_path):
    """Synthetic ASCAT-like NetCDF: surface_soil_moisture_saturation as percent.

    Four timesteps on a 3x3 grid (0-360 longitudes = -116..-114). Saturation is
    stored as a percentage (0-100), the common CDS representation, so the
    connector must divide by 100 then multiply by porosity. The last timestep is
    entirely NaN (no retrieval) so it must reduce to MISSING; one cell in an
    otherwise-valid layer is out of range to exercise the physical-range mask.
    """
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    times = np.array(
        ["2020-06-15", "2020-06-16", "2020-06-17", "2020-06-18"],
        dtype="datetime64[ns]",
    )
    lats = np.array([50.0, 51.0, 52.0])
    lons = np.array([244.0, 245.0, 246.0])  # 0-360 (= -116..-114)
    data = np.empty((4, 3, 3))
    data[0] = 40.0          # 40% saturation -> 0.40 frac -> *0.45 = 0.18 m³/m³
    data[1] = 80.0          # 80% saturation -> 0.80 frac -> *0.45 = 0.36 m³/m³
    data[1, 0, 0] = 500.0   # absurd value: /100 -> 5.0 -> *0.45 = 2.25 -> masked
    data[2] = 60.0          # 60% -> 0.60 -> *0.45 = 0.27
    data[3] = np.nan        # entirely missing -> MISSING
    ds = xr.Dataset(
        {"surface_soil_moisture_saturation": (("time", "lat", "lon"), data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    path = tmp_path / "ascat_synth.nc"
    ds.to_netcdf(path)
    return path


def _spec(area_km2):
    return ReductionSpec(
        domain_name="bow",
        bbox=(50.0, -116.0, 52.0, -114.0),
        centroid=(51.0, -115.0),
        area_km2=area_km2,
    )


def test_canonicalize_saturation_percent_to_volumetric():
    """Pure helper: 40% saturation -> 0.40 frac -> *0.45 porosity = 0.18 m³/m³."""
    arr = np.full((1, 2, 2), 40.0)
    out = _canonicalize_saturation(arr, "surface_soil_moisture_saturation", DEFAULT_POROSITY)
    assert np.allclose(out, 0.18)


def test_canonicalize_saturation_already_fraction():
    """A saturation fraction (0-1) skips the /100 step but still scales by porosity."""
    arr = np.full((1, 2, 2), 0.40)
    out = _canonicalize_saturation(arr, "surface_soil_moisture_saturation", DEFAULT_POROSITY)
    assert np.allclose(out, 0.18)


def test_canonicalize_masks_out_of_range():
    """Values that fall outside 0 < sm < 1 after conversion become NaN (MISSING)."""
    arr = np.array([[[500.0]]])  # /100 -> 5.0 -> *0.45 = 2.25 -> out of range
    out = _canonicalize_saturation(arr, "saturation", DEFAULT_POROSITY)
    assert np.isnan(out).all()


def test_reduce_file_basin_mean_units_and_values(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),  # large -> basin_mean
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.kind == ObservationKind.SOIL_MOISTURE
    assert series.unit == "m3/m3"  # canonical volumetric, converted from saturation
    assert series.reduction == SpatialReduction.BASIN_MEAN
    assert series.site.kind == "reduced_region"
    assert series.site.site_id == "ascat:domain:bow"

    by_day = {p.timestamp.day: p for p in series.points}
    # 40% saturation -> 0.40 -> *0.45 porosity = 0.18 m³/m³.
    assert by_day[15].value == pytest.approx(0.18, abs=1e-9)
    assert by_day[15].quality == QualityFlag.GOOD
    # 80% layer with one absurd (masked) cell -> remaining cells 0.36 m³/m³.
    assert by_day[16].value == pytest.approx(0.36, abs=1e-9)
    # 60% -> 0.27 m³/m³.
    assert by_day[17].value == pytest.approx(0.27, abs=1e-9)


def test_all_missing_reduces_to_missing(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    assert by_day[18].value is None
    assert by_day[18].quality == QualityFlag.MISSING


def test_porosity_override(ascat_nc):
    """A config porosity overrides the default in the saturation conversion."""
    conn = ASCATSoilMoistureConnector(config={"porosity": 0.50})
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    by_day = {p.timestamp.day: p for p in series.points}
    # 40% -> 0.40 -> *0.50 = 0.20 m³/m³.
    assert by_day[15].value == pytest.approx(0.20, abs=1e-9)


def test_small_basin_defaults_to_nearest_cell(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    series = conn.reduce_file(
        ascat_nc, _spec(500.0),  # small -> nearest_cell
        datetime(2020, 6, 1, tzinfo=UTC), datetime(2020, 7, 1, tzinfo=UTC),
    )
    assert series.reduction == SpatialReduction.NEAREST_CELL
    assert series.site.site_id.startswith("ascat:cell:")


def test_window_trim_half_open(ascat_nc):
    conn = ASCATSoilMoistureConnector()
    # Half-open [06-15, 06-17): includes 06-15 and 06-16, excludes 06-17.
    series = conn.reduce_file(
        ascat_nc, _spec(8000.0),
        datetime(2020, 6, 15, tzinfo=UTC), datetime(2020, 6, 17, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in series.points}
    assert days == {15, 16}


@pytest.mark.asyncio
async def test_fetch_series_without_ncpath_errors():
    conn = ASCATSoilMoistureConnector()
    spec = _spec(8000.0)
    with pytest.raises(Exception, match="NetCDF"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
