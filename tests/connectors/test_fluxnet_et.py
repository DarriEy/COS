"""FLUXNET ET connector — hermetic test of the flux-tower / station path.

Builds a synthetic FLUXNET2015 FULLSET CSV and parses it; no network, no auth.
Proves the LE (W/m^2) -> ET (mm/day) canonicalization, QC gating, fill/negative
masking, and half-open UTC window trim that mirror the native SYMFLUENCE handler.
"""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.fluxnet_et import LE_TO_ET_FACTOR, FluxnetETConnector
from cos.core.exceptions import ConnectorError, DataFormatError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# FLUXNET2015 daily FULLSET: TIMESTAMP (YYYYMMDD), LE_F_MDS (W/m^2), LE_F_MDS_QC.
# LE=283.5 W/m^2 -> ET = 283.5 * 0.03527 ~= 10.0 mm/day.
MOCK_DAILY = """\
# FLUXNET2015 FULLSET DD for US-Ne1
TIMESTAMP,LE_F_MDS,LE_F_MDS_QC,H_F_MDS
20200101,283.5,0,120.0
20200102,141.7,1,90.0
20200103,200.0,2,80.0
20200104,-9999,0,-9999
20200105,-50.0,0,30.0
20210601,100.0,0,50.0
"""


def test_parse_le_to_et_mm_per_day_and_window():
    points = FluxnetETConnector.parse_report(
        MOCK_DAILY,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # 2021-06-01 is outside [start, end) -> excluded entirely.
    assert "2021-06-01" not in by_date
    # LE 283.5 W/m^2 -> ~10 mm/day, QC=0 measured -> GOOD.
    assert by_date["2020-01-01"].value == pytest.approx(283.5 * LE_TO_ET_FACTOR)
    assert by_date["2020-01-01"].value == pytest.approx(10.0, abs=0.05)
    assert by_date["2020-01-01"].quality.value == "good"
    # QC=1 good gap-fill is kept (default max_qc=1).
    assert by_date["2020-01-02"].quality.value == "good"


def test_qc_gate_drops_medium_gapfill():
    points = FluxnetETConnector.parse_report(
        MOCK_DAILY,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # QC=2 (medium) exceeds default max_qc=1 -> MISSING.
    assert by_date["2020-01-03"].value is None
    assert by_date["2020-01-03"].quality.value == "missing"


def test_fill_and_negative_masked_to_missing():
    points = FluxnetETConnector.parse_report(
        MOCK_DAILY,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # -9999 fill sentinel -> MISSING.
    assert by_date["2020-01-04"].value is None
    assert by_date["2020-01-04"].quality.value == "missing"
    # Negative ET (quality artefact) -> MISSING (mirrors convert_le_to_et -> NaN).
    assert by_date["2020-01-05"].value is None
    assert by_date["2020-01-05"].quality.value == "missing"


def test_window_trim_half_open():
    # Half-open [2020-01-02, 2020-01-04): includes 01-02, 01-03; excludes 01-04.
    points = FluxnetETConnector.parse_report(
        MOCK_DAILY,
        datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC),
    )
    dates = {p.timestamp.date().isoformat() for p in points}
    assert dates == {"2020-01-02", "2020-01-03"}


def test_relaxed_qc_keeps_medium():
    points = FluxnetETConnector.parse_report(
        MOCK_DAILY,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        max_qc=2,
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert by_date["2020-01-03"].quality.value == "good"


def test_precomputed_et_column_passthrough():
    csv = "TIMESTAMP,et_mm_day\n20200101,3.5\n20200102,\n"
    points = FluxnetETConnector.parse_report(
        csv, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert by_date["2020-01-01"].value == pytest.approx(3.5)
    assert by_date["2020-01-02"].value is None


def test_missing_et_and_le_columns_raises():
    with pytest.raises(DataFormatError):
        FluxnetETConnector.parse_report(
            "TIMESTAMP,H_F_MDS\n20200101,120.0\n",
            datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


def test_missing_timestamp_column_raises():
    with pytest.raises(DataFormatError):
        FluxnetETConnector.parse_report(
            "LE_F_MDS,LE_F_MDS_QC\n283.5,0\n",
            datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


@pytest.mark.asyncio
async def test_list_sites_strips_namespace():
    conn = FluxnetETConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("US-Ne1", "fluxnet:CA-NS7"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"fluxnet:US-Ne1", "fluxnet:CA-NS7"}
    assert all(s.kind == "station" for s in sites)


@pytest.mark.asyncio
async def test_fetch_series_from_csv_path(tmp_path):
    csv = tmp_path / "FLX_US-Ne1_FLUXNET2015_FULLSET_DD.csv"
    csv.write_text(MOCK_DAILY)
    conn = FluxnetETConnector(config={"path": str(csv), "station": "US-Ne1"})
    spec = ReductionSpec(domain_name="mead", station_ids=("US-Ne1",), centroid=(41.16, -96.47))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.ET
    assert s.unit == "mm/day"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "fluxnet:US-Ne1"
    good = [p for p in s.points if p.value is not None]
    assert len(good) == 2  # 01-01 and 01-02; QC/fill/neg masked


@pytest.mark.asyncio
async def test_fetch_series_without_path_errors():
    conn = FluxnetETConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("US-Ne1",))
    with pytest.raises(ConnectorError, match="FULLSET"):
        await conn.fetch_series(spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_fluxnet_et():
    """LIVE smoke placeholder — AmeriFlux pull is keyed and deferred.

    Run with: pytest -m network tests/connectors/test_fluxnet_et.py -k live
    """
    pytest.skip("AmeriFlux live pull is keyed/deferred; parse path is the proven part.")
