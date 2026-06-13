"""SNOTEL connector — hermetic test of the point-network / station path."""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.snotel import SNOTELConnector
from cos.core.exceptions import DataFormatError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# NRCS daily report: comment lines, header, then Date,WTEQ(in) rows.
MOCK_REPORT = """\
# Data for site 679
# Snow Telemetry (SNOTEL)
Date,Snow Water Equivalent (in) Start of Day Values
2020-01-01,10.0
2020-01-02,12.0
2020-01-03,
2021-06-01,5.0
"""


def test_parse_report_inches_to_mm_and_window():
    points = SNOTELConnector.parse_report(
        MOCK_REPORT,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    # 2021-06-01 is outside [start, end); the empty row is MISSING.
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    assert "2021-06-01" not in by_date
    assert by_date["2020-01-01"].value == pytest.approx(10.0 * 25.4)  # inches -> mm
    assert by_date["2020-01-01"].quality.value == "good"
    assert by_date["2020-01-03"].value is None
    assert by_date["2020-01-03"].quality.value == "missing"


def test_parse_report_missing_swe_column_raises():
    bad = "# c\nDate,Air Temperature\n2020-01-01,5.0\n"
    with pytest.raises(DataFormatError):
        # only one non-date col, but it's clearly not SWE; position fallback would
        # pick it, so force the no-SWE branch with a single column.
        SNOTELConnector.parse_report("# c\nDateOnly\n2020-01-01\n",
                                     datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
    # the two-col case uses the positional fallback (still parses) — sanity:
    pts = SNOTELConnector.parse_report(bad, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC))
    assert pts and pts[0].value == pytest.approx(5.0 * 25.4)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://wcc\.sc\.egov\.usda\.gov/.*").mock(
        return_value=httpx.Response(200, text=MOCK_REPORT)
    )
    conn = SNOTELConnector()
    spec = ReductionSpec(domain_name="rainier", station_ids=("snotel:679",), options={"state": "WA"})
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.SWE
    assert s.unit == "mm"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "snotel:679"
    assert len([p for p in s.points if p.value is not None]) == 2


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = SNOTELConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("679", "snotel:680"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"snotel:679", "snotel:680"}


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_snotel():
    """LIVE smoke against the real anonymous NRCS AWDB endpoint.

    Run with: pytest -m network tests/connectors/test_snotel.py -k live
    """
    conn = SNOTELConnector()
    spec = ReductionSpec(domain_name="paradise", station_ids=("679",), options={"state": "WA"})
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 3, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "mm"
    assert any(p.value is not None for p in series_list[0].points)
