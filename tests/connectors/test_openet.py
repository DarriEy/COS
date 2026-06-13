"""OpenET connector — hermetic test of the ensemble / flux-tower path."""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.openet import OpenETConnector
from cos.core.exceptions import AuthRequiredError, DataFormatError
from cos.core.models import ObservationKind, ReductionSpec

# OpenET monthly timeseries (mm per month).
MOCK_TS = [
    {"time": "2020-01-01", "et": 31.0},   # Jan: 31 days -> 1.0 mm/day
    {"time": "2020-02-01", "et": 29.0},   # Feb 2020: 29 days -> 1.0 mm/day
    {"time": "2020-03-01", "et": None},   # missing
]


def test_parse_monthly_to_mm_per_day():
    points = OpenETConnector.parse_timeseries(MOCK_TS, "monthly")
    by_month = {p.timestamp.month: p for p in points}
    assert by_month[1].value == pytest.approx(1.0)
    assert by_month[2].value == pytest.approx(1.0)
    assert by_month[3].value is None
    assert by_month[3].quality.value == "missing"


def test_parse_daily_passthrough():
    rows = [{"time": "2020-01-01", "et": 2.5}]
    pts = OpenETConnector.parse_timeseries(rows, "daily")
    assert pts[0].value == pytest.approx(2.5)


def test_parse_bad_payload_raises():
    with pytest.raises(DataFormatError):
        OpenETConnector.parse_timeseries({"not": "a list"}, "monthly")


def test_token_required():
    conn = OpenETConnector()
    with pytest.raises(AuthRequiredError):
        conn._token()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_ensemble_default():
    respx.post(url__regex=r"https://openet-api\.org/.*").mock(
        return_value=httpx.Response(200, json=MOCK_TS)
    )
    conn = OpenETConnector(config={"token": "fake-key", "interval": "monthly"})
    spec = ReductionSpec(domain_name="ca_field", centroid=(38.5, -121.5))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 4, 1, tzinfo=UTC)
        )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.ET
    assert s.unit == "mm/day"
    assert s.site.extra["model"] == "ensemble"
    assert s.site.site_id.endswith(":ensemble")


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_multi_model():
    respx.post(url__regex=r"https://openet-api\.org/.*").mock(
        return_value=httpx.Response(200, json=MOCK_TS)
    )
    conn = OpenETConnector(config={"token": "k", "interval": "monthly"})
    spec = ReductionSpec(domain_name="x", centroid=(38.5, -121.5), options={"models": ["ssebop", "ensemble"]})
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 4, 1, tzinfo=UTC)
        )
    assert {s.site.extra["model"] for s in series_list} == {"ssebop", "ensemble"}
