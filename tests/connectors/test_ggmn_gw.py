"""GGMN groundwater connector — hermetic test of the point-network / station path."""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.ggmn_gw import GGMNGroundwaterConnector
from cos.core.exceptions import DataFormatError
from cos.core.models import ObservationKind, ReductionSpec, SpatialReduction

# GGMN WellLevelMeasurement/list payload: each row carries an HTML fragment with
# hidden inputs name="time" (ISO timestamp) and name="value_value" (metres).
def _row(time: str, value: str) -> dict:
    return {
        "html": (
            f'<form><input type="hidden" name="time" value="{time}">'
            f'<input type="hidden" name="value_value" value="{value}"></form>'
        )
    }


MOCK_PAYLOAD = {
    "data": [
        _row("2020-01-01T00:00:00", "12.5"),
        _row("2020-06-15T00:00:00", "13.0"),
        _row("2020-03-01T00:00:00", ""),       # empty -> MISSING
        _row("2020-04-01T00:00:00", "n/a"),    # non-numeric -> MISSING
        _row("2021-06-01T00:00:00", "9.0"),    # outside [start, end)
    ]
}


def test_parse_measurements_units_window_and_quality():
    points = GGMNGroundwaterConnector.parse_measurements(
        MOCK_PAYLOAD,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # window-trim: 2021-06-01 dropped (half-open [start, end))
    assert "2021-06-01" not in by_date
    # units: GGMN metres == canonical groundwater metres (identity, no scaling)
    assert by_date["2020-01-01"].value == pytest.approx(12.5)
    assert by_date["2020-01-01"].quality.value == "good"
    assert by_date["2020-06-15"].value == pytest.approx(13.0)
    # fill / non-numeric -> MISSING
    assert by_date["2020-03-01"].value is None
    assert by_date["2020-03-01"].quality.value == "missing"
    assert by_date["2020-04-01"].value is None
    assert by_date["2020-04-01"].quality.value == "missing"
    # sorted ascending
    assert [p.timestamp for p in points] == sorted(p.timestamp for p in points)


def test_parse_measurements_empty_data():
    assert GGMNGroundwaterConnector.parse_measurements(
        {"data": []}, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    ) == []


def test_parse_measurements_non_mapping_raises():
    with pytest.raises(DataFormatError):
        GGMNGroundwaterConnector.parse_measurements(
            [], datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


def test_z_suffix_timestamp_parsed_as_utc():
    payload = {"data": [_row("2020-02-02T06:00:00Z", "5.0")]}
    points = GGMNGroundwaterConnector.parse_measurements(
        payload, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert len(points) == 1
    assert points[0].timestamp == datetime(2020, 2, 2, 6, 0, tzinfo=UTC)
    assert points[0].value == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = GGMNGroundwaterConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("4242", "ggmn_gw:4343"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"ggmn_gw:4242", "ggmn_gw:4343"}
    assert all(s.kind == "station" for s in sites)


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://ggis\.un-igrac\.org/groundwater/record/.*").mock(
        return_value=httpx.Response(200, json=MOCK_PAYLOAD)
    )
    conn = GGMNGroundwaterConnector()
    spec = ReductionSpec(domain_name="aquifer", station_ids=("4242",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.GROUNDWATER
    assert s.unit == "m"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "ggmn_gw:4242"
    # two GOOD points in-window (12.5, 13.0), two MISSING, one trimmed
    assert len([p for p in s.points if p.value is not None]) == 2


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION                                                        #
# --------------------------------------------------------------------------- #
# Native handler: symfluence/data/observation/handlers/ggmn.py
#
# Native value pipeline (acquire): for each measurement row it parses the HTML
# with BeautifulSoup, reads input[name=time] and input[name=value_value],
# builds a DataFrame, then:
#     df['groundwater_level'] = pd.to_numeric(values, errors='coerce')
#     df = df.dropna()                      # bad/empty values are DROPPED
# There is NO unit scaling — GGMN reports metres and the native column stays in
# metres (identity m -> m). The native handler applies NO time-window filter at
# acquire; cross-station daily-mean aggregation happens later in process().
#
# COS is the per-station (point_network / STATION) boundary: parse_measurements
# does the SAME identity m->m extraction, but (a) trims to half-open [start,end)
# UTC and (b) keeps bad/empty rows as QualityFlag.MISSING instead of dropping
# them. The parity-relevant invariant is therefore: the GOOD (finite) points COS
# emits for a window must equal, value-for-value and timestamp-for-timestamp,
# the native dropna'd rows restricted to that same window. Because both sides are
# identity unit conversions of the same scalars, agreement is EXACT (float
# identity), not tolerance-based — this is a point network, not a cos-lat
# basin-mean.


def _native_rows(payload: dict) -> dict[str, float]:
    """Reimplement the native ggmn.py value pipeline inline.

    Mirrors GGMNHandler.acquire: BeautifulSoup-parse each row's html, read the
    `time`/`value_value` inputs, coerce values to numeric (errors='coerce'),
    then drop NaN rows. Returns {iso_timestamp: value_metres} for the survivors
    (native keeps no MISSING placeholder — coerced NaNs are dropped).
    """
    import pandas as pd
    from bs4 import BeautifulSoup

    times: list[str] = []
    values: list[str] = []
    for item in payload.get("data", []):
        html = (item or {}).get("html", "")
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        t_in = soup.find("input", {"name": "time"})
        v_in = soup.find("input", {"name": "value_value"})
        if t_in and v_in:
            times.append(t_in.get("value"))
            values.append(v_in.get("value"))
    df = pd.DataFrame({"datetime": times, "groundwater_level": values})
    # identity unit (no scaling), exactly as native (no *factor)
    df["groundwater_level"] = pd.to_numeric(df["groundwater_level"], errors="coerce")
    df = df.dropna()
    return {t: float(v) for t, v in zip(df["datetime"], df["groundwater_level"])}


def test_parity_cos_good_points_equal_native_dropna_identity():
    """COS GOOD points == native dropna'd rows (same window), EXACT.

    Identity unit (m -> m) means there is no factor to lose; the two pipelines
    must agree to float identity, not just a tolerance.
    """
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)

    cos_points = GGMNGroundwaterConnector.parse_measurements(MOCK_PAYLOAD, start, end)
    native = _native_rows(MOCK_PAYLOAD)

    # COS GOOD points, keyed by the same naive-ISO timestamp the native CSV uses.
    cos_good = {
        p.timestamp.strftime("%Y-%m-%dT%H:%M:%S"): p.value
        for p in cos_points
        if p.value is not None
    }
    # Native has no window concept; restrict it to COS's half-open [start, end)
    # so we compare the SAME observations. (2021-06-01 is native-kept but
    # out-of-window for COS — that trimming is COS's documented contract, not a
    # value divergence.)
    native_in_window = {
        t: v
        for t, v in native.items()
        if start <= datetime.fromisoformat(t).replace(tzinfo=UTC) < end
    }

    assert cos_good == native_in_window  # EXACT float identity, no tolerance
    # And the bad rows native DROPPED are exactly the rows COS marks MISSING.
    cos_missing = {
        p.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        for p in cos_points
        if p.value is None
    }
    assert cos_missing == {"2020-03-01T00:00:00", "2020-04-01T00:00:00"}
    assert all(t not in native for t in cos_missing)  # native dropped them


def test_parity_unit_factor_is_identity():
    """A raw metres value must pass through unchanged (factor == 1.0).

    A wrong unit factor (e.g. cm->m or ft->m) would be the classic port bug; pin
    it to identity against a synthetic single value.
    """
    raw_m = 17.3456
    payload = {"data": [_row("2020-05-05T12:00:00", str(raw_m))]}
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)

    cos = GGMNGroundwaterConnector.parse_measurements(payload, start, end)
    native = _native_rows(payload)

    assert len(cos) == 1
    assert cos[0].value == pytest.approx(raw_m, rel=0, abs=0)  # exact identity
    assert next(iter(native.values())) == pytest.approx(raw_m, rel=0, abs=0)
    assert cos[0].value == next(iter(native.values()))


def test_parity_half_open_window_trim():
    """[start, end): the start instant is kept, the end instant is excluded.

    Native applies no window, so this is a COS-only contract — we assert the
    boundary behaviour directly and confirm the values inside the window still
    match native exactly.
    """
    payload = {
        "data": [
            _row("2020-01-01T00:00:00", "1.0"),   # == start -> kept
            _row("2020-06-01T00:00:00", "2.0"),   # interior -> kept
            _row("2021-01-01T00:00:00", "3.0"),   # == end -> excluded (half-open)
        ]
    }
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)
    cos = GGMNGroundwaterConnector.parse_measurements(payload, start, end)
    kept = {p.timestamp.strftime("%Y-%m-%dT%H:%M:%S"): p.value for p in cos}
    assert kept == {"2020-01-01T00:00:00": 1.0, "2020-06-01T00:00:00": 2.0}
    # the kept values equal native's identity values for the same rows
    native = _native_rows(payload)
    assert all(kept[t] == native[t] for t in kept)


def test_parity_fill_and_missing_to_quality_flag():
    """Empty and non-numeric values -> QualityFlag.MISSING (and native drops).

    Native coerces both to NaN and drops; COS keeps them as MISSING placeholders.
    The finite set is identical between the two — that is the parity that matters.
    """
    payload = {
        "data": [
            _row("2020-02-01T00:00:00", "4.5"),
            _row("2020-02-02T00:00:00", ""),       # empty
            _row("2020-02-03T00:00:00", "NaN"),    # pandas-coercible NaN token
            _row("2020-02-04T00:00:00", "blah"),   # non-numeric
        ]
    }
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)
    cos = GGMNGroundwaterConnector.parse_measurements(payload, start, end)
    by_date = {p.timestamp.strftime("%Y-%m-%dT%H:%M:%S"): p for p in cos}
    assert by_date["2020-02-01T00:00:00"].quality.value == "good"
    for t in ("2020-02-02T00:00:00", "2020-02-03T00:00:00", "2020-02-04T00:00:00"):
        assert by_date[t].value is None
        assert by_date[t].quality.value == "missing"
    # native: only the finite 4.5 survives dropna -> COS GOOD set matches.
    native = _native_rows(payload)
    cos_good = {t: p.value for t, p in by_date.items() if p.value is not None}
    assert cos_good == native == {"2020-02-01T00:00:00": 4.5}


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_ggmn():
    """LIVE smoke against the real anonymous IGRAC GGMN endpoint.

    Run with: pytest -m network tests/connectors/test_ggmn_gw.py -k live
    """
    conn = GGMNGroundwaterConnector()
    # A small bbox somewhere with known GGMN coverage; discovery-driven.
    spec = ReductionSpec(domain_name="nl", bbox=(51.0, 4.0, 53.0, 7.0))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2010, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m"
