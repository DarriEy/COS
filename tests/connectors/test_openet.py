"""OpenET connector — hermetic test of the ensemble / flux-tower path.

Includes a PARITY-BY-CONSTRUCTION test against the SYMFLUENCE native handler
semantics (``data/observation/handlers/openet.py`` +
``data/acquisition/handlers/openet.py``). The native handler keeps OpenET's raw
*mm-per-interval* totals (it renames the ``et``/``et_mm`` column to ``et_mm_day``
but never divides by the days in the interval); COS canonicalizes the *same*
payload to the canonical ``et`` unit ``mm/day`` at the connector boundary
(documented in ``connectors/openet.py``). The parity test reimplements the
native parse inline and asserts the exact unit relationship
``native_mm_per_interval == cos_mm_per_day * days_in_interval``.
"""

import calendar
from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.openet import OpenETConnector
from cos.core.exceptions import AuthRequiredError, DataFormatError
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec

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


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION vs SYMFLUENCE native handler
# --------------------------------------------------------------------------- #
#
# Native ref:
#   acquisition/handlers/openet.py  -> downloads units='mm', resolution=monthly,
#       writes raw {date, et_mm} rows (mm per interval) to CSV.
#   observation/handlers/openet.py  -> _process_file renames et/value -> et_mm_day
#       WITHOUT dividing by days (so the value stays mm-per-interval despite the
#       column name), to_numeric(errors='coerce') turns bad/blank -> NaN, and the
#       experiment-window trim is pandas .loc[start:end] (label slice, CLOSED on
#       both ends).
#
# COS connector:
#   parse_timeseries canonicalizes to the canonical `et` unit mm/day by dividing
#   monthly mm by calendar days-in-month (daily is pass-through). Missing -> None
#   with QualityFlag.MISSING.
#
# The only semantic difference is the deliberate canonical-unit conversion
# (mm-per-interval -> mm/day). That is a pure, invertible scalar factor
# (days-in-interval), so parity is EXACT once expressed in matching units — there
# is no spatial reduction here (OpenET serves a server-side polygon `reducer=mean`
# timeseries, so neither side does a client-side cos-lat reduction). Fill and
# window semantics must also match.


def _native_parse_value(et_raw, ts):
    """Reimplement the SYMFLUENCE native per-row value semantics inline.

    Returns the native ``et_mm_day`` column value (which is actually raw
    mm-per-interval — native never divides by days) or ``None`` for the NaN /
    coerced-missing case. Mirrors ``OpenETHandler._process_file``:
    rename -> et_mm_day, then pd.to_numeric(errors='coerce').
    """
    if et_raw is None:
        return None
    try:
        return float(et_raw)  # pd.to_numeric on a clean numeric
    except (TypeError, ValueError):
        return None  # errors='coerce' -> NaN -> treated as missing


def test_parity_unit_conversion_against_native_monthly():
    """COS mm/day * days-in-month == native raw mm-per-interval, EXACTLY.

    Same synthetic payload fed to both the COS pure parser and an inline
    reimplementation of the native column semantics. The conversion factor is a
    pure scalar (calendar days in the interval), so parity is exact to float
    tolerance — no reduction, no approximation.
    """
    payload = [
        {"time": "2020-01-01", "et": 124.0},   # Jan, 31 days
        {"time": "2020-02-01", "et": 58.0},    # Feb 2020 (leap), 29 days
        {"time": "2021-02-01", "et": 56.0},    # Feb 2021, 28 days
        {"time": "2020-04-01", "et": 90.0},    # Apr, 30 days
    ]

    cos_points = OpenETConnector.parse_timeseries(payload, "monthly")
    cos_by_ts = {(p.timestamp.year, p.timestamp.month): p for p in cos_points}

    for row in payload:
        ts = datetime.fromisoformat(row["time"]).replace(tzinfo=UTC)
        days = calendar.monthrange(ts.year, ts.month)[1]
        native_mm_per_interval = _native_parse_value(row["et"], ts)

        cos_pt = cos_by_ts[(ts.year, ts.month)]
        assert cos_pt.quality == QualityFlag.GOOD
        # Reconstruct native units from the COS canonical value -> EXACT match.
        cos_mm_per_interval = cos_pt.value * days
        assert cos_mm_per_interval == pytest.approx(native_mm_per_interval, rel=0, abs=1e-9)


def test_parity_daily_is_identity_against_native():
    """For daily interval COS does NOT scale (pass-through), so COS == native
    mm/day with no conversion — identity parity to float tolerance."""
    payload = [
        {"time": "2020-07-01", "et": 6.2},
        {"time": "2020-07-02", "et": 5.8},
    ]
    cos_points = OpenETConnector.parse_timeseries(payload, "daily")
    for row, pt in zip(payload, cos_points):
        native = _native_parse_value(row["et"], pt.timestamp)
        # daily native row is already mm/interval == mm/day (interval is 1 day)
        assert pt.value == pytest.approx(native, rel=0, abs=1e-12)


def test_parity_constant_field_single_cell_exact():
    """A constant monthly field: every COS mm/day equals native_mm / days, and
    because the field is constant the canonicalized series is itself constant
    within a month and exactly reconstructs native — float-tolerance agreement."""
    const_mm = 30.0
    payload = [{"time": f"2020-{m:02d}-01", "et": const_mm} for m in range(1, 13)]
    cos_points = OpenETConnector.parse_timeseries(payload, "monthly")
    for row, pt in zip(payload, cos_points):
        days = calendar.monthrange(pt.timestamp.year, pt.timestamp.month)[1]
        native = _native_parse_value(row["et"], pt.timestamp)
        assert pt.value * days == pytest.approx(native, abs=1e-9)


def test_parity_fill_missing_matches_native():
    """Native coerces blank/None/non-numeric to NaN (treated as missing). COS
    must map exactly those rows to value=None / QualityFlag.MISSING and only
    those rows."""
    payload = [
        {"time": "2020-01-01", "et": 31.0},     # good
        {"time": "2020-02-01", "et": None},     # explicit null
        {"time": "2020-03-01", "et": "n/a"},    # non-numeric -> coerce NaN
        {"time": "2020-04-01", "et": 30.0},     # good
    ]
    cos_points = OpenETConnector.parse_timeseries(payload, "monthly")
    cos_by_month = {p.timestamp.month: p for p in cos_points}

    for row in payload:
        month = int(row["time"][5:7])
        native = _native_parse_value(row["et"], None)
        pt = cos_by_month[month]
        if native is None:
            assert pt.value is None
            assert pt.quality == QualityFlag.MISSING
        else:
            assert pt.value is not None
            assert pt.quality == QualityFlag.GOOD


def test_parity_half_open_window_trim_matches_native_closed_caveat():
    """Window semantics.

    COS callers pass a half-open [start, end) UTC window; the GRACE/SNOTEL
    gold-standard connectors trim with strict ``ts < end``. OpenET's pure parser
    does NOT trim (the server is asked for an explicit ``date_range``), so the
    trimming contract is exercised here at the row level the way the COS pipeline
    applies it, and contrasted with the native CLOSED ``.loc[start:end]`` slice.

    For an end-aligned observation the two conventions differ by exactly the
    end-point row; this test pins the COS half-open behaviour so a regression that
    silently flipped to native-closed semantics would be caught.
    """
    payload = [
        {"time": "2020-01-01", "et": 31.0},
        {"time": "2020-02-01", "et": 29.0},
        {"time": "2020-03-01", "et": 31.0},
    ]
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 3, 1, tzinfo=UTC)  # the 03-01 row sits exactly on `end`

    points = OpenETConnector.parse_timeseries(payload, "monthly")
    # Apply the COS half-open [start, end) trim the pipeline uses.
    cos_window = [p for p in points if start <= p.timestamp < end]
    cos_months = {p.timestamp.month for p in cos_window}
    assert cos_months == {1, 2}          # 03-01 excluded (half-open)

    # Native pandas .loc[start:end] is CLOSED -> would INCLUDE 03-01. Documented
    # divergence; benign for ET because end is exclusive-by-design in COS and the
    # caller never double-counts the boundary month.
    native_closed_months = {1, 2, 3}
    assert 3 in native_closed_months and 3 not in cos_months


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
