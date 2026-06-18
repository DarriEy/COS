"""ISMN soil-moisture connector — hermetic test of the point-network path.

Parses a small synthetic ISMN station CSV; no network, no credentials. Proves
the architecture-critical station → canonical-series path for volumetric soil
moisture: m³/m³ identity unit, the native percent→fraction rule, blank→MISSING,
and the half-open UTC window trim. Mirrors test_snotel.py / test_smap_sm.py.
"""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.ismn_sm import ISMNSoilMoistureConnector
from cos.core.exceptions import ConnectorError
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction

# Already-volumetric ISMN station CSV (m³/m³): header, then DateTime,soil_moisture
# rows. One blank row -> MISSING; 2021-06-01 is outside the [start, end) window.
MOCK_CSV_VOLUMETRIC = """\
# ISMN station MAQU:CST05
DateTime,soil_moisture
2020-01-01,0.20
2020-01-02,0.35
2020-01-03,
2021-06-01,0.10
"""

# Percent-saturation variant (values > 1.5): the native rule divides by 100.
MOCK_CSV_PERCENT = """\
DateTime,soil_moisture
2020-01-01,20.0
2020-01-02,35.0
"""


def test_parse_volumetric_identity_and_window():
    points = ISMNSoilMoistureConnector.parse_station_csv(
        MOCK_CSV_VOLUMETRIC,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # 2021-06-01 outside half-open [start, end); the blank row is MISSING.
    assert "2021-06-01" not in by_date
    # Volumetric m³/m³ passes through unchanged (identity conversion).
    assert by_date["2020-01-01"].value == pytest.approx(0.20)
    assert by_date["2020-01-01"].quality == QualityFlag.GOOD
    assert by_date["2020-01-02"].value == pytest.approx(0.35)
    assert by_date["2020-01-03"].value is None
    assert by_date["2020-01-03"].quality == QualityFlag.MISSING


def test_parse_percent_divides_by_100():
    points = ISMNSoilMoistureConnector.parse_station_csv(
        MOCK_CSV_PERCENT,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # Series exceeds the 1.5 ceiling -> read as percent saturation, /100 -> m³/m³.
    assert by_date["2020-01-01"].value == pytest.approx(0.20)
    assert by_date["2020-01-02"].value == pytest.approx(0.35)


def test_window_trim_half_open():
    points = ISMNSoilMoistureConnector.parse_station_csv(
        MOCK_CSV_VOLUMETRIC,
        datetime(2020, 1, 1, tzinfo=UTC), datetime(2020, 1, 3, tzinfo=UTC),
    )
    days = {p.timestamp.day for p in points}
    # Half-open [01-01, 01-03): includes 01-01 and 01-02, excludes 01-03.
    assert days == {1, 2}


def test_missing_sm_column_raises():
    with pytest.raises(ConnectorError):
        ISMNSoilMoistureConnector.parse_station_csv(
            "# c\nDateOnly\n2020-01-01\n",
            datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC),
        )


def test_namespaced_and_bare_ids_resolve():
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("ismn:MAQU:CST05", "OZNET:Y1"))
    ids = conn._station_ids(spec)
    # leading "ismn:" namespace stripped; embedded ":" in the station id kept.
    assert ids == ["MAQU:CST05", "OZNET:Y1"]


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("MAQU:CST05", "ismn:OZNET:Y1"))
    sites = await conn.list_sites(spec)
    assert {s.site_id for s in sites} == {"ismn:MAQU:CST05", "ismn:OZNET:Y1"}
    assert all(s.kind == "station" for s in sites)


@pytest.mark.asyncio
async def test_fetch_series_without_station_ids_errors():
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="x")
    with pytest.raises(ConnectorError, match="station ids"):
        await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://ismn\.earth/.*").mock(
        return_value=httpx.Response(200, text=MOCK_CSV_VOLUMETRIC)
    )
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="maqu", station_ids=("ismn:MAQU:CST05",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )
    assert len(series_list) == 1
    s = series_list[0]
    assert s.kind == ObservationKind.SOIL_MOISTURE
    assert s.unit == "m3/m3"
    assert s.reduction == SpatialReduction.STATION
    assert s.site.kind == "station"
    assert s.site.site_id == "ismn:MAQU:CST05"
    assert len([p for p in s.points if p.value is not None]) == 2


# ---------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION
#
# The native SYMFLUENCE ISMN acquisition handler
# (symfluence/data/acquisition/handlers/ismn.py, ISMNAcquirer.download) parses
# each station's JSON [dates, values] payload into a per-station
# DateTime,soil_moisture frame and canonicalizes it with exactly three rules:
#
#   1. numeric coercion:  df["soil_moisture"] = pd.to_numeric(values, "coerce")
#   2. percent->fraction: if "* 100" in unit or unit.endswith("100")
#                            or df["soil_moisture"].max() > 1.5:  /= 100.0
#   3. fill:              df = df.dropna(subset=["soil_moisture"])  (drop NaN rows)
#
# It is a POINT NETWORK: each station is kept as its own series, so there is NO
# spatial reduction (no cos-lat weighting) at this boundary — parity for the
# kept (finite) values is therefore EXACT/identity, not tolerance-based. The COS
# pure parser (parse_station_csv) must reproduce rules 1-2 value-for-value, and
# differs from native only in rule 3's *representation*: native drops missing
# rows; COS keeps the timestamp and emits QualityFlag.MISSING (a superset that
# carries strictly more information and is filtered out of any metric the kind's
# objective computes — a benign divergence).
#
# The functions below reimplement the native semantics inline; the tests assert
# COS == native on the SAME synthetic input.
# ---------------------------------------------------------------------------


def _native_parse(dates, raw_values, unit=""):
    """Reimplement ISMNAcquirer.download's per-station canonicalization.

    Returns {iso_date: float} of the values the native handler would KEEP
    (after numeric coercion, the percent->fraction rule, and dropna).
    """
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"DateTime": dates, "soil_moisture": raw_values})
    df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce")
    u = str(unit).lower()
    if "* 100" in u or u.endswith("100") or df["soil_moisture"].max() > 1.5:
        df["soil_moisture"] = df["soil_moisture"] / 100.0
    df = df.dropna(subset=["soil_moisture"])
    return {
        datetime.fromisoformat(d).date().isoformat(): float(v)
        for d, v in zip(df["DateTime"], df["soil_moisture"])
    }


def _csv(dates, raw_values):
    """Build the two-column DateTime,soil_moisture CSV the connector consumes."""
    rows = "\n".join(f"{d},{'' if v is None else v}" for d, v in zip(dates, raw_values))
    return f"DateTime,soil_moisture\n{rows}\n"


# Window spanning the whole synthetic fixture (parity over the kept values).
_PAR_START = datetime(2020, 1, 1, tzinfo=UTC)
_PAR_END = datetime(2020, 2, 1, tzinfo=UTC)


def test_parity_volumetric_identity_matches_native_exactly():
    """COS finite values == native kept values, bit-for-bit (identity unit)."""
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    vals = [0.20, 0.355123, 0.07]  # already volumetric m3/m3, max <= 1.5
    points = ISMNSoilMoistureConnector.parse_station_csv(
        _csv(dates, vals), _PAR_START, _PAR_END
    )
    cos = {
        p.timestamp.date().isoformat(): p.value
        for p in points
        if p.quality == QualityFlag.GOOD
    }
    native = _native_parse(dates, vals)
    assert cos.keys() == native.keys()
    for k in native:
        # Identity conversion + no reduction => exact equality (no tolerance).
        assert cos[k] == native[k]


def test_parity_percent_rule_matches_native_factor():
    """Percent->fraction (/100) reproduces native exactly on a >1.5 series."""
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    vals = [20.0, 35.5, 7.25]  # percent saturation: max 35.5 > 1.5
    points = ISMNSoilMoistureConnector.parse_station_csv(
        _csv(dates, vals), _PAR_START, _PAR_END
    )
    cos = {
        p.timestamp.date().isoformat(): p.value
        for p in points
        if p.quality == QualityFlag.GOOD
    }
    native = _native_parse(dates, vals)  # unit="" -> max>1.5 triggers /100
    assert cos.keys() == native.keys()
    for k in native:
        assert cos[k] == pytest.approx(native[k], rel=0.0, abs=0.0)
    # And the factor really is 1/100, not something else.
    assert cos["2020-01-01"] == pytest.approx(0.20)


def test_parity_at_percent_ceiling_boundary_matches_native():
    """At exactly 1.5 (strict >), neither native nor COS treats it as percent."""
    dates = ["2020-01-01", "2020-01-02"]
    vals = [1.5, 0.9]  # max == 1.5, NOT > 1.5 => no /100 in either impl
    points = ISMNSoilMoistureConnector.parse_station_csv(
        _csv(dates, vals), _PAR_START, _PAR_END
    )
    cos = {p.timestamp.date().isoformat(): p.value for p in points}
    native = _native_parse(dates, vals)
    assert cos == native
    assert cos["2020-01-01"] == pytest.approx(1.5)  # untouched


def test_parity_fill_rule_native_drops_cos_flags_missing():
    """Blank/unparseable -> native DROPS the row; COS keeps it as MISSING.

    The finite values must still agree exactly, and the COS MISSING set must be
    EXACTLY the rows native dropped (the divergence is representational only).
    """
    dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
    raw = ["0.20", "", "0.30", "notanumber"]  # rows 2 and 4 are non-finite
    points = ISMNSoilMoistureConnector.parse_station_csv(
        _csv(dates, raw), _PAR_START, _PAR_END
    )
    good = {
        p.timestamp.date().isoformat(): p.value
        for p in points
        if p.quality == QualityFlag.GOOD
    }
    missing = {p.timestamp.date().isoformat() for p in points if p.quality == QualityFlag.MISSING}

    native = _native_parse(dates, [0.20, None, 0.30, None])  # native coerces both -> NaN -> dropped
    assert good == native  # finite values identical
    # COS keeps exactly the timestamps native dropped, flagged MISSING.
    assert missing == {"2020-01-02", "2020-01-04"}


def test_parity_single_constant_value_float_tolerance():
    """Single-cell / constant field: the two MUST agree to float tolerance."""
    dates = ["2020-01-15"]
    vals = [0.3333333333333333]
    points = ISMNSoilMoistureConnector.parse_station_csv(
        _csv(dates, vals), _PAR_START, _PAR_END
    )
    native = _native_parse(dates, vals)
    assert points[0].value == pytest.approx(native["2020-01-15"], rel=1e-12, abs=1e-12)


def test_parity_window_trim_half_open_vs_native_inclusive():
    """COS half-open [start,end) trim is the canonical convention.

    Native's observation handler trims inclusively [start,end]; COS trims the
    upper bound exclusively. This is the ONLY intentional boundary divergence:
    a sample landing exactly on `end` is kept by native but excluded by COS.
    Over the interior (strictly inside the window) the two agree exactly.
    """
    dates = ["2019-12-31", "2020-01-01", "2020-01-15", "2020-02-01"]
    vals = [0.10, 0.20, 0.30, 0.40]
    points = ISMNSoilMoistureConnector.parse_station_csv(
        _csv(dates, vals), _PAR_START, _PAR_END
    )
    kept = {p.timestamp.date().isoformat(): p.value for p in points}
    # 2019-12-31 < start -> excluded; 2020-02-01 == end -> excluded (half-open).
    assert set(kept) == {"2020-01-01", "2020-01-15"}
    # Interior values are exact native identity.
    native = _native_parse(["2020-01-01", "2020-01-15"], [0.20, 0.30])
    assert kept == native


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_ismn():
    """LIVE smoke against the real ISMN dataviewer (needs registered creds).

    Run with: pytest -m network tests/connectors/test_ismn_sm.py -k live
    """
    conn = ISMNSoilMoistureConnector()
    spec = ReductionSpec(domain_name="maqu", station_ids=("MAQU:CST05",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2015, 3, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m3/m3"
