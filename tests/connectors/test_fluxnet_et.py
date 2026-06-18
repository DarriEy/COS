"""FLUXNET ET connector — hermetic test of the flux-tower / station path.

Builds a synthetic FLUXNET2015 FULLSET CSV and parses it; no network, no auth.
Proves the LE (W/m^2) -> ET (mm/day) canonicalization, QC gating, fill/negative
masking, and half-open UTC window trim that mirror the native SYMFLUENCE handler.
"""

from datetime import UTC, datetime

import pytest

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


# ---------------------------------------------------------------------------
# PARITY-BY-CONSTRUCTION
#
# These tests reimplement the SYMFLUENCE native reduction INLINE (from
# symfluence/data/acquisition/handlers/fluxnet{,_constants}.py) over the SAME
# synthetic CSV the COS parser consumes, then assert COS == native intent.
#
# Native reference semantics (acquisition handler `_process_fluxnet_data` +
# `convert_le_to_et`), so they can be checked without importing SYMFLUENCE:
#   * column pick: first present alias from FLUXNET_VARIABLE_MAPPING priority
#     lists  (LE: LE_F_MDS, LE_PI_F_1_1_1, LE_CORR, LE_1_1_1, LE, LE_F);
#   * LE -> ET: convert_le_to_et = LE / (rho_w * lambda) * 86400 with
#     rho_w=1000, lambda=2.45e6;
#   * QC filter: `result[qc_col] > max_qc -> NaN`, default max_qc=1 (FLUXNET2015
#     0=measured,1=good gap-fill kept; >=2 dropped);
#   * fill: -9999 sentinel -> NaN;  negative ET -> NaN.
#
# KNOWN, DELIBERATE COS DIVERGENCE (documented, benign-with-a-caveat):
#   The native `convert_le_to_et` math yields metres/day (it omits the *1000
#   mm/m factor) while LABELLING the column mm/day -- 283.5 W/m^2 -> ~0.01 not
#   ~10. COS canonical `et` unit is mm/day, so the connector multiplies by 1000
#   (LE_TO_ET_FACTOR ~= 0.03527) to emit the physically-correct mm/day matching
#   its canonical unit and the named-constant derivation (LE_TO_ET_FACTOR=0.0353
#   in fluxnet_constants is the mm/day intent). Hence:
#       COS_mm_per_day == native_convert_le_to_et_value * 1000  (exact).
#   We therefore prove parity against the NATIVE INTENT (the named mm/day factor
#   / derivation), and separately document the literal-function 1000x offset.
# ---------------------------------------------------------------------------

# Native constants, copied verbatim from fluxnet_constants.py.
_NATIVE_WATER_DENSITY = 1000.0
_NATIVE_LATENT_HEAT = 2.45e6
_NATIVE_SECONDS_PER_DAY = 86400.0
# native named constant (mm/day intent, as documented in the module docstring)
_NATIVE_NAMED_FACTOR = 0.0353


def _native_convert_le_to_et(le: float) -> float | None:
    """Verbatim reimplementation of native convert_le_to_et (yields m/day).

    et = le / (rho_w * lambda) * 86400 ; negative -> NaN (returned as None).
    """
    et = (le / (_NATIVE_WATER_DENSITY * _NATIVE_LATENT_HEAT)) * _NATIVE_SECONDS_PER_DAY
    if et < 0:
        return None
    return et


def _native_reduce(rows, max_qc=1):
    """Inline native semantics over (ts, le, qc) rows -> {ts: et_mm_day | None}.

    Mirrors `_process_fluxnet_data`: pick LE, convert, QC>max_qc -> NaN,
    -9999 -> NaN, negative -> NaN. Returns ET in the native-INTENDED mm/day
    (i.e. convert_le_to_et * 1000), the unit the column is labelled with.
    """
    out = {}
    for ts, le, qc in rows:
        if le == -9999.0:
            out[ts] = None
            continue
        if qc is not None and qc > max_qc:
            out[ts] = None
            continue
        et_m_per_day = _native_convert_le_to_et(le)  # m/day (native math)
        out[ts] = None if et_m_per_day is None else et_m_per_day * 1000.0  # -> mm/day intent
    return out


def test_parity_le_to_et_unit_factor_exact_vs_named_intent():
    """COS LE_TO_ET_FACTOR reproduces the native mm/day derivation EXACTLY.

    The native module derives ET = LE/(rho_w*lambda)*86400 [m/day] and labels it
    mm/day (the named constant 0.0353). COS computes the same derivation with the
    mm/m factor explicit. The two derivations are identical to float tolerance.
    """
    cos_derived = _NATIVE_SECONDS_PER_DAY * 1000.0 / (_NATIVE_WATER_DENSITY * _NATIVE_LATENT_HEAT)
    assert pytest.approx(cos_derived, rel=1e-12) == LE_TO_ET_FACTOR
    # within rounding of the published named constant (0.0353).
    assert pytest.approx(_NATIVE_NAMED_FACTOR, abs=1e-4) == LE_TO_ET_FACTOR
    # and is exactly 1000x the literal native convert_le_to_et factor (m/day).
    native_literal = _NATIVE_SECONDS_PER_DAY / (_NATIVE_WATER_DENSITY * _NATIVE_LATENT_HEAT)
    assert pytest.approx(native_literal * 1000.0, rel=1e-12) == LE_TO_ET_FACTOR


def test_parity_cos_equals_native_intent_on_full_csv():
    """COS parser output == inline native reduction (mm/day intent) on same CSV.

    Identity-grade parity: a station network with a pure unit conversion, no
    spatial averaging, so values must agree to float tolerance for every row.
    """
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)

    # The exact rows of MOCK_DAILY inside [start, end): (YYYYMMDD, LE, QC).
    native_rows = [
        ("2020-01-01", 283.5, 0),
        ("2020-01-02", 141.7, 1),
        ("2020-01-03", 200.0, 2),   # QC=2 > max_qc=1 -> MISSING
        ("2020-01-04", -9999.0, 0),  # fill -> MISSING
        ("2020-01-05", -50.0, 0),    # negative ET -> MISSING
    ]
    native = _native_reduce(native_rows, max_qc=1)

    cos_points = FluxnetETConnector.parse_report(MOCK_DAILY, start, end, max_qc=1)
    cos = {p.timestamp.date().isoformat(): p for p in cos_points}

    # Same set of in-window dates (half-open trims 2021-06-01 in both).
    assert set(cos) == set(native)

    for date, native_val in native.items():
        pt = cos[date]
        if native_val is None:
            assert pt.value is None
            assert pt.quality.value == "missing"
        else:
            # EXACT identity (point network, pure unit conversion).
            assert pt.value == pytest.approx(native_val, rel=1e-9)
            assert pt.quality.value == "good"

    # Anchor: 283.5 W/m^2 -> ~10 mm/day, not the native-literal ~0.01 m/day.
    assert cos["2020-01-01"].value == pytest.approx(9.9977, abs=1e-3)


def test_parity_qc_gate_matches_native_threshold():
    """COS QC gate (>max_qc -> MISSING) matches native `result[qc] > max_qc`."""
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)
    rows = [("2020-01-03", 200.0, 2)]
    for max_qc in (1, 2, 3):
        native = _native_reduce(rows, max_qc=max_qc)["2020-01-03"]
        pts = FluxnetETConnector.parse_report(MOCK_DAILY, start, end, max_qc=max_qc)
        cos = {p.timestamp.date().isoformat(): p for p in pts}["2020-01-03"]
        if native is None:
            assert cos.value is None and cos.quality.value == "missing"
        else:
            assert cos.value == pytest.approx(native, rel=1e-9)
            assert cos.quality.value == "good"


def test_parity_fill_and_negative_to_missing_match_native():
    """Fill (-9999) and negative ET both reduce to MISSING in COS and native."""
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 1, 1, tzinfo=UTC)
    native = _native_reduce([("2020-01-04", -9999.0, 0), ("2020-01-05", -50.0, 0)])
    pts = FluxnetETConnector.parse_report(MOCK_DAILY, start, end)
    cos = {p.timestamp.date().isoformat(): p for p in pts}
    for date in ("2020-01-04", "2020-01-05"):
        assert native[date] is None
        assert cos[date].value is None
        assert cos[date].quality.value == "missing"


def test_parity_half_open_window_is_cos_convention_not_native_closed():
    """COS uses half-open [start, end); native pandas slice is closed-closed.

    This is the one deliberate, framework-wide COS divergence (see GRACE/SNOTEL):
    a point exactly at `end` is INCLUDED by native (df.loc[start:end]) but
    EXCLUDED by COS. We document and pin it here so it cannot regress silently.
    """
    # end == 2020-01-04: COS excludes the 01-04 row; a native closed slice keeps it.
    pts = FluxnetETConnector.parse_report(
        MOCK_DAILY, datetime(2020, 1, 2, tzinfo=UTC), datetime(2020, 1, 4, tzinfo=UTC)
    )
    dates = {p.timestamp.date().isoformat() for p in pts}
    assert "2020-01-04" not in dates          # half-open excludes the right edge
    assert dates == {"2020-01-02", "2020-01-03"}


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_fluxnet_et():
    """LIVE smoke placeholder — AmeriFlux pull is keyed and deferred.

    Run with: pytest -m network tests/connectors/test_fluxnet_et.py -k live
    """
    pytest.skip("AmeriFlux live pull is keyed/deferred; parse path is the proven part.")
