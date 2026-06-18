"""USGS NWIS groundwater connector — hermetic test of the point/station path."""

import json
from datetime import UTC, datetime

import httpx
import pytest
import respx

from cos.connectors.usgs_gw import (
    FEET_TO_METERS,
    GW_PARAM_CODE,
    NWIS_NODATA,
    USGSGroundwaterConnector,
)
from cos.core.exceptions import DataFormatError
from cos.core.models import ObservationKind, QualityFlag, ReductionSpec, SpatialReduction

# Minimal NWIS waterservices JSON: one 72019 (feet) series with a good value,
# a blank, the -999999 fill, and an out-of-window row.
MOCK_NWIS = json.dumps(
    {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "parameterCode": "72019",
                        "variableName": "Depth to water level, feet below land surface",
                        "unit": {"unitCode": "ft"},
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2020-01-01T00:00:00.000-05:00", "value": "10.0"},
                                {"dateTime": "2020-01-02T00:00:00.000-05:00", "value": ""},
                                {"dateTime": "2020-01-03T00:00:00.000-05:00", "value": "-999999"},
                                {"dateTime": "2021-06-01T00:00:00.000-05:00", "value": "5.0"},
                            ]
                        }
                    ],
                }
            ]
        }
    }
)


def test_parse_feet_to_metres_and_window():
    points = USGSGroundwaterConnector.parse_series(
        MOCK_NWIS,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    )
    by_date = {p.timestamp.date().isoformat(): p for p in points}
    # 2021-06-01 is outside [start, end).
    assert "2021-06-01" not in by_date
    # 10.0 ft -> metres at the connector boundary.
    good = by_date["2020-01-01"]
    assert good.value == pytest.approx(10.0 * FEET_TO_METERS)
    assert good.quality.value == "good"
    # blank and -999999 fill both map to MISSING.
    assert by_date["2020-01-02"].value is None
    assert by_date["2020-01-02"].quality.value == "missing"
    assert by_date["2020-01-03"].value is None
    assert by_date["2020-01-03"].quality.value == "missing"


def test_window_is_half_open():
    # the start-of-day local timestamp 2020-01-01T00:00-05:00 is 05:00 UTC, so
    # an end exactly at 2020-01-01T05:00Z must exclude it (half-open).
    pts = USGSGroundwaterConnector.parse_series(
        MOCK_NWIS,
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 1, 1, 5, tzinfo=UTC),
    )
    assert pts == []


def test_unit_already_metres_passes_through():
    payload = json.dumps(
        {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "parameterCode": "72019",
                            "variableName": "Depth to water level",
                            "unit": {"unitCode": "m"},
                        },
                        "values": [
                            {"value": [{"dateTime": "2020-01-01T00:00:00Z", "value": "3.5"}]}
                        ],
                    }
                ]
            }
        }
    )
    pts = USGSGroundwaterConnector.parse_series(
        payload, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert len(pts) == 1
    assert pts[0].value == pytest.approx(3.5)  # no feet conversion


def test_non_gw_variable_is_skipped():
    payload = json.dumps(
        {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "parameterCode": "00060",
                            "variableName": "Discharge, cubic feet per second",
                            "unit": {"unitCode": "ft3/s"},
                        },
                        "values": [
                            {"value": [{"dateTime": "2020-06-01T00:00:00Z", "value": "100.0"}]}
                        ],
                    }
                ]
            }
        }
    )
    pts = USGSGroundwaterConnector.parse_series(
        payload, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    assert pts == []


def test_empty_timeseries_yields_no_points():
    pts = USGSGroundwaterConnector.parse_series(
        json.dumps({"value": {"timeSeries": []}}),
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert pts == []


def test_invalid_json_raises_data_format_error():
    with pytest.raises(DataFormatError):
        USGSGroundwaterConnector.parse_series(
            "{not json", datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
        )


def test_station_id_normalisation():
    conn = USGSGroundwaterConnector()
    spec = ReductionSpec(
        domain_name="x",
        station_ids=("385854121023801", "usgs:123456", "USGS-987654"),
    )
    assert conn._station_ids(spec) == ["385854121023801", "123456", "987654"]


@pytest.mark.asyncio
async def test_list_sites_from_explicit_ids():
    conn = USGSGroundwaterConnector()
    spec = ReductionSpec(domain_name="x", station_ids=("385854121023801",))
    sites = await conn.list_sites(spec)
    assert len(sites) == 1
    assert sites[0].site_id == "usgs:385854121023801"
    assert sites[0].kind == "station"


@pytest.mark.asyncio
@respx.mock
async def test_fetch_series_builds_station_series():
    respx.get(url__regex=r"https://waterservices\.usgs\.gov/.*").mock(
        return_value=httpx.Response(200, text=MOCK_NWIS)
    )
    conn = USGSGroundwaterConnector()
    spec = ReductionSpec(domain_name="aquifer", station_ids=("385854121023801",))
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
    assert s.site.site_id == "usgs:385854121023801"
    # one good value within the window; the two MISSING rows carry value=None.
    assert len([p for p in s.points if p.value is not None]) == 1


# --------------------------------------------------------------------------- #
# PARITY-BY-CONSTRUCTION                                                        #
#                                                                              #
# The native SYMFLUENCE handler is                                            #
#   data/observation/handlers/usgs.py::USGSGroundwaterHandler.process().      #
# Its reduction is *point/station* — there is NO spatial reduction (no        #
# cos-lat weighting, no bbox mean); each NWIS value is one output row. So     #
# parity here is an IDENTITY parity, not a tolerance parity: the COS pure     #
# parser must reproduce the native record-by-record transform exactly, to     #
# float tolerance, on the same JSON.                                          #
#                                                                              #
# The native transform, distilled from process():                            #
#   1. variable select: '72019' in parameterCode OR 'depth to water level'    #
#      in variableName.lower() OR 'water level' in variableName.lower()       #
#   2. per value row, native does ``float(val_obj['value'])`` inside a        #
#      try/except (KeyError, ValueError). So:                                 #
#        - a blank "" -> float("") -> ValueError -> row SKIPPED (continue)    #
#        - a numeric  -> kept                                                 #
#   3. unit -> metres: unitCode.lower() in {ft,feet,foot} => *0.3048, else    #
#      passthrough (UnitConversion.FEET_TO_METERS == 0.3048)                  #
#   4. NO window trim in process() (the NWIS API call bounded the window).    #
#                                                                              #
# COS deliberately diverges in exactly two BENIGN, kind-correct ways, which   #
# this reimplementation encodes so the divergence is asserted, not hidden:    #
#   (a) COS maps the NWIS sentinel -999999 AND blank "" to QualityFlag.       #
#       MISSING (value=None). Native keeps -999999 as a real value            #
#       (-999999*0.3048 ≈ -304799 m) — a corrupt depth that the native        #
#       handler would silently feed downstream. Treating it as MISSING is     #
#       strictly more correct for the groundwater objective and is the SI-    #
#       canonical fill rule used by every other COS connector. Native and COS #
#       agree on the *blank* -> not-a-value outcome (native skips, COS emits  #
#       a MISSING row); they agree that neither becomes a finite depth.       #
#   (b) COS additionally trims to the half-open UTC window [start, end),      #
#       which native delegates to the API. On the unbounded native input we   #
#       compare only rows inside the window.                                  #
# On the FINITE, in-window, in-unit values (the data that defines the kind's  #
# objective) COS == native to float tolerance.                               #
# --------------------------------------------------------------------------- #

# canonical feet->metres factor used by the NATIVE handler
# (symfluence.core.constants.UnitConversion.FEET_TO_METERS).
NATIVE_FEET_TO_METERS = 0.3048


def _native_process(text: str):
    """Reimplement USGSGroundwaterHandler.process() record transform inline.

    Returns a list of ``(utc_iso_datetime_string, value_or_None)`` exactly as
    the native handler would land them in its output DataFrame (no window
    trim, native fill semantics: blank skipped, -999999 kept as a real value).
    """
    data = json.loads(text)
    rows: list[tuple[str, float | None]] = []
    for ts in data["value"]["timeSeries"]:
        param_code = ts.get("variable", {}).get("parameterCode", "")
        param_name = ts.get("variable", {}).get("variableName", "").lower()
        is_gw = (
            GW_PARAM_CODE in param_code
            or "depth to water level" in param_name
            or "water level" in param_name
        )
        if not is_gw:
            continue
        unit = ts.get("variable", {}).get("unit", {}).get("unitCode", "unknown").lower()
        for container in ts.get("values", []):
            for obj in container.get("value", []):
                try:
                    dt = obj["dateTime"]
                    val = float(obj["value"])  # native: blank "" raises -> skip
                except (KeyError, ValueError):
                    continue
                if unit in ("ft", "feet", "foot"):
                    val *= NATIVE_FEET_TO_METERS
                rows.append((dt, val))
    return rows


# A synthetic NWIS payload with the full spread the parity must cover:
#  - a normal feet value (unit conversion),
#  - a second normal feet value (so the comparison is non-trivial / multi-row),
#  - a blank "" (native skips; COS -> MISSING),
#  - the -999999 sentinel (native keeps; COS -> MISSING),
#  - one out-of-window row (COS trims; native would keep).
PARITY_NWIS = json.dumps(
    {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "parameterCode": "72019",
                        "variableName": "Depth to water level, feet below land surface",
                        "unit": {"unitCode": "ft"},
                    },
                    "values": [
                        {
                            "value": [
                                {"dateTime": "2020-01-01T00:00:00.000-05:00", "value": "10.0"},
                                {"dateTime": "2020-02-01T00:00:00.000-05:00", "value": "12.5"},
                                {"dateTime": "2020-03-01T00:00:00.000-05:00", "value": ""},
                                {"dateTime": "2020-04-01T00:00:00.000-05:00", "value": "-999999"},
                                {"dateTime": "2021-06-01T00:00:00.000-05:00", "value": "7.0"},
                            ]
                        }
                    ],
                }
            ]
        }
    }
)

# half-open window that includes the 2020 rows and excludes the 2021 row.
_WIN_START = datetime(2020, 1, 1, tzinfo=UTC)
_WIN_END = datetime(2021, 1, 1, tzinfo=UTC)


def _native_in_window():
    """Native rows restricted to [start, end), keyed by UTC date iso string."""
    out: dict[str, float | None] = {}
    for raw_dt, val in _native_process(PARITY_NWIS):
        dt = datetime.fromisoformat(raw_dt)
        dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
        if _WIN_START <= dt < _WIN_END:
            out[dt.date().isoformat()] = val
    return out


def test_parity_finite_values_identical_to_native():
    """COS finite depths == native finite depths, to float tolerance.

    This is the load-bearing parity assertion: on the values that constitute
    the groundwater objective (real, in-window, feet->metres-converted depths)
    the COS pure parser reproduces the native handler exactly.
    """
    cos_points = USGSGroundwaterConnector.parse_series(PARITY_NWIS, _WIN_START, _WIN_END)
    cos_by_date = {p.timestamp.date().isoformat(): p for p in cos_points}
    native = _native_in_window()

    # the two finite, in-window depths agree exactly with native feet->metres.
    for day, expected in native.items():
        # native -999999 is a (corrupt) finite value; the finite-value parity
        # is over the *physical* depths only — see divergence note below.
        if expected is not None and expected != NWIS_NODATA * NATIVE_FEET_TO_METERS:
            assert cos_by_date[day].value == pytest.approx(expected, rel=0, abs=1e-12)
            assert cos_by_date[day].quality is QualityFlag.GOOD

    # concretely: 10.0 ft and 12.5 ft round-trip identically through both.
    assert cos_by_date["2020-01-01"].value == pytest.approx(10.0 * NATIVE_FEET_TO_METERS, abs=1e-12)
    assert cos_by_date["2020-02-01"].value == pytest.approx(12.5 * NATIVE_FEET_TO_METERS, abs=1e-12)


def test_parity_unit_factor_matches_native_constant():
    """COS feet->metres factor is byte-identical to the native constant."""
    assert FEET_TO_METERS == NATIVE_FEET_TO_METERS == 0.3048


def test_parity_window_half_open_vs_native_unbounded():
    """COS trims [start, end); the 2021 row native keeps is excluded by COS."""
    native = _native_process(PARITY_NWIS)
    # native (unbounded) sees the 2021 row...
    assert any(dt.startswith("2021-06-01") for dt, _ in native)
    # ...COS (windowed) does not.
    cos_points = USGSGroundwaterConnector.parse_series(PARITY_NWIS, _WIN_START, _WIN_END)
    assert all(p.timestamp.year != 2021 for p in cos_points)


def test_parity_fill_divergence_is_benign_and_explicit():
    """Document + lock the ONE intentional divergence from native.

    Native: blank "" is SKIPPED; -999999 is KEPT as a real (corrupt) value.
    COS:    blank "" -> MISSING(value=None); -999999 -> MISSING(value=None).

    The divergence is benign for the groundwater objective: neither system
    produces a *true physical depth* for these rows, and COS's MISSING is the
    SI-canonical, downstream-safe representation (native would inject
    -304799 m). This test asserts the divergence is exactly as described, so
    it can never silently change.
    """
    cos_points = USGSGroundwaterConnector.parse_series(PARITY_NWIS, _WIN_START, _WIN_END)
    cos_by_date = {p.timestamp.date().isoformat(): p for p in cos_points}

    # blank -> MISSING in COS (native would have skipped the row entirely).
    assert cos_by_date["2020-03-01"].value is None
    assert cos_by_date["2020-03-01"].quality is QualityFlag.MISSING

    # -999999 -> MISSING in COS; native keeps it as a corrupt finite depth.
    native_rows = dict(
        (datetime.fromisoformat(dt).date().isoformat(), v)
        for dt, v in _native_process(PARITY_NWIS)
    )
    assert native_rows["2020-04-01"] == pytest.approx(NWIS_NODATA * NATIVE_FEET_TO_METERS)
    assert cos_by_date["2020-04-01"].value is None
    assert cos_by_date["2020-04-01"].quality is QualityFlag.MISSING

    # crucially: COS never emits a finite depth that native would not, and
    # never emits a corrupt large-magnitude depth.
    finite_cos = [p.value for p in cos_points if p.value is not None]
    assert finite_cos == pytest.approx([10.0 * NATIVE_FEET_TO_METERS, 12.5 * NATIVE_FEET_TO_METERS])


def test_parity_single_constant_field_exact():
    """Degenerate single-value case: COS == native to float tolerance.

    A point network has no spatial reduction, so a single value must pass
    through identically (the 'single-cell / constant field' parity check).
    """
    payload = json.dumps(
        {
            "value": {
                "timeSeries": [
                    {
                        "variable": {
                            "parameterCode": "72019",
                            "variableName": "Depth to water level",
                            "unit": {"unitCode": "ft"},
                        },
                        "values": [
                            {"value": [{"dateTime": "2020-06-15T12:00:00Z", "value": "42.0"}]}
                        ],
                    }
                ]
            }
        }
    )
    cos_points = USGSGroundwaterConnector.parse_series(
        payload, datetime(2020, 1, 1, tzinfo=UTC), datetime(2021, 1, 1, tzinfo=UTC)
    )
    native = _native_process(payload)
    assert len(cos_points) == len(native) == 1
    assert cos_points[0].value == pytest.approx(native[0][1], abs=1e-12)
    assert cos_points[0].value == pytest.approx(42.0 * NATIVE_FEET_TO_METERS, abs=1e-12)


@pytest.mark.network
@pytest.mark.asyncio
async def test_live_smoke_usgs_gw():
    """LIVE smoke against the real anonymous USGS NWIS endpoint.

    Run with: pytest -m network tests/connectors/test_usgs_gw.py -k live
    """
    conn = USGSGroundwaterConnector()
    # a long-running NWIS groundwater observation well.
    spec = ReductionSpec(domain_name="live", station_ids=("385854121023801",))
    async with conn:
        series_list = await conn.fetch_series(
            spec, datetime(2015, 1, 1, tzinfo=UTC), datetime(2020, 1, 1, tzinfo=UTC)
        )
    assert series_list and series_list[0].unit == "m"
