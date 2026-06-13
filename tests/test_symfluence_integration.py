"""SYMFLUENCE integration tests.

The pure helpers are tested standalone (no SYMFLUENCE). The backend registration
+ capability shape is tested with importorskip against the SYMFLUENCE venv (COS
installed editable --no-deps into it). The manager-flow WIRING is deliberately
NOT asserted here — it does not exist (see the integration module docstring and
cos_design.md §4); these tests assert availability and conformance only.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cos.core.models import (
    ObservationKind,
    ObservationPoint,
    ObservationSeries,
    QualityFlag,
    SiteRef,
    SpatialReduction,
)
from cos.integrations.symfluence import (
    OBSERVATION_CAPABILITIES,
    canonical_columns_for_kind,
    series_to_obs_csv_v1_frame,
)


def _swe_series() -> ObservationSeries:
    return ObservationSeries(
        provider="snotel",
        kind=ObservationKind.SWE,
        site=SiteRef(kind="station", site_id="snotel:679"),
        reduction=SpatialReduction.STATION,
        unit="mm",
        points=[
            ObservationPoint(timestamp=datetime(2020, 1, 1, tzinfo=UTC), value=254.0, quality=QualityFlag.GOOD),
            ObservationPoint(timestamp=datetime(2020, 1, 2, tzinfo=UTC), value=None, quality=QualityFlag.MISSING),
            ObservationPoint(timestamp=datetime(2020, 1, 3, tzinfo=UTC), value=300.0, quality=QualityFlag.GOOD),
        ],
        fetched_at=datetime(2020, 2, 1, tzinfo=UTC),
    )


def test_obs_csv_v1_frame_drops_missing_and_window_trims():
    pd = pytest.importorskip("pandas")
    df = series_to_obs_csv_v1_frame(
        _swe_series(),
        start=datetime(2020, 1, 1), end=datetime(2020, 1, 3),  # half-open: excludes 01-03
    )
    assert list(df.columns) == ["datetime", "value", "quality_flag"]
    # 01-01 kept; 01-02 dropped (missing value); 01-03 excluded by half-open end.
    assert len(df) == 1
    assert df.iloc[0]["value"] == 254.0
    assert pd.Timestamp(df.iloc[0]["datetime"]).tzinfo is None  # naive == UTC


def test_canonical_columns_match_symfluence_standard_columns():
    # The widening targets must match observation/base.py STANDARD_COLUMNS.
    assert canonical_columns_for_kind(ObservationKind.TWS) == ["datetime", "tws_anomaly_mm", "uncertainty_mm"]
    assert canonical_columns_for_kind(ObservationKind.SWE) == ["datetime", "swe_mm", "quality_flag"]
    assert canonical_columns_for_kind(ObservationKind.ET) == ["datetime", "et_mm_day", "quality_flag"]
    assert canonical_columns_for_kind(ObservationKind.SOIL_MOISTURE) == ["datetime", "value", "depth_m", "quality_flag"]


def test_capabilities_declare_only_implemented_nonstreamflow_kinds():
    provider_ids = {s.provider_id for s in OBSERVATION_CAPABILITIES}
    assert provider_ids == {"grace", "snotel", "openet"}
    kinds = {s.kind for s in OBSERVATION_CAPABILITIES}
    assert ObservationKind.SWE in kinds and ObservationKind.TWS in kinds and ObservationKind.ET in kinds
    # No streamflow — that is CSFS's.
    assert all(s.kind.value != "streamflow" for s in OBSERVATION_CAPABILITIES)
    # All ungated at scaffold time (no native parity yet) — honest.
    assert all(s.parity_grade is None for s in OBSERVATION_CAPABILITIES)


def test_import_does_not_require_symfluence():
    # The module must import whether or not symfluence is present.
    import cos.integrations.symfluence as integ

    assert hasattr(integ, "register")
    assert hasattr(integ, "CommunityObservationBackend")


# -- SYMFLUENCE-present tests (skipped when the framework is absent) ----------


def test_backend_registers_and_declares_kinds():
    pytest.importorskip("symfluence")
    from symfluence.core.registries import R

    import cos.integrations.symfluence as integ

    integ.register()
    backends = getattr(R, "observation_backends", None)
    assert backends is not None, "framework has no observation_backends registry"
    assert "community-observation" in backends

    backend = integ.CommunityObservationBackend()
    caps = backend.capabilities()
    served_kinds = {k for c in caps for k in c.kinds}
    assert "swe" in served_kinds and "tws" in served_kinds and "et" in served_kinds
    assert "streamflow" not in served_kinds


def test_backend_interface_version_compatible():
    contract = pytest.importorskip("symfluence.data.backends.contract")
    import cos.integrations.symfluence as integ

    assert contract.is_compatible(integ.TARGET_INTERFACE_VERSION)
