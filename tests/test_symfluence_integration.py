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
    _LICENSE_POSTURE,
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


def test_capabilities_declare_registered_nonstreamflow_connectors():
    # One capability per registered COS connector (the list is derived from the
    # connector registry, so it tracks the build-out automatically).
    from cos.core.registry import discover, list_providers

    discover()
    provider_ids = {s.provider_id for s in OBSERVATION_CAPABILITIES}
    assert provider_ids == set(list_providers())
    # The core kinds are present; streamflow is NEVER claimed (that is CSFS's).
    kinds = {s.kind for s in OBSERVATION_CAPABILITIES}
    assert {ObservationKind.TWS, ObservationKind.SWE, ObservationKind.ET} <= kinds
    assert all(s.kind.value != "streamflow" for s in OBSERVATION_CAPABILITIES)
    # Parity-validated connectors carry a real grade (admitted without the
    # ALLOW_UNGATED waiver); the rest stay ungated (None) pending a native run.
    graded = {s.provider_id for s in OBSERVATION_CAPABILITIES if s.parity_grade}
    assert {"grace", "snotel"} <= graded


def test_every_connector_has_an_explicit_license_posture():
    # Roster integrity: nothing ships unscoped. Every registered connector must
    # carry an explicit posture entry (mirrors the tier / grade integrity gates),
    # so a new connector cannot silently default to UNKNOWN.
    from cos.core.registry import discover, list_providers

    discover()
    registered = set(list_providers())
    missing = sorted(registered - set(_LICENSE_POSTURE))
    assert not missing, f"connectors with no license posture: {missing}"
    ghosts = sorted(set(_LICENSE_POSTURE) - registered)
    assert not ghosts, f"license posture references unregistered connectors: {ghosts}"
    # Postures use the contract's redistribution vocabulary.
    bad = sorted(s for s, (r, *_rest) in _LICENSE_POSTURE.items()
                 if r not in {"open", "attribution", "restricted", "unknown"})
    assert not bad, f"connectors with invalid redistribution posture: {bad}"


def test_restricted_sources_are_labeled_for_the_gate():
    # The license-scoping pass found exactly these no-third-party-redistribution
    # sources; the SYMFLUENCE gate refuses to MIRROR them via COS (users fetch
    # them natively). cmc_swe was native-parity-validated but still may not be
    # re-served — the whole point of the scoping.
    restricted = {s for s, (r, *_rest) in _LICENSE_POSTURE.items() if r == "restricted"}
    assert restricted == {"cmc_swe", "cmc_snow_depth", "ismn_sm"}
    # ...and they propagate onto the derived capability specs.
    caps = {s.provider_id: s for s in OBSERVATION_CAPABILITIES}
    assert caps["cmc_swe"].redistribution == "restricted"
    assert caps["cmc_swe"].attribution  # CMC citation propagated even though refused


def test_noncommercial_is_orthogonal_to_redistribution():
    # CC-BY-NC / OpenET / GLEAM are REDISTRIBUTABLE WITH ATTRIBUTION but
    # non-commercial — surfaced as a warning, NOT refused. They must NOT be marked
    # 'restricted' (that would wrongly block a redistributable source).
    nc = {s for s, (_r, _dl, _a, ncf) in _LICENSE_POSTURE.items() if ncf}
    assert {"mswep_precip", "gleam_et", "openet"} <= nc
    for slug in ("mswep_precip", "gleam_et", "openet"):
        r, _dl, attribution, ncf = _LICENSE_POSTURE[slug]
        assert r == "attribution" and ncf is True and attribution


def test_attribution_sources_carry_attribution_text():
    # Every attribution / restricted source must propagate a non-empty attribution
    # string (open/CC0 sources require none).
    for slug, (r, _dl, attribution, _nc) in _LICENSE_POSTURE.items():
        if r in {"attribution", "restricted"}:
            assert attribution, f"{slug}: {r} posture must carry attribution text"
        if r == "open":
            assert attribution == "", f"{slug}: open posture should require no attribution"


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


def test_capabilities_propagate_license_posture_to_contract():
    contract = pytest.importorskip("symfluence.data.backends.contract")
    import cos.integrations.symfluence as integ

    caps = {c.provider_id: c for c in integ.CommunityObservationBackend().capabilities()}
    # Restricted CMC: maps onto the contract's hard-refusal enum.
    assert caps["cmc_swe"].redistribution == contract.Redistribution.RESTRICTED
    assert caps["cmc_swe"].noncommercial is True
    assert caps["cmc_swe"].attribution
    # Open NASA source: OPEN, no attribution required.
    assert caps["grace"].redistribution == contract.Redistribution.OPEN
    assert caps["grace"].attribution == ""
    # Attribution + NC source: redistributable WITH attribution, NC flagged.
    assert caps["mswep_precip"].redistribution == contract.Redistribution.ATTRIBUTION
    assert caps["mswep_precip"].noncommercial is True
    assert caps["mswep_precip"].attribution
