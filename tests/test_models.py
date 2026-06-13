"""Canonical-model contract tests — the make-or-break design (design §2)."""

from datetime import UTC, datetime

import pytest

from cos.core.models import (
    KIND_UNITS,
    ObservationKind,
    ObservationPoint,
    ObservationSeries,
    QualityFlag,
    SiteRef,
    SpatialReduction,
)


def _series(kind: ObservationKind, unit: str) -> ObservationSeries:
    return ObservationSeries(
        provider="x",
        kind=kind,
        site=SiteRef(kind="station", site_id="x:1"),
        reduction=SpatialReduction.STATION,
        unit=unit,
        points=[ObservationPoint(timestamp=datetime(2020, 1, 1, tzinfo=UTC), value=1.0, quality=QualityFlag.GOOD)],
        fetched_at=datetime(2020, 1, 2, tzinfo=UTC),
    )


def test_every_kind_has_a_canonical_unit():
    for kind in ObservationKind:
        assert kind in KIND_UNITS, f"{kind} missing from KIND_UNITS"


def test_streamflow_is_not_a_kind():
    # Streamflow belongs to CSFS; COS must never serve it.
    assert "streamflow" not in {k.value for k in ObservationKind}


def test_series_enforces_canonical_unit():
    # Correct unit accepted.
    s = _series(ObservationKind.SWE, "mm")
    assert s.unit == "mm"
    # Wrong unit rejected loudly — the contract neutralizes the inches landmine.
    with pytest.raises(ValueError, match="canonical unit"):
        _series(ObservationKind.SWE, "in")


def test_site_ref_serves_both_worlds():
    station = SiteRef(kind="station", site_id="snotel:679")
    region = SiteRef(kind="reduced_region", site_id="grace:domain:bow")
    assert station.kind == "station"
    assert region.kind == "reduced_region"


def test_tws_carries_uncertainty():
    s = ObservationSeries(
        provider="grace",
        kind=ObservationKind.TWS,
        site=SiteRef(kind="reduced_region", site_id="grace:domain:x"),
        reduction=SpatialReduction.BASIN_MEAN,
        unit="mm",
        points=[ObservationPoint(timestamp=datetime(2020, 1, 1, tzinfo=UTC), value=5.0, uncertainty=1.2)],
        fetched_at=datetime(2020, 1, 2, tzinfo=UTC),
    )
    assert s.points[0].uncertainty == 1.2
