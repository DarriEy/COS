from cos.core.validation import validation_report, validation_summary, validation_tier


def test_validation_tiers_are_conservative():
    assert validation_tier(None) == "unvalidated"
    assert validation_tier("LIVE: exact") == "live-parity"
    assert validation_tier("LIVE-spec: product spec") == "live-spec"
    assert validation_tier("parity-by-construction vs native") == "parity-by-construction"


def test_every_connector_has_a_grade_and_report_row():
    from cos.core.registry import discover, list_providers

    discover()
    rows = validation_report()
    assert {row["provider"] for row in rows} == set(list_providers())
    assert all(row["grade"] for row in rows)
    assert validation_summary(rows).get("unvalidated", 0) == 0
