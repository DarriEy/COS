"""Operational-integrity guards for the COS connector roster.

Catches silent regressions unit tests miss: a registered connector with no tier,
no test, or no inventory entry; an inventory overclaim; a tier ghost; a class
missing required metadata or whose slug disagrees with its registry key.
Mirrors the CSFS roster-integrity regime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cos.core.models import ObservationKind
from cos.core.registry import discover, get_connector, list_providers
from cos.scheduler.tiers import PROVIDER_TIERS, TIER_LOOKBACK_DAYS

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONNECTOR_TESTS_DIR = Path(__file__).resolve().parent / "connectors"
_INVENTORY_PATH = _REPO_ROOT / "inventory" / "providers.yaml"


@pytest.fixture(scope="module")
def registered() -> set[str]:
    discover()
    return set(list_providers())


@pytest.fixture(scope="module")
def tier_assignments() -> dict[str, int]:
    counts: dict[str, int] = {}
    for slugs in PROVIDER_TIERS.values():
        for slug in slugs:
            counts[slug] = counts.get(slug, 0) + 1
    return counts


def test_every_connector_is_in_exactly_one_tier(registered, tier_assignments):
    orphaned = sorted(s for s in registered if s not in tier_assignments)
    duplicated = sorted(s for s, n in tier_assignments.items() if n > 1)
    assert not orphaned, f"registered connectors with no tier: {orphaned}"
    assert not duplicated, f"connectors in multiple tiers: {duplicated}"


def test_no_tier_references_an_unregistered_connector(registered, tier_assignments):
    ghosts = sorted(s for s in tier_assignments if s not in registered)
    assert not ghosts, f"tiers reference unregistered connectors: {ghosts}"


def test_every_tier_has_a_lookback():
    tiers = set(PROVIDER_TIERS)
    assert tiers <= set(TIER_LOOKBACK_DAYS), (
        f"tiers missing a lookback: {sorted(tiers - set(TIER_LOOKBACK_DAYS))}"
    )


def test_connector_classes_have_required_metadata(registered):
    valid_kinds = {k.value for k in ObservationKind}
    valid_classes = {"gridded", "point_network", "flux_tower"}
    problems: list[str] = []
    for slug in registered:
        cls = get_connector(slug)
        if getattr(cls, "slug", None) != slug:
            problems.append(f"{slug}: class.slug={getattr(cls, 'slug', None)!r} != key")
        if not getattr(cls, "display_name", ""):
            problems.append(f"{slug}: missing display_name")
        if not getattr(cls, "base_url", ""):
            problems.append(f"{slug}: missing base_url")
        kind = getattr(cls, "kind", None)
        if kind is None or kind.value not in valid_kinds:
            problems.append(f"{slug}: invalid kind {kind!r}")
        if kind is not None and kind.value == "streamflow":
            problems.append(f"{slug}: COS must NOT serve streamflow (that is CSFS)")
        if getattr(cls, "structural_class", None) not in valid_classes:
            problems.append(f"{slug}: invalid structural_class {getattr(cls, 'structural_class', None)!r}")
        if not isinstance(getattr(cls, "auth", None), frozenset):
            problems.append(f"{slug}: auth must be a frozenset")
    assert not problems, "connector metadata problems:\n" + "\n".join(problems)


def test_no_connector_serves_streamflow(registered):
    """Hard scope boundary: streamflow is CSFS's, never COS's."""
    offenders = [s for s in registered if getattr(get_connector(s), "kind", None)
                 and get_connector(s).kind.value == "streamflow"]
    assert not offenders, f"COS connectors serving streamflow (forbidden): {offenders}"


@pytest.fixture(scope="module")
def inventory_entries() -> list[dict]:
    return [e for e in yaml.safe_load(_INVENTORY_PATH.read_text(encoding="utf-8")) if isinstance(e, dict)]


@pytest.fixture(scope="module")
def inventory_slugs(inventory_entries) -> set[str]:
    return {e["slug"] for e in inventory_entries if "slug" in e}


def test_every_registered_connector_is_documented_in_inventory(registered, inventory_slugs):
    undocumented = sorted(registered - inventory_slugs)
    assert not undocumented, (
        "registered connectors missing from inventory/providers.yaml: " + ", ".join(undocumented)
    )


def test_implemented_inventory_entries_have_a_registered_connector(registered, inventory_entries):
    overclaims = sorted(
        e["slug"] for e in inventory_entries
        if e.get("status") == "implemented" and e.get("slug") not in registered
    )
    assert not overclaims, (
        "inventory marks these 'implemented' but no connector is registered: " + ", ".join(overclaims)
    )


def test_every_registered_connector_has_a_test(registered):
    files = {p.name for p in _CONNECTOR_TESTS_DIR.glob("test_*.py")}
    blob = "".join((_CONNECTOR_TESTS_DIR / f).read_text(encoding="utf-8") for f in files)
    missing = [
        slug for slug in sorted(registered)
        if f"test_{slug}.py" not in files and f'"{slug}"' not in blob and f"'{slug}'" not in blob
    ]
    assert not missing, "registered connectors with NO test coverage:\n  " + "\n  ".join(missing)


def test_inventory_kinds_are_valid(inventory_entries):
    valid = {k.value for k in ObservationKind}
    bad = sorted(e["slug"] for e in inventory_entries if e.get("kind") not in valid)
    assert not bad, f"inventory entries with invalid/missing kind: {bad}"
