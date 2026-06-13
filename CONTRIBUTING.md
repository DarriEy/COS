# Contributing to COS

Thanks for helping extend the Community Observation Service. COS covers the
**non-streamflow** hydrological observation kinds; streamflow belongs to
[CSFS](https://github.com/DarriEy/CSFS) and must never be added here.

## Development setup

```bash
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/cos/
pytest -m "not network"        # hermetic suite; must stay green
```

Tests are hermetic: `tests/conftest.py` blocks real network access, so
connectors must mock HTTP with `respx` or use synthetic local payloads. A test
that genuinely needs an upstream is marked `@pytest.mark.network` and deselected
in CI.

## The roster-integrity contract (read before adding a connector)

`tests/test_connector_integrity.py` enforces the same honesty regime as CSFS.
A new connector is rejected by the suite unless **all** hold:

1. **Registered** — decorated with `@register("<slug>")`; `class.slug` equals the
   registry key.
2. **Serves a non-streamflow kind** — `kind` is a valid `ObservationKind`; it is
   **never** `streamflow` (hard scope boundary, asserted).
3. **Metadata complete** — `display_name`, `base_url`, `structural_class`
   (`gridded`/`point_network`/`flux_tower`), and `auth` (a `frozenset`).
4. **Tiered** — listed in exactly one tier in `cos/scheduler/tiers.py`.
5. **Tested** — has `tests/connectors/test_<slug>.py` (or is referenced by slug
   in a shared test).
6. **Documented in `inventory/providers.yaml`** — and you may mark it
   `status: implemented` **only** if the connector is actually registered.
   Otherwise use `planned` / `research` / `manual` / `fallback`. Inventory
   overclaims fail the suite.

## The canonical contract (the make-or-break rule)

Every connector must emit `ObservationSeries` **in the kind's canonical SI unit**
(`KIND_UNITS`) and **UTC**, with all unit conversions done at the connector
boundary (inches→mm, cm→mm, LE→ET, scale/offset, etc.). The model validates the
unit and rejects a mismatch — do not work around it. Gridded connectors reduce
via `cos.core.reduce`; point connectors select stations from the `ReductionSpec`.

## Parity gate

A connector ships `implemented` once its hermetic tests pass. It earns a
`parity_grade` in the SYMFLUENCE backend capability only after a recorded
comparison against the native SYMFLUENCE handler on a reference basin/station
(tolerance-based for gridded reductions — basin-mean is not bitwise). Do not
claim a parity grade you have not measured.

## Honesty

Report gaps, not wins. Coverage is a fraction of the native surface; say so. No
premature "done" — `pytest` green is necessary, not sufficient.

## Commits & attribution

License header on every `.py` file:
`# SPDX-License-Identifier: GPL-3.0-or-later`. Line length 120, target 3.11+.
