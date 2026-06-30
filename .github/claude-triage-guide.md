# CI Triage Guide — COS (Community Observation Service)

This file is read by the automated CI triage agent (`.github/workflows/ci-autotriage.yml`).
It defines how to classify a CI failure for **this** service and what is safe to auto-fix.

## What this service is

COS connectors fetch **earth-observation series** (SWE, ET, soil moisture, LST, TWS, …) from
gridded products and point networks, and return a canonical `ObservationSeries` — **in the kind's
canonical SI unit (`KIND_UNITS`) and UTC**, with all unit conversions done at the connector
boundary. One connector wraps one upstream product/provider.

## Classifications and actions

Pick exactly one. The action column is enforced by the workflows — the auto-merge job
**only** merges `adapter_drift`/`data_drift` fixes, and **only** when every changed file is
under `src/cos/connectors/` or `tests/`.

| Classification | What it means | Action |
|---|---|---|
| `adapter_drift` | A **data provider** changed something a connector consumes (endpoint, variable/band, scale factor, fill/flag sentinel, units, grid). Fixable **entirely inside `src/cos/connectors/<slug>.py`** and/or its test. | Fix PR → **auto-merge on green** |
| `data_drift` | Contract and live provider are fine, but a recorded fixture / expected value in a test is stale. Fixable inside `connectors/` or `tests/`. | Fix PR → **auto-merge on green** |
| `contract_change` | The failure involves the canonical schema/contract — anything under `src/cos/core/` (`models.py`: `SiteRef`, `ObservationSeries`, the `KIND_UNITS` map, `ObservationKind`) or `BaseObservationConnector`. | Fix PR → **human merge** |
| `tooling_drift` | A build / CI / dependency / tooling failure: mypy or ruff config, a dependency version bump (numpy, xarray, …), type stubs, packaging, or the CI workflow itself. Also the **roster-integrity** test (`tests/test_connector_integrity.py`) failing because a connector isn't registered/tiered/inventoried — unless the right fix is purely in `connectors/`. **Not** a data-provider change. | Fix PR → **human merge** |
| `outage` | Transient external failure: HTTP 429/5xx, DNS, connection timeouts, provider/auth-service hiccups. | **Report only** (recommend re-run) |
| `real_bug` | A genuine logic error in non-adapter COS code. | **Report only** (describe the fix) |
| `other` | You cannot confidently classify it. | **Report only** |

## The canonical contract — never auto-fixed

Editing anything under `src/cos/core/` is a `contract_change` (human-only), never drift:
- `models.py` — `SiteRef`, `ObservationSeries`, `ObservationKind`, and especially `KIND_UNITS`
  (the canonical SI unit per kind: SWE mm, ET mm/day, SOIL_MOISTURE m3/m3, LST K, TWS mm, …).
  The model validates the unit and rejects a mismatch — never work around that validation.
- `BaseObservationConnector` (`connectors/base.py`) — the public connector interface.

## The scope rule (critical — read before opening any fix PR)

An `adapter_drift` / `data_drift` fix **must change only files under `src/cos/connectors/` or
`tests/`**. If the minimal fix would touch **any** other path — `pyproject.toml`, `.github/`,
`src/cos/core/`, `inventory/`, docs, packaging — then it is **not** adapter/data drift.
Reclassify:
- touches `src/cos/core/` (or `BaseObservationConnector`/`KIND_UNITS`) → `contract_change`
- touches build/CI/deps/inventory (e.g. `pyproject.toml` mypy/ruff/version config) → `tooling_drift`

Both take the **human-gated** path (label `needs-human-review`, never `automerge-on-green`).
"Upstream changed" applies to **data providers**, not to libraries like numpy/mypy. The
auto-merge job will refuse to merge any PR that changes files outside `connectors/`/`tests/`,
even if mislabeled.

## CI commands (what "green" means here)

```
ruff check src/ tests/
mypy src/cos/
pytest tests/ -v --tb=short -m "not network"
```

Never make CI pass by skipping/weakening tests, loosening assertions, or marking things `network`
to deselect them. Fix the cause or classify honestly.
