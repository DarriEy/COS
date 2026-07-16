# Changelog

All notable changes to COS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
semantic versioning.

## [Unreleased]

## [0.2.0] — 2026-07-15

### Added
- Complete roster of 50 connectors across 20 canonical observation kinds.
- Structured parity evidence for every connector; 44 connectors validated on
  real provider data and the remainder explicitly labeled by gating reason.
- `cos validation` / `--json-output` validation reporting and `cos sites`
  station/feature discovery.
- JSON canonical-series output from `cos fetch`.
- Weekly live-health workflow with four bounded anonymous provider sentinels,
  plus validation-report and JUnit artifacts.
- SYMFLUENCE community-backend capabilities, data-license posture gates, and
  non-streamflow routing for six evaluator families.

### Changed
- Release and documentation now distinguish live native parity, live product-
  specification validation, and parity-by-construction.
- Package version and connector user agent advanced to 0.2.

### Fixed
- Registered the `live` pytest marker used by product smoke tests.

## [0.1.0] — 2026-06-13

Initial scaffold. **This was a scaffold, not a complete service:** 3 of ~32
enumerated non-streamflow connectors are implemented (≈9% coverage). The
architecture, the canonical contract, and the SYMFLUENCE integration are
complete; most of the connector surface is unbuilt.

### Added
- **Canonical heterogeneous-observation contract** (`cos.core.models`): one
  `ObservationSeries` model that both gridded reductions and point/tower stations
  collapse into, tagged by `ObservationKind` (carrying the frozen SI unit per
  kind) and `SiteRef` (station vs reduced region). `ReductionSpec` carries the
  geometry + reduction policy.
- **Gridded spatial-reduction kernels** (`cos.core.reduce`): area-weighted
  (cos-lat) `basin_mean`, `nearest_cell`/`point_sample`, 0–360 longitude
  normalization, NaN→MISSING.
- **Three proof connectors** spanning the structural split: `grace` (TWS,
  gridded basin reduction, cm→mm anomaly), `snotel` (SWE, point network,
  inches→mm, **live-smoked**), `openet` (ET, ensemble, mm/period→mm/day).
- **Connector registry + discovery**, per-provider config + credential
  resolution (Earthdata/CDS/OpenET/AmeriFlux pass-through), roster health.
- **`cos` CLI**: `providers`, `kinds`, `health`, `fetch`.
- **SYMFLUENCE `ObservationBackend`** (contract 0.3.0) declaring the implemented
  non-streamflow kinds, with the OBS_CSV_V1 protocol delivery + sidecar manifest;
  defensive symfluence import; entry-point + self-registration.
- **Honest roster** (`inventory/providers.yaml`): all 32 connectors,
  status-labeled (3 implemented, 19 planned, 9 research, 1 manual).
- **Roster-integrity tests** (every connector tiered/tested/documented;
  no streamflow; valid kind/class/auth), hermetic connector tests
  (DNS-block + synthetic payloads), reduction kernel tests, CLI + integration
  tests, JOSS-ready repo files, CI (ruff+mypy+pytest), docs (mkdocs).

### Known gaps (reported, not hidden)
- 29 of 32 connectors unbuilt.
- No native parity grades yet — connectors are unit/contract-validated and
  (SNOTEL) live-smoked, not compared against the native SYMFLUENCE handlers.
- COS is **not wired into the SYMFLUENCE manager flow** — registering the backend
  does not route the evaluation pipeline through COS for non-streamflow kinds;
  that is a required SYMFLUENCE follow-up.
