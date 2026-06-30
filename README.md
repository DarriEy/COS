# COS — Community Observation Service

Harmonized acquisition of **non-streamflow** hydrological observations — total
water storage, snow water equivalent, snow cover, evapotranspiration, soil
moisture, groundwater, LAI, LST, precipitation-as-observation, surface water,
and water level — reduced to one canonical per-site time-series contract for
hydrological model **evaluation and calibration**.

COS is the non-streamflow sibling of [CSFS](https://github.com/DarriEy/CSFS)
(streamflow). It is part of the SYMFLUENCE community-services family
(CSFS streamflow · CFS forcing · CAS attributes · COS observations).

## Statement of need

Evaluating a hydrological model against observations today means a bespoke
acquisition + reduction pipeline *per product*: GRACE basin-averaging, SNOTEL CSV
scraping, OpenET ensemble pulls, SMAP gridding, MODIS scale-factor handling —
each with its own units, time conventions, auth, and spatial-reduction quirks.
SYMFLUENCE alone carries ~38 such handlers across 13 kinds. The conventions
diverge silently (SNOTEL in inches, FLUXNET in latent-heat flux, GRACE in cm on a
0–360 grid), so a unit or window mistake propagates into the objective function
unnoticed.

COS replaces that with **one canonical contract**. Heterogeneous sources —
gridded products that must be *spatially reduced* to the basin, and point
networks / flux towers that must be *selected* by domain — collapse into a single
`ObservationSeries` (datetime / value / quality, one kind, one SI unit, UTC),
with every unit conversion pushed to the connector boundary. SYMFLUENCE pulls it
through the versioned `ObservationBackend` protocol.

## Honest coverage (this is a scaffold)

COS implements **3 of ~32** enumerated non-streamflow connectors — coverage
≈ **9%**. The architecture, the canonical contract, and the SYMFLUENCE
integration are complete; the bulk of the connector surface is unbuilt. The
three proof connectors span the structural split:

| connector | kind | structural class | auth | tested |
|---|---|---|---|---|
| `grace` | tws | gridded (basin reduction) | Earthdata | hermetic (synthetic NetCDF) |
| `snotel` | swe | point network | none | hermetic + **live-smoked** |
| `openet` | et | flux/ensemble | OpenET key | hermetic (synthetic JSON) |

See [`papers/cos_design.md`](../papers/cos_design.md) for the full design, the
roster, and `inventory/providers.yaml` for the honestly status-labeled roster of
all 32 connectors.

## What COS is NOT

- **Not streamflow** — that is [CSFS](https://github.com/DarriEy/CSFS). COS never
  duplicates a streamflow connector.
- **Not raw raster / forcing delivery** — that is CAS (rasters) and CFS (forcing).
  COS delivers reduced time series only.

## Install

```bash
pip install -e ".[dev]"          # dev: tests + xarray + pandas
pip install -e ".[gridded]"      # NetCDF connectors (GRACE etc.)
```

## Use

```bash
cos providers          # registered connectors, kind, structural class, auth
cos kinds              # canonical kinds + SI units
cos fetch snotel -s snotel:679 --start 2022-01-01 --end 2022-03-01
```

```python
import cos
from cos import ReductionSpec
from datetime import datetime, UTC

spec = ReductionSpec(domain_name="paradise", station_ids=("snotel:679",), options={"state": "WA"})
series = cos.fetch_series_sync("snotel", spec, datetime(2022,1,1,tzinfo=UTC), datetime(2022,3,1,tzinfo=UTC))
```

## SYMFLUENCE integration — and an honest wiring gap

COS registers a `CommunityObservationBackend` (contract 0.3.0) declaring its
implemented non-streamflow kinds. **Registering it does NOT yet route the
SYMFLUENCE manager flow through COS** for those kinds — the manager routes only
streamflow through the observation-backend tier today; the other kinds go through
separate per-kind evaluation paths. Wiring COS into the pipeline is a required
SYMFLUENCE-side follow-up, out of scope for this repo. See `cos_design.md` §4.

## Automated CI triage

When CI fails on `main`, a Claude Code agent (`.github/workflows/ci-autotriage.yml`) reads the
failure, posts a triage report as a commit comment, and classifies it:

| Classification | Action |
| --- | --- |
| `adapter_drift` / `data_drift` — a data provider changed; fix confined to `connectors/`/`tests/` | fix PR labeled `automerge-on-green`, **auto-merged once CI passes** |
| `contract_change` — touches `src/cos/core/` | PR labeled `needs-human-review` (a human merges) |
| `tooling_drift` — build / CI / dependency / packaging | PR labeled `needs-human-review` (a human merges) |
| `outage` / `real_bug` / `other` | report only, no code change |

**Safety:** the auto-merge workflow (`autofix-automerge.yml`) merges a PR only if its entire diff is
within `connectors/`/`tests/` — a misclassified change can never auto-merge, regardless of label.
Claude authenticates via the `ANTHROPIC_API_KEY_OAUTH` repo secret. Pause anytime with
`gh workflow disable "CI Auto-Triage" -R DarriEy/COS`.

Labels: `claude-autofix` (agent-opened) · `automerge-on-green` (drift fix, self-merges on green) ·
`needs-human-review` (needs a human).

## License

GPL-3.0-or-later (matches CSFS and SYMFLUENCE).
