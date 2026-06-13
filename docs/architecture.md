# Architecture

```
src/cos/
  core/      models (canonical contract) · registry · config (+ credentials) ·
             health · exceptions · reduce (gridded reduction kernels)
  connectors/  base + grace / snotel / openet
  cli/       the `cos` CLI
  integrations/symfluence.py   the ObservationBackend + drop-in handler helpers
  scheduler/tiers.py           connector tiering (roster-integrity parity)
inventory/providers.yaml       the honest roster
```

Mirrors CSFS: connector registry + canonical pydantic models + inventory roster +
roster-integrity tests + CLI + ObservationBackend integration. Differs in the
connector contract — `fetch_series(ReductionSpec, start, end)` returning canonical
series (one per reduced region or selected station), and the gridded
spatial-reduction kernels that have no CSFS analogue.
