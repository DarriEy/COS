## What & why

## Checklist

- [ ] `ruff check src/ tests/` clean
- [ ] `mypy src/cos/` clean
- [ ] `pytest -m "not network"` green
- [ ] If a new connector: registered, serves a **non-streamflow** kind, metadata
      complete, tiered in `scheduler/tiers.py`, has a `tests/connectors/test_<slug>.py`,
      documented in `inventory/providers.yaml` (status honest — `implemented` only
      if registered)
- [ ] Canonical contract honored: series emitted in the kind's SI unit + UTC,
      conversions at the connector boundary
- [ ] No parity grade claimed without a recorded native comparison
- [ ] Gaps reported, not hidden
