# Connector roster

The canonical, generated roster lives in `inventory/providers.yaml`: **50
implemented connectors across 20 kinds**. It is generated from connector class
metadata, so a connector cannot be registered without appearing in the roster.

Validation is deliberately a second axis:

- `live-parity`: real provider data compared with the native SYMFLUENCE path;
- `live-spec`: real provider data checked against the published product spec
  where no native handler exists;
- `parity-by-construction`: hermetic comparison of units, masking, temporal
  semantics, and reduction behavior, awaiting an operational live run.

Use `cos providers` for the roster and `cos validation` for validation tiers,
auth requirements, data-license posture, and the evidence grade. JSON output is
available from both automation and release jobs with
`cos validation --json-output`.
