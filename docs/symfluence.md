# SYMFLUENCE integration

COS registers a `CommunityObservationBackend` (contract 0.3.0) under
`R.observation_backends`, declaring one capability per implemented connector with
its non-streamflow `kinds`. `acquire()` runs the canonical fetch+reduce and writes
the OBS_CSV_V1 protocol delivery + sidecar manifest, window-trimmed to half-open
UTC `[start, end)`.

!!! danger "COS is NOT wired into the manager flow"
    SYMFLUENCE's manager routes **only streamflow** through the
    `ObservationBackend` tier today. The other kinds go through separate per-kind
    evaluation paths (`evaluation.{grace,snotel,...}.download` →
    `R.observation_handlers` → the evaluators). Registering COS makes it
    *available and conformant* but does **not** put it in the evaluation pipeline
    for SWE/TWS/ET. Generalizing the streamflow-only routing to all obs kinds is a
    required SYMFLUENCE-side follow-up, out of scope for COS. See
    `papers/cos_design.md` §4.
