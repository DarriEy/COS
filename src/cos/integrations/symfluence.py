# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""SYMFLUENCE non-streamflow-observation adapter for COS.

This module exposes COS through SYMFLUENCE's ``ObservationBackend`` protocol
(contract 0.3.0) for the **non-streamflow** observation kinds — TWS, SWE, ET,
soil moisture, snow cover, etc. Streamflow stays in CSFS; COS declares disjoint
``kinds`` and never serves streamflow.

CRITICAL HONESTY NOTE — COS IS NOT WIRED INTO THE MANAGER FLOW
==============================================================
SYMFLUENCE's manager flow routes ONLY streamflow through the
``ObservationBackend`` tier today (the Finding-1 fix the CSFS port relied on).
The other observation kinds still go through SEPARATE evaluation paths:
``evaluation.{grace,snotel,smap,...}.download`` flags → ``R.observation_handlers``
→ the per-kind evaluators.

So registering this backend makes COS *available* and *protocol-conformant*, and
makes the drop-in handlers resolvable by registry key — but it does **NOT** make
the manager flow use COS for, say, SWE or TWS. Making the manager actually use
COS requires a SYMFLUENCE-side change (generalizing the streamflow-only routing
to all obs kinds, i.e. routing the ``evaluation.<kind>`` paths through
``R.observation_backends`` under ``DATA_ACCESS: community``). That is a required
SYMFLUENCE follow-up and is OUT OF SCOPE here. Do not read COS registration as
"COS is in the evaluation pipeline" — it is not. (See ``papers/cos_design.md``
§4.)

Design of the adapter mirrors CSFS:

* the canonical-contract helpers (:func:`series_to_obs_csv_v1_frame`,
  :func:`canonical_columns_for_kind`) have no SYMFLUENCE dependency and are
  unit-tested standalone;
* the SYMFLUENCE base class is resolved defensively at import so ``import cos``
  never fails when SYMFLUENCE is absent;
* :func:`register` is the zero-arg ``symfluence.plugins`` entry-point hook;
* a self-registration tail runs ``register()`` when SYMFLUENCE is importable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from cos.core.models import KIND_TO_SYMFLUENCE_OBS_TYPE, ObservationKind, ObservationSeries

if TYPE_CHECKING:
    import pandas as pd

# Defensive resolution of the SYMFLUENCE base class.
try:  # pragma: no cover - exercised only with SYMFLUENCE present
    from symfluence.data.observation.base import BaseObservationHandler as _Base

    HAVE_SYMFLUENCE = True
except Exception:  # noqa: BLE001
    _Base = object  # type: ignore[assignment, misc]
    HAVE_SYMFLUENCE = False

#: The contract version this backend targets (hardcoded so a SYMFLUENCE-side
#: bump is detected as skew, not silently claimed compatible — as in CSFS).
TARGET_INTERFACE_VERSION = "0.3.0"

#: OBS_CSV_V1 protocol-delivery columns (UTC, SI) — the generic obs delivery.
OBS_CSV_V1_COLUMNS = ["datetime", "value", "quality_flag"]


def _require_pandas() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The SYMFLUENCE integration requires pandas. Install it with: "
            'pip install "community-observation-service[pandas]"'
        ) from exc


def canonical_columns_for_kind(kind: ObservationKind) -> list[str]:
    """SYMFLUENCE ``STANDARD_COLUMNS`` layout for a COS kind.

    Mirrors ``observation/base.py::STANDARD_COLUMNS`` so a COS series widens
    onto the right per-obs_type CSV shape. Kept here (not imported) so the
    helper has no SYMFLUENCE dependency.
    """
    table: dict[ObservationKind, list[str]] = {
        ObservationKind.SOIL_MOISTURE: ["datetime", "value", "depth_m", "quality_flag"],
        ObservationKind.SNOW_COVER: ["datetime", "sca_fraction", "quality_flag"],
        ObservationKind.SWE: ["datetime", "swe_mm", "quality_flag"],
        ObservationKind.ET: ["datetime", "et_mm_day", "quality_flag"],
        ObservationKind.TWS: ["datetime", "tws_anomaly_mm", "uncertainty_mm"],
        ObservationKind.LST: ["datetime", "lst_k", "quality_flag"],
        ObservationKind.LAI: ["datetime", "lai", "quality_flag"],
        ObservationKind.PRECIPITATION: ["datetime", "precip_mm", "quality_flag"],
        ObservationKind.GROUNDWATER: ["datetime", "groundwater_level", "quality_flag"],
    }
    return table.get(kind, ["datetime", "value", "quality_flag"])


def series_to_obs_csv_v1_frame(
    series: ObservationSeries, start: Any = None, end: Any = None,
) -> pd.DataFrame:
    """Shape a canonical series onto the contract's OBS_CSV_V1 layout.

    Columns ``datetime,value,quality_flag``: tz-naive UTC timestamps (naive ==
    UTC per the contract), value in the kind's canonical SI unit, quality flag
    passed through. Trims to the half-open UTC ``[start, end)`` window.
    """
    _require_pandas()
    import pandas as pd

    rows = [
        {"datetime": p.timestamp, "value": p.value, "quality_flag": p.quality.value}
        for p in series.points
    ]
    df = pd.DataFrame(rows, columns=OBS_CSV_V1_COLUMNS)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["value"]).sort_values("datetime").reset_index(drop=True)
        if start is not None:
            df = df[df["datetime"] >= _utc_naive(start)]
        if end is not None:
            df = df[df["datetime"] < _utc_naive(end)]
    return df[OBS_CSV_V1_COLUMNS].reset_index(drop=True)


def _utc_naive(value: Any) -> Any:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


# ---------------------------------------------------------------------------
# Capability declarations (pure, framework-free)
# ---------------------------------------------------------------------------


class ObservationCapabilitySpec(NamedTuple):
    """Pure capability facts for one COS connector served as a backend provider."""

    provider_id: str          # connector slug, e.g. "grace"
    kind: ObservationKind
    structural_class: str
    auth: frozenset[str]
    parity_grade: str | None
    notes: str


#: The providers the COS backend claims, one per IMPLEMENTED connector. Parity
#: grade is None for all three — they are unit/contract-validated and (SNOTEL)
#: live-smoked, NOT yet native-parity-gated (design §6). None means the
#: SYMFLUENCE parity gate refuses them unless ALLOW_UNGATED_BACKENDS: true.
OBSERVATION_CAPABILITIES: tuple[ObservationCapabilitySpec, ...] = (
    ObservationCapabilitySpec(
        provider_id="grace",
        kind=ObservationKind.TWS,
        structural_class="gridded",
        auth=frozenset({"earthdata"}),
        parity_grade=None,
        notes="GRACE/GRACE-FO TWS, basin-mean / nearest-cell reduction, cm→mm anomaly. "
              "Ungated: not yet compared to the native grace.py processed CSV.",
    ),
    ObservationCapabilitySpec(
        provider_id="snotel",
        kind=ObservationKind.SWE,
        structural_class="point_network",
        auth=frozenset(),
        parity_grade=None,
        notes="NRCS SNOTEL SWE, anonymous AWDB report CSV, inches→mm. Live-smoked; "
              "ungated (native snotel.py keeps inches, COS delivers mm — parity TBD).",
    ),
    ObservationCapabilitySpec(
        provider_id="openet",
        kind=ObservationKind.ET,
        structural_class="flux_tower",
        auth=frozenset({"openet"}),
        parity_grade=None,
        notes="OpenET ensemble ET, keyed API, mm/period→mm/day. Ungated.",
    ),
)


def _backend_contract() -> Any:  # pragma: no cover - symfluence-only
    from symfluence.data.backends import contract

    return contract


def _backend_errors() -> Any:  # pragma: no cover - symfluence-only
    from symfluence.data.backends import errors

    return errors


# ---------------------------------------------------------------------------
# ObservationBackend (contract 0.3.0)
# ---------------------------------------------------------------------------


class CommunityObservationBackend:
    """COS exposed through SYMFLUENCE's ObservationBackend protocol (0.3.0).

    Declares one capability per implemented COS connector with its non-streamflow
    ``kinds``. ``acquire()`` resolves the connector, runs the canonical
    fetch+reduce, and writes the OBS_CSV_V1 protocol delivery + sidecar manifest,
    window-trimmed to the half-open UTC ``[start, end)``.

    NB: registering this backend does NOT route the SYMFLUENCE manager flow
    through COS for non-streamflow kinds — see the module docstring. The backend
    is conformant and available; the manager-flow wiring is a SYMFLUENCE
    follow-up out of scope here.
    """

    name = "community-observation"
    interface_version = TARGET_INTERFACE_VERSION

    def __init__(self, config: Any = None, logger: Any = None) -> None:
        self.config = config
        self.logger = logger or _integration_logger()

    def capabilities(self) -> tuple[Any, ...]:  # pragma: no cover - symfluence-only
        contract = _backend_contract()
        return tuple(
            contract.ObservationCapability(
                provider_id=spec.provider_id,
                kinds=frozenset({KIND_TO_SYMFLUENCE_OBS_TYPE[spec.kind]}),
                station_id_scheme=f"{spec.structural_class}; see 'cos providers'",
                temporal=None,
                auth=spec.auth,
                parity_grade=spec.parity_grade,
                notes=spec.notes,
            )
            for spec in OBSERVATION_CAPABILITIES
        )

    def acquire(self, request: Any) -> Any:  # pragma: no cover - exercised by integration tests
        """Serve an ``ObservationRequest`` via the COS connector internals."""
        import cos
        from cos.core.exceptions import AuthRequiredError, ConnectorError

        contract = _backend_contract()
        errors = _backend_errors()

        provider_key = str(request.provider_id).strip().lower()
        spec_match = next((s for s in OBSERVATION_CAPABILITIES if s.provider_id == provider_key), None)
        if spec_match is None:
            served = sorted(s.provider_id for s in OBSERVATION_CAPABILITIES)
            raise errors.DatasetUnsupported(
                f"The COS observation backend does not serve provider "
                f"'{request.provider_id}' (served: {served})",
                dataset_id=request.provider_id,
                backend=self.name,
            )
        expected_kind = KIND_TO_SYMFLUENCE_OBS_TYPE[spec_match.kind]
        if request.kind != expected_kind:
            raise errors.DatasetUnsupported(
                f"COS provider '{provider_key}' serves kind {expected_kind!r}, not {request.kind!r}",
                dataset_id=request.provider_id,
                backend=self.name,
            )

        reduction_spec = self._build_reduction_spec(request)
        start, end = self._window(request)

        try:
            series_list = cos.fetch_series_sync(
                provider_key, reduction_spec, start, end,
                config=dict(request.options.get("connector_config", {})) if request.options else None,
            )
        except AuthRequiredError as exc:
            raise errors.AuthRequired(str(exc), provider=provider_key) from exc
        except ConnectorError as exc:
            raise errors.UpstreamOutage(
                f"COS connector failure acquiring '{request.provider_id}': {exc}",
                upstream=provider_key,
            ) from exc
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise errors.AcquisitionError(
                f"COS observation acquisition for '{request.provider_id}' failed: {exc}"
            ) from exc

        _require_pandas()
        target_dir = Path(request.target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for series in series_list:
            frame = series_to_obs_csv_v1_frame(series, start=start, end=end)
            safe = series.site.site_id.replace(":", "_").replace("/", "_")
            out = target_dir / f"cos_{safe}_obs_v1.csv"
            frame.to_csv(out, index=False)
            paths.append(out)

        if not paths:
            raise errors.IntegrityError(
                f"COS provider '{provider_key}' returned no series for {request.provider_id}"
            )

        result = contract.AcquisitionResult(
            paths=tuple(paths),
            schema=contract.SchemaId.OBS_CSV_V1,
            dataset_id=request.provider_id,
            backend=self.name,
            provenance={
                "integration": f"{__name__}.CommunityObservationBackend",
                "cos_version": getattr(cos, "__version__", "unknown"),
                "provider_id": provider_key,
                "kind": expected_kind,
                "acquired_at": datetime.now(UTC).isoformat(),
            },
            variables_delivered=frozenset({expected_kind}),
        )
        contract.write_manifest(result, target_dir)
        return result

    def _build_reduction_spec(self, request: Any) -> Any:  # pragma: no cover - symfluence-only
        import cos

        opts = dict(request.options or {})
        return cos.ReductionSpec(
            domain_name=str(opts.get("domain_name", "domain")),
            geometry=opts.get("geometry"),
            bbox=tuple(opts["bbox"]) if opts.get("bbox") else None,
            centroid=tuple(opts["centroid"]) if opts.get("centroid") else None,
            area_km2=opts.get("area_km2"),
            station_ids=tuple(request.station_ids or ()),
            options={k: v for k, v in opts.items()
                     if k not in {"domain_name", "geometry", "bbox", "centroid", "area_km2", "connector_config"}},
        )

    def _window(self, request: Any) -> tuple[datetime, datetime]:  # pragma: no cover - symfluence-only
        if not request.window:
            raise ValueError("COS observation acquisition requires a window [start, end)")
        s, e = request.window
        return _to_dt(s), _to_dt(e)


def _to_dt(value: Any) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _integration_logger() -> Any:
    import logging

    return logging.getLogger("cos.integrations.symfluence")


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register() -> None:
    """Register the COS observation backend with SYMFLUENCE (idempotent).

    Zero-arg hook referenced by the ``symfluence.plugins`` entry point.
    Registers :class:`CommunityObservationBackend` under ``R.observation_backends``
    (skipped on a framework without that registry). See the module docstring:
    this makes COS available and conformant, NOT wired into the manager flow for
    non-streamflow kinds.
    """
    if not HAVE_SYMFLUENCE:
        raise ImportError(
            "Cannot register the COS plugin: symfluence is not importable in this environment."
        )
    from symfluence.core.registries import R  # pragma: no cover - symfluence-only

    backends = getattr(R, "observation_backends", None)  # pragma: no cover - symfluence-only
    if backends is not None and "community-observation" not in backends:  # pragma: no cover
        backends.add("community-observation", CommunityObservationBackend)


# Self-register when SYMFLUENCE is importable (complements the entry point;
# robust to import order). register() is idempotent.
if HAVE_SYMFLUENCE:  # pragma: no cover - exercised only with SYMFLUENCE present
    import contextlib

    with contextlib.suppress(Exception):
        register()
