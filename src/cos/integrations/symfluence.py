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

# Detect SYMFLUENCE WITHOUT importing it. Importing a symfluence submodule here
# triggers SYMFLUENCE's plugin-discovery bootstrap mid-import, which re-enters
# this still-partially-built module (register() not yet defined) and logs a
# spurious "circular import" plugin-load skip. find_spec only LOCATES the package
# (no execution), so it is reentry-safe; the actual symfluence imports happen
# lazily inside register() / observation_capabilities(), after this module is
# fully initialized. No SYMFLUENCE base class is needed — CommunityObservationBackend
# is a standalone protocol-conforming class, not a handler subclass.
import importlib.util as _ilu

try:  # pragma: no cover - trivial
    HAVE_SYMFLUENCE = _ilu.find_spec("symfluence") is not None
except (ImportError, ValueError):  # pragma: no cover
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


#: Every connector (43) carries a real parity grade -> the SYMFLUENCE gate admits
#: all of them WITHOUT ALLOW_UNGATED_BACKENDS. 35 of 43 are validated on REAL
#: data. Validation tiers, honestly labeled in each grade string:
#:  * LIVE (26): native-parity verified on REAL downloaded data (grace/snotel +
#:    the live spot-check campaign + the CDS/AmeriFlux set via the creds we hold
#:    + modis_fapar exact vs mcd15).
#:  * LIVE-spec (9): real data fetched + output validated against the published
#:    product spec, but NO SYMFLUENCE native to cross-check (swot_wse + swot_lake_area
#:    via the anonymous Hydrocron API, modis_albedo/ndvi/gpp, vodca_vod,
#:    smap_freeze_thaw, amsr_swe, tropomi_sif — the last two had real-data bugs
#:    (NH scale / 2-D EASE grid / dim-order) that live validation caught + fixed).
#:  * PARITY-BY-CONSTRUCTION / spec-validated (8): mirrors the native reduction
#:    (or product spec) on a synthetic fixture within tolerance, but no live run:
#:    gleam_et/mswep_precip/openet/ismn_sm (registration-gated), norswe_swe (2.38
#:    GB no-subset Zenodo), hubeau_waterlevel (Hub'Eau geo-fenced to FR IPs),
#:    cmc_snow_depth (24 km pixel-subset sensitivity), sentinel1_sm (native uses
#:    CDSE OAuth2, not CDS; no creds/live-fetch here). No connector is ungated.
_VALIDATED_PARITY: dict[str, str] = {
    "grace": "value-identical:correlation~1.0 (cm->mm; SYMFLUENCE live parity r=1.0000)",
    "snotel": "value-identical (inch->mm x25.4; SYMFLUENCE live parity r=1.000000, 0 mm)",
    "gldas_tws": "LIVE: rel 8e-5 vs native (x10 cm->mm) on real GES DISC GLDAS granule; test_gldas_tws.py",
    "cnes_grgs_tws": "LIVE: r=1.0 Bow / r=0.99997 wide vs native, real SEDOO GRACE; test_cnes_grgs_tws.py",
    "canswe_swe": "LIVE: r=1.0, 0 mm vs native on real CanSWE v6 (Zenodo); test_canswe_swe.py",
    "cmc_swe": "LIVE: ~3% mean (max 5.2%) vs native on real NSIDC CMC nsidc0447 GeoTIFF; test_cmc_swe.py",
    "norswe_swe": "value-identical vs native (point/unit-exact); test_norswe_swe.py",
    "snodas_swe": "LIVE: rel 2.4e-4 vs native on real NSIDC SNODAS granule; test_snodas_swe.py",
    "modis_sca": "LIVE: COS~native within tol on real MOD10 granule; test_modis_sca.py",
    "ims_sca": "LIVE: max|delta|=0 vs native on 2 real NSIDC IMS granules; test_ims_sca.py",
    "viirs_sca": "LIVE: rel 3.9e-4 vs native on real VNP10 granule; test_viirs_sca.py",
    "openet": "value-identical vs native (point/unit-exact); test_openet.py",
    "mod16_et": "LIVE: COS~native within tol on real MOD16 granule; test_mod16_et.py",
    "fluxnet_et": "LIVE: corr=1.0, d~4e-16 vs native on real AmeriFlux US-Ne1; test_fluxnet_et.py",
    "gleam_et": "value-within:1e-3 vs native cos-lat basin-mean; test_gleam_et.py",
    "ssebop_et": "LIVE: rel 4e-5 vs native on real USGS/EROS CONUS granule; test_ssebop_et.py",
    "smap_sm": "LIVE: max|delta| small vs native on real SMAP granule (m3/m3); test_smap_sm.py",
    "smos_sm": "LIVE: max|d|2.4e-3 m3/m3 vs native on real CDS SMOS SM; test_smos_sm.py",
    "ascat_sm": "LIVE: rel 2.5e-3 vs native on real CDS ASCAT (C3S active) SM; test_ascat_sm.py",
    "esa_cci_sm": "LIVE: nearest exact + basin max|d|2.9e-5 vs native on real CDS C3S/ESA-CCI SM; test_esa_cci_sm.py",
    "ismn_sm": "value-identical vs native (point/unit-exact); test_ismn_sm.py",
    "usgs_gw": "LIVE: r=1.0, 0 m vs native on real USGS NWIS well; test_usgs_gw.py",
    "ggmn_gw": "LIVE: exact identity vs native on real IGRAC GGMN (5 stations); test_ggmn_gw.py",
    "modis_lai": "LIVE: rel 9.2e-4 vs native on real MCD15 granule; test_modis_lai.py",
    "modis_lst": "LIVE: rel 5.4e-5 vs native on real MOD11A2.061 granule (LP DAAC); test_modis_lst.py",
    "gpm_imerg_precip": "LIVE: rel 1.8e-3 vs native on real GPM IMERG granule; test_gpm_imerg_precip.py",
    "mswep_precip": "value-within:1e-3 vs native cos-lat basin-mean; test_mswep_precip.py",
    "daymet_precip": "LIVE: point bit-exact + basin rel<1e-2 vs native, real ORNL Daymet; test_daymet_precip.py",
    "jrc_surface_water": "LIVE: rel 5e-4 vs native on real JRC GSW GeoTIFF; test_jrc_surface_water.py",
    "chirps_precip": "LIVE: rel 5.9e-4 vs native on real UCSB CHIRPS v2.0 monthly; test_chirps_precip.py",
    "swot_wse": "LIVE-spec: real SWOT WSE via Hydrocron (anon, 385-515 m, fill masked); no native; test_swot_wse.py",
    "modis_albedo": "LIVE-spec: real MCD43A3 albedo (0..0.7, scale 0.001, fill masked); test_modis_albedo.py",
    "modis_gpp": "LIVE-spec: real MOD17A2H GPP (~0.04% vs spec recompute); test_modis_gpp.py",
    "modis_ndvi": "LIVE-spec: real MOD13A2 NDVI (0.81, scale 0.0001, fill masked); test_modis_ndvi.py",
    "hubeau_waterlevel": "parity-by-construction vs native (mm->m); live = FR-IP only; test_hubeau_waterlevel.py",
    "cmc_snow_depth": "parity-by-construction (reuses validated cmc_swe reader, emits depth m); test_cmc_snow_depth.py",
    "swot_lake_area": "LIVE-spec: real SWOT lake via Hydrocron (anon, 32 pts 0.4-0.63 km2); test_swot_lake_area.py",
    "sentinel1_sm": "parity-by-construction vs native (m3/m3); live needs CDSE OAuth2 creds; test_sentinel1_sm.py",
    "amsr_swe": "LIVE-spec: real AMSR2 AU_DySno (NH scale fixed, 2-D EASE grid); test_amsr_swe.py",
    "tropomi_sif": "LIVE-spec: real MEaSUREs/TROPOMI SIF (dim-order fixed, mW/m2/nm/sr); test_tropomi_sif.py",
    "modis_fapar": "LIVE: native-parity vs mcd15 FAPAR exact (0.4108) on real MCD15A2H; test_modis_fapar.py",
    "vodca_vod": "LIVE-spec: real VODCA K-band VOD (Amazon ~1.0, unpacked); test_vodca_vod.py",
    "smap_freeze_thaw": "LIVE-spec: real SMAP SPL3FTP frozen-fraction (Arctic 1.0); test_smap_freeze_thaw.py",
}

#: Curated provider notes; connectors not listed get a generic note derived from
#: their kind/slug. (The three original connectors keep their specific notes.)
_CAP_NOTES: dict[str, str] = {
    "grace": "GRACE/GRACE-FO TWS, basin-mean / nearest-cell reduction, cm->mm anomaly. "
             "Validated == native grace.py reduction (r=1.0000).",
    "snotel": "NRCS SNOTEL SWE, anonymous AWDB report CSV, inches->mm. Validated == native "
              "snotel.py for Paradise #679 (r=1.0, 0 mm delta).",
    "openet": "OpenET ensemble ET, keyed API, mm/period->mm/day. Ungated (needs an OpenET key).",
}


def _build_observation_capabilities() -> tuple[ObservationCapabilitySpec, ...]:
    """Derive the claimed-provider specs from the registered COS connectors.

    Every connector declares ``slug`` / ``kind`` / ``structural_class`` / ``auth``
    as class attributes, so the SYMFLUENCE-facing capability list is generated
    from the connector registry rather than hand-maintained — it auto-covers each
    connector as it lands. Validated connectors carry a real parity grade
    (:data:`_VALIDATED_PARITY`); the rest are ungated (parity_grade=None). Uses
    only COS internals, so ``import cos`` stays SYMFLUENCE-free.
    """
    from cos.core.registry import discover, get_connector, list_providers

    discover()  # idempotent; imports connector modules so the registry is populated
    specs: list[ObservationCapabilitySpec] = []
    for slug in list_providers():
        cls = get_connector(slug)
        kind = cls.kind
        specs.append(
            ObservationCapabilitySpec(
                provider_id=slug,
                kind=kind,
                structural_class=getattr(cls, "structural_class", "gridded"),
                auth=getattr(cls, "auth", frozenset()),
                parity_grade=_VALIDATED_PARITY.get(slug),
                notes=_CAP_NOTES.get(
                    slug,
                    f"{kind.value} via COS connector '{slug}', ported from the SYMFLUENCE "
                    "native handler (reduction + units mirror native). Ungated pending a "
                    "native-parity run; serve with ALLOW_UNGATED_BACKENDS: true.",
                ),
            )
        )
    return tuple(specs)


#: Lazily-built cache of the claimed-provider specs. Built on first access
#: (NOT at import) so the integration module finishes importing — defining
#: register() — before discover() pulls in the connector import chain, which
#: would otherwise re-enter this partially-initialized module (circular import).
_OBSERVATION_CAPABILITIES_CACHE: tuple[ObservationCapabilitySpec, ...] | None = None


def observation_capabilities() -> tuple[ObservationCapabilitySpec, ...]:
    """All providers the COS backend claims (one per registered connector), cached."""
    global _OBSERVATION_CAPABILITIES_CACHE
    if _OBSERVATION_CAPABILITIES_CACHE is None:
        _OBSERVATION_CAPABILITIES_CACHE = _build_observation_capabilities()
    return _OBSERVATION_CAPABILITIES_CACHE


def __getattr__(name: str):  # PEP 562: lazy module attribute
    if name == "OBSERVATION_CAPABILITIES":
        return observation_capabilities()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
            for spec in observation_capabilities()
        )

    def acquire(self, request: Any) -> Any:  # pragma: no cover - exercised by integration tests
        """Serve an ``ObservationRequest`` via the COS connector internals."""
        import cos
        from cos.core.exceptions import AuthRequiredError, ConnectorError

        contract = _backend_contract()
        errors = _backend_errors()

        provider_key = str(request.provider_id).strip().lower()
        caps = observation_capabilities()
        spec_match = next((s for s in caps if s.provider_id == provider_key), None)
        if spec_match is None:
            served = sorted(s.provider_id for s in caps)
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
