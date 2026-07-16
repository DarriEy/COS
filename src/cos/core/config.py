# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Per-provider configuration + credential resolution.

Two concerns:

* :func:`load_config` — provider config blocks from a YAML file (mirrors CSFS).
* :func:`resolve_credentials` — the credential pass-through posture (design §3).
  Most gridded kinds need NASA Earthdata; some need CDS or an OpenET key. When
  COS is driven by SYMFLUENCE the framework owns resolution and passes a
  resolved mapping; standalone, COS falls back to ``~/.netrc`` / environment.
  The provider declares which auth ids it needs (``connector.auth``); this only
  *reads* them.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger()

_DEFAULT_PATHS = (
    Path("cos.yaml"),
    Path.home() / ".config" / "cos" / "config.yaml",
)

#: auth-provider id -> (env var for the secret, netrc machine host).
_AUTH_SOURCES: dict[str, tuple[dict[str, str], str]] = {
    "earthdata": ({"token": "EARTHDATA_TOKEN"}, "urs.earthdata.nasa.gov"),
    "cds": ({"token": "CDSAPI_KEY"}, "cds.climate.copernicus.eu"),
    "openet": ({"token": "OPENET_API_KEY"}, "openet-api.org"),
    "ameriflux": ({"user_id": "AMERIFLUX_USER_ID", "email": "AMERIFLUX_USER_EMAIL"}, "ameriflux.lbl.gov"),
    "gleam": ({"username": "GLEAM_USERNAME", "password": "GLEAM_PASSWORD"}, "gleam.eu"),
    "ismn": ({"username": "ISMN_USERNAME", "password": "ISMN_PASSWORD"}, "ismn.earth"),
    "gloh2o": ({"username": "GLOH2O_USERNAME", "password": "GLOH2O_PASSWORD"}, "gloh2o.org"),
    "cdse": ({"client_id": "CDSE_CLIENT_ID", "client_secret": "CDSE_CLIENT_SECRET"},
             "identity.dataspace.copernicus.eu"),
}


def load_config(path: Path | None = None) -> dict[str, dict]:
    """Load per-provider config blocks from a YAML file.

    Returns a dict mapping connector slugs to their config dicts. If no file is
    found, returns ``{}`` (anonymous connectors work without config).
    """
    if path is not None:
        return _read(path)
    for candidate in _DEFAULT_PATHS:
        if candidate.is_file():
            return _read(candidate)
    return {}


def _read(path: Path) -> dict[str, dict]:
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("config_load_failed", path=str(path), error=str(exc))
        return {}
    providers = data.get("providers", {})
    if not isinstance(providers, dict):
        logger.warning("config_invalid_providers_key", path=str(path))
        return {}
    return providers


def resolve_credentials(
    auth: frozenset[str],
    *,
    supplied: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Resolve the credentials a connector declared in ``connector.auth``.

    Resolution order per auth id: framework-*supplied* mapping (the
    ``CredentialContext`` pass-through) → environment variable → ``~/.netrc``.
    Returns ``{auth_id: {...secret...}}`` for the ids that resolved; absent ids
    are simply omitted (the connector decides whether that is fatal).
    """
    supplied = supplied or {}
    out: dict[str, dict[str, str]] = {}
    for auth_id in auth:
        if auth_id in supplied and supplied[auth_id]:
            out[auth_id] = dict(supplied[auth_id])
            continue
        env_fields, host = _AUTH_SOURCES.get(auth_id, ({}, ""))
        env_creds = {field: os.environ[var] for field, var in env_fields.items() if os.environ.get(var)}
        if env_fields and len(env_creds) == len(env_fields):
            out[auth_id] = env_creds
            continue
        netrc_creds = _from_netrc(host)
        if netrc_creds:
            out[auth_id] = netrc_creds
    return out


def credential_report(auth_ids: set[str] | None = None) -> list[dict[str, object]]:
    """Return secret-free credential readiness for declared auth providers."""
    ids = sorted(auth_ids if auth_ids is not None else _AUTH_SOURCES)
    resolved = resolve_credentials(frozenset(ids))
    return [
        {
            "auth_id": auth_id,
            "resolved": auth_id in resolved,
            "environment": list(_AUTH_SOURCES.get(auth_id, ({}, ""))[0].values()),
            "netrc_host": _AUTH_SOURCES.get(auth_id, ({}, ""))[1] or None,
        }
        for auth_id in ids
    ]


def _from_netrc(host: str) -> dict[str, str] | None:
    if not host:
        return None
    try:
        import netrc

        auth = netrc.netrc().authenticators(host)
    except (FileNotFoundError, OSError):
        return None
    if not auth:
        return None
    login, _account, password = auth
    return {"username": login or "", "password": password or ""}
