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
_AUTH_SOURCES: dict[str, tuple[str, str]] = {
    "earthdata": ("EARTHDATA_TOKEN", "urs.earthdata.nasa.gov"),
    "cds": ("CDSAPI_KEY", "cds.climate.copernicus.eu"),
    "openet": ("OPENET_API_KEY", "openet-api.org"),
    "ameriflux": ("AMERIFLUX_API_KEY", "ameriflux.lbl.gov"),
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
        env_var, host = _AUTH_SOURCES.get(auth_id, ("", ""))
        token = os.environ.get(env_var) if env_var else None
        if token:
            out[auth_id] = {"token": token}
            continue
        netrc_creds = _from_netrc(host)
        if netrc_creds:
            out[auth_id] = netrc_creds
    return out


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
