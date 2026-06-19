# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Live-fetch helpers for gridded connectors.

COS gridded connectors reduce a *supplied* source file; these helpers let a
connector fetch that file itself when none is supplied, so the gridded path works
without a pre-staged granule. Two acquisition modes cover the gridded products:

* :func:`http_download` — a streamed anonymous HTTP(S) download for products
  served from an open URL (CSR / GSFC GRACE mascons, CHIRPS, Daymet, ...);
* :func:`earthaccess_granules` — an Earthdata search + download (login via
  ``~/.netrc``) for the NASA DAAC products behind Earthdata auth.

Both cache into a per-run directory (:func:`cache_dir`) and skip a re-download
when the file is already present, so repeated reductions in one workflow fetch
once. Kept dependency-light: ``httpx`` is already a COS dependency; ``earthaccess``
is an optional extra imported lazily only when an Earthdata product is fetched.
"""

from __future__ import annotations

import os
import ssl
import tempfile
from pathlib import Path

from cos.core.exceptions import AuthRequiredError, ConnectorError


def cache_dir(config: dict | None = None) -> Path:
    """The local granule cache: ``config['cache_dir']``, ``$COS_CACHE_DIR``, or a temp dir."""
    raw = (config or {}).get("cache_dir") or os.environ.get("COS_CACHE_DIR")
    path = Path(raw) if raw else Path(tempfile.gettempdir()) / "cos_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


#: User-Agent for live downloads (some open data hosts reject the default).
_USER_AGENT = "cos-live-fetch/1.0 (+https://github.com/DarriEy/COS)"


def _ssl_verify() -> ssl.SSLContext | bool:
    """OS-native verification via truststore when available, else httpx's certifi.

    truststore (the platform trust store) completes the chain for hosts that ship
    an incomplete cert chain — exactly the AIA case certifi cannot follow.
    """
    try:
        import truststore

        ctx: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return ctx
    except ImportError:
        return True  # httpx default (certifi)


def _curl_download(url: str, part: Path, timeout: float) -> bool:
    """Fallback download via the system ``curl`` (system trust store + AIA fetch).

    Some open mascon hosts (e.g. download.csr.utexas.edu) ship an incomplete cert
    chain that only an AIA-following client accepts; curl handles it. Returns True
    on a non-empty download, False when curl is unavailable or failed.
    """
    import shutil
    import subprocess

    if not shutil.which("curl"):
        return False
    result = subprocess.run(  # noqa: S603 - fixed argv, url is a connector constant/config
        ["curl", "-fsSL", "--max-time", str(int(timeout)), "-A", _USER_AGENT, "-o", str(part), url],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0 and part.exists() and part.stat().st_size > 0


def http_download(
    url: str,
    dest: Path,
    *,
    slug: str = "http",
    timeout: float = 600.0,
    chunk: int = 1 << 20,
) -> Path:
    """Stream an anonymous HTTP(S) *url* to *dest* (atomic; skips if already cached).

    A complete cached file (non-empty) is returned untouched. The download writes
    to a ``.part`` sibling and renames on success so a truncated download is never
    mistaken for a complete one. Verification prefers the OS trust store
    (truststore) and falls back to the system ``curl`` when httpx cannot verify a
    host with an incomplete cert chain. Redirects are followed.
    """
    import httpx

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    headers = {"User-Agent": _USER_AGENT}
    try:
        with httpx.stream(
            "GET", url, follow_redirects=True, timeout=timeout,
            headers=headers, verify=_ssl_verify(),
        ) as resp:
            resp.raise_for_status()
            with open(part, "wb") as fh:
                for block in resp.iter_bytes(chunk):
                    fh.write(block)
    except (httpx.HTTPError, OSError) as exc:
        part.unlink(missing_ok=True)
        if not _curl_download(url, part, timeout):
            raise ConnectorError(slug, f"live download failed for {url}: {exc}") from exc
    part.replace(dest)
    return dest


def earthaccess_granules(
    short_name: str,
    version: str | None,
    temporal: tuple[str, str],
    bbox: tuple[float, float, float, float] | None,
    dest_dir: Path,
    *,
    slug: str = "earthdata",
    count: int = 1,
) -> list[Path]:
    """Search + download Earthdata granules (login via ``~/.netrc``).

    *bbox* is COS order ``(lat_min, lon_min, lat_max, lon_max)`` and is converted
    to earthaccess's ``(lon_min, lat_min, lon_max, lat_max)``. Raises
    :class:`AuthRequiredError` when Earthdata login fails (no usable netrc), and
    :class:`ConnectorError` when the search returns nothing.
    """
    try:
        import earthaccess
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ConnectorError(
            slug, "live Earthdata fetch needs the 'earthaccess' extra "
            "(pip install 'community-observation-service[earthdata]')",
        ) from exc

    try:
        auth = earthaccess.login(strategy="netrc")
    except Exception as exc:  # noqa: BLE001 - earthaccess raises varied auth errors
        raise AuthRequiredError(slug, f"Earthdata login failed (need ~/.netrc): {exc}") from exc
    if auth is None or not getattr(auth, "authenticated", False):
        raise AuthRequiredError(slug, "Earthdata login failed (no usable ~/.netrc credentials)")

    kwargs: dict = {"short_name": short_name, "temporal": temporal, "count": count}
    if version:
        kwargs["version"] = version
    if bbox is not None:
        lat_min, lon_min, lat_max, lon_max = bbox
        kwargs["bounding_box"] = (lon_min, lat_min, lon_max, lat_max)
    results = earthaccess.search_data(**kwargs)
    if not results:
        raise ConnectorError(
            slug, f"no Earthdata granules for {short_name} {version or ''} "
            f"over {temporal} bbox={bbox}",
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = earthaccess.download(results[:count], local_path=str(dest_dir))
    return [Path(p) for p in downloaded if p]
