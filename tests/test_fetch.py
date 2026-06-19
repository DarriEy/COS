# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Live-fetch helpers (cos.core.fetch): caching, curl fallback, Earthdata auth.

All hermetic — httpx / curl / earthaccess are mocked, so no network is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cos.core import fetch
from cos.core.exceptions import AuthRequiredError, ConnectorError


def test_cache_dir_prefers_config_then_env(tmp_path, monkeypatch):
    monkeypatch.delenv("COS_CACHE_DIR", raising=False)
    cfg_dir = fetch.cache_dir({"cache_dir": str(tmp_path / "cfg")})
    assert cfg_dir == tmp_path / "cfg" and cfg_dir.is_dir()

    monkeypatch.setenv("COS_CACHE_DIR", str(tmp_path / "env"))
    env_dir = fetch.cache_dir(None)
    assert env_dir == tmp_path / "env" and env_dir.is_dir()


def test_http_download_returns_cached_without_network(tmp_path):
    dest = tmp_path / "already.nc"
    dest.write_bytes(b"cached-bytes")
    # A bogus URL: if it tried the network this would fail; the cache short-circuits.
    out = fetch.http_download("https://nope.invalid/x.nc", dest, slug="t")
    assert out == dest and out.read_bytes() == b"cached-bytes"


def test_http_download_falls_back_to_curl_on_ssl_failure(tmp_path, monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("SSL: CERTIFICATE_VERIFY_FAILED")
    monkeypatch.setattr(httpx, "stream", boom)

    def fake_curl(url, part, timeout):
        Path(part).write_bytes(b"curl-bytes")  # simulate a successful curl download
        return True
    monkeypatch.setattr(fetch, "_curl_download", fake_curl)

    dest = tmp_path / "viacurl.nc"
    out = fetch.http_download("https://incomplete-chain.example/m.nc", dest, slug="t")
    assert out == dest and out.read_bytes() == b"curl-bytes"


def test_http_download_raises_when_httpx_and_curl_both_fail(tmp_path, monkeypatch):
    import httpx

    def boom(*a, **k):
        raise httpx.ConnectError("nope")
    monkeypatch.setattr(httpx, "stream", boom)
    monkeypatch.setattr(fetch, "_curl_download", lambda *a, **k: False)

    with pytest.raises(ConnectorError, match="live download failed"):
        fetch.http_download("https://x.invalid/m.nc", tmp_path / "x.nc", slug="t")


def test_earthaccess_granules_auth_error(monkeypatch, tmp_path):
    """A failed Earthdata login surfaces as AuthRequiredError (routing -> native)."""
    import sys
    import types

    fake = types.ModuleType("earthaccess")
    fake.login = lambda strategy=None: (_ for _ in ()).throw(RuntimeError("no netrc"))
    fake.search_data = lambda **k: []
    fake.download = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "earthaccess", fake)

    with pytest.raises(AuthRequiredError):
        fetch.earthaccess_granules("SHORT", "1", ("2004-01-01", "2004-02-01"), None, tmp_path, slug="t")


def test_earthaccess_granules_no_results(monkeypatch, tmp_path):
    import sys
    import types

    auth = types.SimpleNamespace(authenticated=True)
    fake = types.ModuleType("earthaccess")
    fake.login = lambda strategy=None: auth
    fake.search_data = lambda **k: []   # nothing for the window/bbox
    fake.download = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "earthaccess", fake)

    with pytest.raises(ConnectorError, match="no Earthdata granules"):
        fetch.earthaccess_granules("SHORT", "1", ("2004-01-01", "2004-02-01"), None, tmp_path, slug="t")
