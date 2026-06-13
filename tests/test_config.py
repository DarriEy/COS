"""Config + credential-resolution tests."""

from cos.core.config import load_config, resolve_credentials


def test_load_config_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_config() == {}


def test_load_config_reads_providers(tmp_path):
    p = tmp_path / "cos.yaml"
    p.write_text("providers:\n  openet:\n    interval: monthly\n")
    cfg = load_config(p)
    assert cfg == {"openet": {"interval": "monthly"}}


def test_load_config_bad_providers_key(tmp_path):
    p = tmp_path / "cos.yaml"
    p.write_text("providers: not-a-mapping\n")
    assert load_config(p) == {}


def test_resolve_credentials_prefers_supplied():
    out = resolve_credentials(
        frozenset({"earthdata"}),
        supplied={"earthdata": {"token": "abc"}},
    )
    assert out["earthdata"]["token"] == "abc"


def test_resolve_credentials_from_env(monkeypatch):
    monkeypatch.setenv("OPENET_API_KEY", "k123")
    out = resolve_credentials(frozenset({"openet"}))
    assert out["openet"]["token"] == "k123"


def test_resolve_credentials_absent_is_omitted(monkeypatch):
    monkeypatch.delenv("OPENET_API_KEY", raising=False)
    # No env, no netrc machine for a made-up id -> simply absent.
    out = resolve_credentials(frozenset({"openet"}))
    assert "openet" not in out or out == {}
