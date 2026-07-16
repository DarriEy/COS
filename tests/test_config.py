"""Config + credential-resolution tests."""

from cos.core.config import credential_report, load_config, resolve_credentials


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


def test_multi_field_credentials_require_complete_pair(monkeypatch):
    monkeypatch.setenv("ISMN_USERNAME", "user")
    monkeypatch.delenv("ISMN_PASSWORD", raising=False)
    assert "ismn" not in resolve_credentials(frozenset({"ismn"}))
    monkeypatch.setenv("ISMN_PASSWORD", "secret")
    assert resolve_credentials(frozenset({"ismn"}))["ismn"] == {"username": "user", "password": "secret"}


def test_credential_report_never_contains_secret_values(monkeypatch):
    monkeypatch.setenv("OPENET_API_KEY", "do-not-leak")
    report = credential_report({"openet"})
    assert report[0]["resolved"] is True
    assert "do-not-leak" not in repr(report)
    assert report[0]["environment"] == ["OPENET_API_KEY"]
