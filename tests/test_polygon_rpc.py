import os

from polymarket_agent.infra.polygon_rpc import DEFAULT_RPC_URLS, get_rpc_urls


def test_defaults_when_no_env(monkeypatch):
    monkeypatch.delenv("POLYGON_RPC_URL", raising=False)
    monkeypatch.delenv("POLYGON_RPC_URLS", raising=False)
    assert get_rpc_urls() == DEFAULT_RPC_URLS


def test_single_env_prepended(monkeypatch):
    monkeypatch.delenv("POLYGON_RPC_URLS", raising=False)
    monkeypatch.setenv("POLYGON_RPC_URL", "https://custom.example/rpc")
    urls = get_rpc_urls()
    assert urls[0] == "https://custom.example/rpc"
    assert len(urls) == 1 + len(DEFAULT_RPC_URLS)


def test_list_env_parsed(monkeypatch):
    monkeypatch.setenv("POLYGON_RPC_URLS", "https://a.example, https://b.example ,,")
    urls = get_rpc_urls()
    assert urls == ["https://a.example", "https://b.example"]
    # cleanup inherited env not needed; monkeypatch scoped
    assert os.environ.get("POLYGON_RPC_URLS")
