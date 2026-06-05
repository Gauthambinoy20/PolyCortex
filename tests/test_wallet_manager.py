"""Tests for infra.wallet_manager."""

from __future__ import annotations

import pytest
from eth_account import Account

from polymarket_agent.infra.wallet_manager import WalletManager


def _mkkey(seed: int) -> str:
    return Account.create(f"seed-{seed}").key.hex()


def test_init_single_wallet():
    wm = WalletManager([_mkkey(1)])
    assert len(wm) == 1
    assert wm.wallets[0].allocation == 1.0


def test_init_multi_wallet_equal():
    wm = WalletManager([_mkkey(1), _mkkey(2), _mkkey(3)])
    assert len(wm) == 3
    assert all(abs(w.allocation - 1 / 3) < 1e-9 for w in wm.wallets)


def test_init_allocations_normalized():
    wm = WalletManager([_mkkey(1), _mkkey(2)], allocations=[2, 8])
    assert wm.wallets[0].allocation == pytest.approx(0.2)
    assert wm.wallets[1].allocation == pytest.approx(0.8)


def test_init_mismatched_allocations():
    with pytest.raises(ValueError):
        WalletManager([_mkkey(1), _mkkey(2)], allocations=[1.0])


def test_init_zero_allocation():
    with pytest.raises(ValueError):
        WalletManager([_mkkey(1)], allocations=[0.0])


def test_init_empty_keys():
    with pytest.raises(ValueError):
        WalletManager([])


def test_round_robin_selection():
    keys = [_mkkey(i) for i in range(3)]
    wm = WalletManager(keys)
    picks = [wm.get_wallet("round_robin") for _ in range(6)]
    assert picks[0] == picks[3]
    assert picks[1] == picks[4]
    assert len(set(picks)) == 3


def test_min_utilization_selection():
    wm = WalletManager([_mkkey(1), _mkkey(2)])
    addrs = wm.addresses
    wm.record_utilization(addrs[0], 100.0)
    chosen = wm.get_wallet("min_utilization")
    assert chosen == addrs[1]


def test_weighted_selection():
    wm = WalletManager([_mkkey(1), _mkkey(2)], allocations=[0.2, 0.8])
    # High-allocation wallet preferred until utilization catches up
    first = wm.get_wallet("weighted")
    wm.record_utilization(first, 100.0)
    # Now utilization/allocation ratios should favor the other wallet
    second = wm.get_wallet("weighted")
    assert second != first or wm.get_wallet("weighted") != first


def test_unknown_strategy():
    wm = WalletManager([_mkkey(1)])
    with pytest.raises(ValueError):
        wm.get_wallet("nonsense")


def test_get_private_key():
    pk = _mkkey(1)
    wm = WalletManager([pk])
    addr = wm.addresses[0]
    assert wm.get_private_key(addr) == pk
    with pytest.raises(KeyError):
        wm.get_private_key("0x0000000000000000000000000000000000000000")


def test_from_env_plain(monkeypatch):
    keys = [_mkkey(1), _mkkey(2)]
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEYS", ",".join(keys))
    monkeypatch.delenv("POLYMARKET_PRIVATE_KEY", raising=False)
    wm = WalletManager.from_env()
    assert len(wm) == 2


def test_from_env_weighted(monkeypatch):
    keys = [_mkkey(1), _mkkey(2)]
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEYS", f"{keys[0]}:3,{keys[1]}:1")
    wm = WalletManager.from_env()
    assert wm.wallets[0].allocation == pytest.approx(0.75)


def test_from_env_fallback_single(monkeypatch):
    monkeypatch.delenv("POLYMARKET_PRIVATE_KEYS", raising=False)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", _mkkey(1))
    wm = WalletManager.from_env()
    assert len(wm) == 1


def test_from_env_missing(monkeypatch):
    monkeypatch.delenv("POLYMARKET_PRIVATE_KEYS", raising=False)
    monkeypatch.delenv("POLYMARKET_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError):
        WalletManager.from_env()


def test_as_dict_and_reset():
    wm = WalletManager([_mkkey(1)])
    addr = wm.addresses[0]
    wm.record_utilization(addr, 50.0)
    d = wm.as_dict()
    assert d[0]["utilization"] == 50.0
    assert d[0]["trade_count"] == 1
    wm.reset_utilization()
    assert wm.wallets[0].utilization == 0.0
