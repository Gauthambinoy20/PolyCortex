"""Tests for infra.signer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from eth_account import Account

from polymarket_agent.infra.signer import (
    AwsKmsSigner,
    EnvSigner,
    Signer,
    VaultSigner,
    build_signer_from_env,
)


@pytest.fixture
def pk() -> str:
    return Account.create("test-signer").key.hex()


def test_env_signer_roundtrip(pk):
    s = EnvSigner(pk)
    assert s.address.startswith("0x")
    assert len(s.address) == 42
    # Sign a simple message
    sig = s.sign_message(b"hello")
    assert isinstance(sig, bytes)
    assert len(sig) == 65  # standard secp256k1 signature length


def test_env_signer_satisfies_protocol(pk):
    s = EnvSigner(pk)
    assert isinstance(s, Signer)


def test_env_signer_sign_transaction(pk):
    s = EnvSigner(pk)
    tx = {
        "nonce": 0,
        "gasPrice": 1000000000,
        "gas": 21000,
        "to": "0x0000000000000000000000000000000000000001",
        "value": 1,
        "data": b"",
        "chainId": 137,
    }
    raw = s.sign_transaction(tx)
    assert isinstance(raw, bytes)
    assert len(raw) > 0


def test_vault_signer_loads_from_vault(pk):
    pytest.importorskip("hvac")  # optional "secrets" extra
    fake_client = MagicMock()
    fake_client.secrets.kv.v2.read_secret_version.return_value = {"data": {"data": {"private_key": pk}}}
    with patch("hvac.Client", return_value=fake_client):
        vs = VaultSigner("http://vault", "tok", "secret/path")
    assert vs.address.startswith("0x")
    sig = vs.sign_message(b"x")
    assert isinstance(sig, bytes)


def test_vault_signer_missing_hvac(pk):
    with patch.dict("sys.modules", {"hvac": None}):
        # hvac is actually importable so directly patching the import inside is harder.
        # Instead verify missing by reloading without hvac
        import polymarket_agent.infra.signer as signer_mod

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "hvac":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import), pytest.raises(ImportError):
            signer_mod.VaultSigner("http://vault", "tok", "secret/path")


def test_aws_kms_signer_sign_message():
    pytest.importorskip("boto3")  # optional "secrets" extra
    fake_kms = MagicMock()
    fake_kms.sign.return_value = {"Signature": b"DER-encoded-sig"}
    with patch("boto3.client", return_value=fake_kms):
        ks = AwsKmsSigner("key-id")
    sig = ks.sign_message(b"msg")
    assert sig == b"DER-encoded-sig"
    fake_kms.sign.assert_called_once()


def test_aws_kms_address_not_implemented():
    pytest.importorskip("boto3")  # optional "secrets" extra
    fake_kms = MagicMock()
    with patch("boto3.client", return_value=fake_kms):
        ks = AwsKmsSigner("key-id")
    with pytest.raises(NotImplementedError):
        _ = ks.address


def test_aws_kms_sign_tx_not_implemented():
    pytest.importorskip("boto3")  # optional "secrets" extra
    fake_kms = MagicMock()
    with patch("boto3.client", return_value=fake_kms):
        ks = AwsKmsSigner("key-id")
    with pytest.raises(NotImplementedError):
        ks.sign_transaction({})


def test_build_signer_env(monkeypatch, pk):
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    monkeypatch.delenv("VAULT_SECRET_PATH", raising=False)
    monkeypatch.delenv("AWS_KMS_KEY_ID", raising=False)
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", pk)
    s = build_signer_from_env()
    assert isinstance(s, EnvSigner)


def test_build_signer_none_configured(monkeypatch):
    for k in ("VAULT_ADDR", "VAULT_TOKEN", "VAULT_SECRET_PATH", "AWS_KMS_KEY_ID", "POLYMARKET_PRIVATE_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        build_signer_from_env()


def test_build_signer_vault(monkeypatch, pk):
    pytest.importorskip("hvac")  # optional "secrets" extra
    monkeypatch.setenv("VAULT_ADDR", "http://vault")
    monkeypatch.setenv("VAULT_TOKEN", "tok")
    monkeypatch.setenv("VAULT_SECRET_PATH", "path")
    fake_client = MagicMock()
    fake_client.secrets.kv.v2.read_secret_version.return_value = {"data": {"data": {"private_key": pk}}}
    with patch("hvac.Client", return_value=fake_client):
        s = build_signer_from_env()
    assert isinstance(s, VaultSigner)


def test_build_signer_kms(monkeypatch):
    pytest.importorskip("boto3")  # optional "secrets" extra
    for k in ("VAULT_ADDR", "VAULT_TOKEN", "VAULT_SECRET_PATH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_KMS_KEY_ID", "kid")
    with patch("boto3.client", return_value=MagicMock()):
        s = build_signer_from_env()
    assert isinstance(s, AwsKmsSigner)
