"""Pluggable transaction signer (env/Vault/KMS)."""

from __future__ import annotations

import logging
from typing import Protocol, cast, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Signer(Protocol):
    """Protocol for any component that can sign Ethereum transactions/messages."""

    @property
    def address(self) -> str: ...

    def sign_transaction(self, tx: dict) -> bytes: ...

    def sign_message(self, msg: bytes) -> bytes: ...


class EnvSigner:
    """Signs using a private key held in memory (loaded from env)."""

    def __init__(self, private_key: str) -> None:
        try:
            from eth_account import Account
        except ImportError as exc:  # pragma: no cover
            raise ImportError("eth-account required") from exc
        self._account = Account.from_key(private_key)

    @property
    def address(self) -> str:
        return str(self._account.address)

    def sign_transaction(self, tx: dict) -> bytes:
        signed = self._account.sign_transaction(tx)
        return bytes(signed.rawTransaction) if hasattr(signed, "rawTransaction") else bytes(signed.raw_transaction)

    def sign_message(self, msg: bytes) -> bytes:
        from eth_account.messages import encode_defunct

        signed = self._account.sign_message(encode_defunct(msg))
        return bytes(signed.signature)


class VaultSigner:
    """Signs using a private key stored in HashiCorp Vault KV v2.

    The secret at ``secret_path`` must contain a ``private_key`` field.
    """

    def __init__(self, vault_url: str, vault_token: str, secret_path: str) -> None:
        try:
            import hvac
        except ImportError as exc:
            raise ImportError("hvac package required: pip install hvac") from exc
        self._client = hvac.Client(url=vault_url, token=vault_token)
        self._secret_path = secret_path
        self._env_signer = self._load_signer()

    def _load_signer(self) -> EnvSigner:
        secret = self._client.secrets.kv.v2.read_secret_version(path=self._secret_path)
        private_key = secret["data"]["data"]["private_key"]
        return EnvSigner(private_key)

    @property
    def address(self) -> str:
        return self._env_signer.address

    def sign_transaction(self, tx: dict) -> bytes:
        return self._env_signer.sign_transaction(tx)

    def sign_message(self, msg: bytes) -> bytes:
        return self._env_signer.sign_message(msg)


class AwsKmsSigner:
    """Signs using an AWS KMS asymmetric key.

    Transaction signing requires EIP-155 R/S/V construction from the DER
    output and is left as a hook for production deployments.
    """

    def __init__(self, key_id: str, region: str = "us-east-1") -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 package required: pip install boto3") from exc
        self._kms = boto3.client("kms", region_name=region)
        self._key_id = key_id
        self._address: str | None = None

    @property
    def address(self) -> str:
        if self._address is None:
            raise NotImplementedError("KMS address derivation requires secp256k1 pubkey parsing")
        return self._address

    def sign_transaction(self, tx: dict) -> bytes:
        raise NotImplementedError("KMS transaction signing requires EIP-155 implementation")

    def sign_message(self, msg: bytes) -> bytes:
        resp = self._kms.sign(
            KeyId=self._key_id,
            Message=msg,
            MessageType="RAW",
            SigningAlgorithm="ECDSA_SHA_256",
        )
        return cast("bytes", resp["Signature"])


def build_signer_from_env() -> Signer:
    """Factory: construct a signer from env vars.

    Priority: ``VAULT_ADDR + VAULT_TOKEN + VAULT_SECRET_PATH`` → Vault,
    ``AWS_KMS_KEY_ID`` → KMS, else ``POLYMARKET_PRIVATE_KEY`` → env.
    """
    import os

    vault_addr = os.environ.get("VAULT_ADDR")
    vault_token = os.environ.get("VAULT_TOKEN")
    vault_path = os.environ.get("VAULT_SECRET_PATH")
    if vault_addr and vault_token and vault_path:
        return VaultSigner(vault_addr, vault_token, vault_path)

    kms_key_id = os.environ.get("AWS_KMS_KEY_ID")
    if kms_key_id:
        return AwsKmsSigner(kms_key_id, region=os.environ.get("AWS_REGION", "us-east-1"))

    pk = os.environ.get("POLYMARKET_PRIVATE_KEY")
    if not pk:
        raise RuntimeError("No signer configured. Set POLYMARKET_PRIVATE_KEY or Vault/KMS env vars.")
    return EnvSigner(pk)


__all__ = ["Signer", "EnvSigner", "VaultSigner", "AwsKmsSigner", "build_signer_from_env"]
