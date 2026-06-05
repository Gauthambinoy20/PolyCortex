# Secrets & Signer Backends

PolymarketTrader ships with three `Signer` implementations.  All share the
same `Signer` protocol (`address`, `sign_transaction`, `sign_message`).

## 1. `EnvSigner` — development / single-machine

Reads the private key from `POLYMARKET_PRIVATE_KEY`.  Simple, but the key is
held in process memory and the env var is visible to anything that can
introspect the process environment.  **Acceptable for paper trading and
personal use only.**

```bash
export POLYMARKET_PRIVATE_KEY=0x...
```

## 2. `VaultSigner` — HashiCorp Vault (KV v2)

Fetches the key once at startup from a Vault secret.  The key still lives in
memory during the process lifetime, but Vault controls distribution and
access auditing.

```bash
export VAULT_ADDR=https://vault.example.com:8200
export VAULT_TOKEN=s.xxxxx
export VAULT_SECRET_PATH=polymarket/prod
pip install hvac
```

The secret body must contain a `private_key` field:

```hcl
vault kv put polymarket/prod private_key=0xabc...
```

## 3. `AwsKmsSigner` — AWS KMS (asymmetric secp256k1)

The private key never leaves KMS.  The signer sends message hashes and KMS
returns DER-encoded signatures.  Transaction signing requires the caller to
re-encode R/S/V with EIP-155; a hook is provided but the full implementation
is left to production deployments because it depends on your chain-ID and
legacy/EIP-1559 tx layout.

```bash
export AWS_KMS_KEY_ID=arn:aws:kms:us-east-1:123456789012:key/...
export AWS_REGION=us-east-1
pip install boto3
```

Required KMS key settings:

* Key spec: `ECC_SECG_P256K1`
* Key usage: `SIGN_VERIFY`
* IAM policy: `kms:Sign`, `kms:GetPublicKey`

## Factory

`polymarket_agent.infra.signer.build_signer_from_env()` picks the first
matching backend: Vault → KMS → Env.

## Rotating keys

See `docs/RUNBOOK.md` — section "Key rotation".
