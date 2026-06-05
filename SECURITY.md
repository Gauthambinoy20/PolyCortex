# Security Policy

## Supported Versions

| Version | Supported            |
|---------|----------------------|
| 1.0.x   | ✅                    |
| 0.2.x   | ⚠️ security fixes only |
| < 0.2   | ❌                    |

## Reporting a Vulnerability

Please report security issues **privately** via GitHub Security Advisories
(Security tab → Report a vulnerability) or email the maintainers.  Do not
open a public issue.

We aim to acknowledge reports within **2 business days** and ship a fix
within **30 days** for high/critical severity findings.

## Scope

In scope:

* The agent code under `src/polymarket_agent/`
* Docker images produced by this repo
* The dashboard under `dashboard/`

Out of scope:

* Polymarket's hosted APIs and exchange contracts
* Third-party dependencies (report upstream; we will pin/patch)

## Hardening checklist

- Private keys never logged (see `SensitiveFilter`)
- Kill switch (`data/KILL_SWITCH`) halts all order placement
- Per-order balance and gas checks (`OrderExecutor._assert_sufficient_balance`)
- Config validation (`utils/config_validator.py`)
- Docker image runs read-only with dropped capabilities
- Dependabot + CodeQL enabled
- `bandit` and `pip-audit` run in CI

See also [docs/SECRETS.md](docs/SECRETS.md) for signer / secret management
options (Env / Vault / AWS KMS).
