# Roadmap

## v0.2.0 — Production Hardening (shipped)

See `CHANGELOG.md` for the full list.

## v1.1 — Deferred items

The following were evaluated during the v0.2.0 hardening pass but deferred:

- **Automatic ERC-20 allowance top-up**: We currently require buyers to grant USDC
  allowance to the Polymarket exchange contract manually (see TROUBLESHOOTING.md).
  A startup `ApprovalChecker` that detects and submits approval automatically is
  planned — it was deferred to avoid an eth_account / web3.py dependency tree bump.
- **py-clob-client `redeem()` integration**: `MarketResolver` stubs final redemption.
  When the upstream SDK exposes a redeem method, wire it in.
- **Bracket order atomicity**: If stop-loss placement fails we should cancel the
  take-profit too (and vice versa). Current implementation logs and continues.
- **Position sizer awareness of open order notional**: `UnifiedPositionSizer`
  should subtract open-order USDC from available bankroll before sizing.
- **Typed `uint256` overflow handling** in `eth_call` balance parsing.
- **Structured trace_id propagation** across `asyncio.create_task` boundaries
  using `contextvars.copy_context()`.
- **Prometheus histogram** for order latency (currently Counter/Gauge only).
- **Dashboard**: delegate to SSO / OIDC instead of a shared password.
- **Coverage gate**: raise from 60% → 80% once new modules have complete tests.

## Beyond v1.1

- Multi-exchange support (e.g., Augur, Kalshi where legally accessible).
- On-chain settlement verification via Polygonscan indexer.
- Portfolio-level VaR instead of per-trade Kelly sizing.
- HMM regime detector auto-retrain scheduler (already present — harden and expose).

## v1.0.0 — Phase 2 Production Hardening (shipped)

- Coverage 88%, 490+ tests, integration suite
- Postgres support + Alembic migrations
- Multi-strategy registry (bayesian_tcn, momentum) with weighted ensembling
- Tax export (IRS Form 8949, generic CSV)
- Multi-wallet manager with allocation strategies
- Pluggable signer (Env / HashiCorp Vault / AWS KMS)
- OpenTelemetry scaffold, Grafana dashboard, Prometheus alerts
- SECURITY.md, Dependabot, CodeQL, hardened Docker (read-only, cap_drop, resource limits)
- Centralized Polymarket constants + contract-shape tests
- Reliability: circuit breaker, bounded backpressure queue, fcntl PID lock
- Performance docs + loadtest script + tracemalloc guard
- MkDocs site, runbook, CONTRIBUTING, Code of Conduct, issue/PR templates
- Release workflow (GHCR + wheel on v* tags)

## Deferred / future work

- Full live Vault-backed key rotation automation (signer is in place; runbook documents process)
- AWS KMS signer — transaction-signing path requires secp256k1 DER→RSV conversion; scaffold + tests in place, live path deferred
- Cross-chain support (Solana, zkSync) — architecture ready via `constants.py` but not wired
- Paper-to-live A/B testing harness
