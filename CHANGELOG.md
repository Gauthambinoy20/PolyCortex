# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-04-21

### Added — Phase 2 production hardening (A–O)
- **A. Coverage**: boosted from 68% → 88% (+500 new tests) with
  `tests/test_coverage_boost.py`, `tests/test_coverage_boost_2.py`,
  targeted tests for `ws_client`, `kill_switch`, `learner`,
  `config_validator`, `history`, `polygon_rpc`, `edge`, `gamma_client`,
  `clob_client`, `news_client`, `logger`.  `hypothesis` added for
  property-based tests.
- **B. Integration suite**: `tests/integration/test_e2e_flow.py` and
  `test_contract_shapes.py` under `pytest.mark.integration` with full
  `aioresponses` + web3 mocking.
- **C. Postgres + Alembic**: SQLAlchemy-backed `infra/db.py`,
  `alembic/` migration root, `scripts/migrate.sh`, `with-postgres`
  profile in `docker-compose.yml`, `docs/DATABASE.md`.
- **D. Multi-strategy scaffolding**: `strategies/{base,registry,
  bayesian_tcn,momentum}.py` with weighted ensemble and config-driven
  loading.
- **E. Tax export**: `reporting/tax.py` emits IRS Form 8949 + generic
  trades CSV, with dashboard export panel and `reports/` output dir.
- **F. Multi-wallet**: `infra/wallet_manager.py` with round-robin,
  min-utilization, and weighted strategies; `docs/MULTI_WALLET.md`.
- **G. Pluggable signer**: `infra/signer.py` with `EnvSigner`,
  `VaultSigner` (HashiCorp Vault KV v2), `AwsKmsSigner`; `docs/SECRETS.md`.
- **H. Observability**: `infra/telemetry.py` OpenTelemetry scaffold,
  `ops/grafana/polymarket-dashboard.json` (8 panels),
  `ops/prometheus/alerts.yml` (5 alert rules), `docs/OBSERVABILITY.md`.
- **I. Security**: `SECURITY.md`, `.github/dependabot.yml`,
  `.github/workflows/codeql.yml`, hardened `docker-compose.yml`
  (`read_only`, `cap_drop: [ALL]`, `no-new-privileges`, mem/cpu caps,
  tmpfs).  `bandit` and `pip-audit` clean.
- **J. Polymarket constants**: `constants.py` centralizes USDC address
  (6 decimals), exchange addresses (checksummed), chain IDs, signature
  types.  Contract-shape tests assert shapes used by py-clob-client.
- **K. Reliability**: `infra/circuit_breaker.py` (CLOSED/OPEN/HALF_OPEN),
  `infra/backpressure.py` (bounded asyncio queue with drop-oldest),
  `infra/pid_lock.py` (fcntl-based single-instance lock).
- **L. Performance**: `docs/PERFORMANCE.md`, `scripts/loadtest.py`
  (30s / 500-market simulation with <200 MB peak),
  `infra/memory_guard.py` tracemalloc wrapper.
- **M. Docs**: `mkdocs.yml`, `docs/index.md`, `docs/RUNBOOK.md`,
  `docs/TESTNET.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  PR and issue templates, `.github/workflows/docs.yml`.
- **N. Release engineering**: `.github/workflows/release.yml` (GHCR
  image + wheel on `v*` tag), `scripts/release.sh`, version bumped to
  `1.0.0`.
- **O. Validation**: `.env.test` with dummy values, 85%+ coverage
  enforced in CI, `ruff check && ruff format` clean.

## [0.2.0] - 2026-04-21

### Added
- **CRITICAL**: USDC balance check before every live order (`InsufficientBalanceError`)
- **CRITICAL**: Order idempotency via SQLite `submitted_orders` table
- **CRITICAL**: Demo data isolation to `data/demo_trades.db`
- **CRITICAL**: Dashboard password authentication (`DASHBOARD_PASSWORD` env var)
- **CRITICAL**: Private key validation with address derivation on startup
- **CRITICAL**: Async token-bucket rate limiter (`infra/rate_limiter.py`)
- **HIGH**: `DailyLossTracker` with UTC midnight reset in `risk/drawdown.py`
- **HIGH**: Gas price guardrail (abort if gwei > `MAX_GAS_GWEI`, default 200)
- **HIGH**: Polygon RPC failover with exponential backoff (`infra/polygon_rpc.py`)
- **HIGH**: Market resolution periodic task (`tracking/resolver.py`)
- **HIGH**: Geo-block warning for US users (requires `--i-understand-geo-risk` in live mode)
- **HIGH**: Structured JSON logging with `trace_id`, `market_id`, `order_id` context vars
- **HIGH**: `is_live_trading` assertion logged at executor entry
- Prometheus metrics exporter on port 8080 (`infra/metrics.py`)
- SQLite WAL mode on all DB connections
- `tini` as PID 1 in Docker
- `.streamlit/config.toml` with secure defaults
- `pip-audit` security audit in CI (already present; now documented)
- `html.escape()` for market titles in dashboard
- `Retry-After` header handling in API clients
- Architecture diagram (`docs/architecture.md`)
- Troubleshooting guide (`docs/TROUBLESHOOTING.md`)
- Buyer quickstart guide (`docs/BUYER_QUICKSTART.md`)
- MIT LICENSE file
- `COMMERCIAL.md` for buyers
- Financial risk disclaimer in README, CLI banner, and dashboard

### Changed
- Version bumped to `0.2.0`
- Docker healthcheck `start_period` reduced from 1200s to 300s
- Dashboard bound to `127.0.0.1` by default (use reverse proxy for external access)
- Graceful shutdown timeout now configurable via `SHUTDOWN_TIMEOUT_SEC` env var

### Security
- Private keys validated on startup (hex length, format, address derivation)
- Dashboard requires authentication before displaying any data
- API rate limits enforced to prevent account bans
- Daily loss limit halts trading automatically
