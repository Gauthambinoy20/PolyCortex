# Buyer Quickstart

Welcome! This guide gets you from zero to running the PolymarketTrader safely
in paper mode, then optionally switching to live trading.

## 1. Prerequisites

- Python 3.11 or 3.12
- Docker + Docker Compose (optional but recommended)
- A Polymarket-compatible Polygon wallet with USDC.e
  (only required for live mode)

## 2. Install

```bash
git clone <your-repo-url> polymarket-trader
cd polymarket-trader
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" -c constraints.txt
```

## 3. Configure environment

```bash
cp .env.example .env
# Edit .env — at a minimum set:
#   DASHBOARD_PASSWORD=<long random string>
#   POLYGON_RPC_URL=<your private RPC>   # public RPCs rate-limit heavily
#   ANTHROPIC_API_KEY=<optional for sentiment>
```

For **live mode**, additionally:
```
POLYMARKET_PRIVATE_KEY=<64-char hex, no 0x prefix required>
POLYMARKET_WALLET_ADDRESS=<0x... wallet for balance checks>
```

## 4. Run in paper mode (always do this first)

```bash
python -m polymarket_agent.main trade --dry-run --once
```

Expected: a single cycle runs, trades print to stdout with `is_paper=True`.
No on-chain calls are made.

## 5. Start the dashboard

```bash
streamlit run dashboard/app.py
# → open http://localhost:8501 and enter DASHBOARD_PASSWORD
```

## 6. Seed demo data (optional)

```bash
python -m polymarket_agent.main demo-seed
# writes to data/demo_trades.db, does NOT touch live trades.db
```

## 7. Go live (only after multiple days of profitable paper runs)

```bash
python -m polymarket_agent.main trade --live
```

The CLI will:
1. Print the financial disclaimer.
2. Validate your env (including `POLYMARKET_PRIVATE_KEY`).
3. Run a geo-check — if you appear to be in the US, add `--i-understand-geo-risk`.
4. Check USDC balance and Polygon gas price before each order.

## 8. Production deployment (Docker)

```bash
docker compose up -d
```

The `trader` container runs the bot; `dashboard` runs Streamlit bound to `127.0.0.1`.
Put an authenticating reverse proxy (Caddy, nginx + OAuth2 proxy, Cloudflare Zero
Trust) in front of port 8501 before exposing publicly.

## 9. Monitoring

- Prometheus metrics: `http://<host>:8080/metrics`
- Logs: `data/logs/agent.log` (rotating, 10 MB × 5)
- SQLite DB: `data/trades.db`

## 10. Emergency stop

```bash
python -m polymarket_agent.main kill-switch on --reason "manual halt"
```

All new orders are rejected until you run `kill-switch off`.
