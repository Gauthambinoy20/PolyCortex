# Troubleshooting

## "InsufficientBalanceError: have $X, need $Y"

Your wallet's USDC.e balance on Polygon is below the order size.
- Fund the wallet with USDC.e (bridged USDC) on Polygon.
- Verify `POLYMARKET_WALLET_ADDRESS` matches the wallet funded with USDC.

## "Insufficient allowance" errors during live trading

Polymarket's CLOB requires the exchange contract to spend your USDC. Grant an ERC-20 approval:

```python
# One-time approval (run from a signed web3 session)
usdc.approve(POLYMARKET_EXCHANGE_ADDRESS, 2**256 - 1)
```

Alternatively, use the Polymarket UI once to trigger the approval, then re-enable the bot.

## "All Polygon RPC endpoints failed"

All configured RPCs returned errors. Fix by:
1. Setting `POLYGON_RPC_URLS` to a comma-separated list of RPCs you trust.
2. Using a paid provider (Alchemy, Infura, QuickNode) for production.

## Kill-switch triggered

If trading halts with `Kill switch is active: <reason>`:

```bash
python -m polymarket_agent.main kill-switch status
python -m polymarket_agent.main kill-switch off   # only after investigating root cause
```

## Dashboard shows "DASHBOARD_PASSWORD is not set or is the default 'changeme'"

Set a strong password in `.env`:

```bash
DASHBOARD_PASSWORD=$(openssl rand -hex 24)
```

Restart the dashboard container / streamlit process.

## "Live trading from US detected"

Polymarket's terms prohibit US users. If you're in a non-restricted region and hit this
false positive (e.g., VPN), use `--i-understand-geo-risk` on the `trade --live` command.
**You remain solely responsible for legal compliance.**

## Paper vs live mode

- **Paper (default):** `dry_run: true` in `config/settings.yaml`. No real orders.
- **Live:** Requires `POLYMARKET_PRIVATE_KEY` env var, plus pass `--live` to the `trade`
  sub-command or set `dry_run: false`. Always test in paper mode first.

## Daily loss limit hit

`DailyLossTracker` halts trading until UTC midnight. Check daily P&L in the dashboard
or logs: `{"msg": "Daily loss limit hit: ..."}`. Resume automatic the next UTC day.

## Gas price too high

If `Gas price too high: N gwei (max: 200)` appears, either wait for lower gas or
raise `MAX_GAS_GWEI` in `.env` if you explicitly accept higher fees.
