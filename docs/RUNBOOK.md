# Runbook

Operational procedures for PolymarketTrader.

## Deploy

```bash
git pull
docker compose build
docker compose up -d
docker compose logs -f trader
```

## Upgrade

1. Review `CHANGELOG.md` for breaking changes.
2. Stop trader: `docker compose stop trader`.
3. Backup DB: `cp data/trades.db data/trades.db.$(date +%F)`.
4. `git pull && docker compose build && ./scripts/migrate.sh`.
5. `docker compose up -d trader`.
6. Confirm `curl localhost:8501/_stcore/health` returns 200.

## Rollback

1. Stop: `docker compose stop trader`.
2. Restore DB: `cp data/trades.db.<date> data/trades.db`.
3. Checkout prior tag: `git checkout v<previous>`.
4. Rebuild: `docker compose build && docker compose up -d trader`.

## Stuck orders

Symptoms: orders in `open` state with no fills, >1 hour old.

```bash
python -m polymarket_agent.main reconcile
```

If that fails, manually cancel via CLOB REST and update DB:

```sql
UPDATE trades SET status='canceled', exit_reason='manual_cleanup'
WHERE status='open' AND timestamp < datetime('now','-1 hour');
```

## Key rotation

1. Provision new key (Vault `vault kv put polymarket/prod private_key=0xNEW` or
   add to `POLYMARKET_PRIVATE_KEYS` first).
2. Drain old wallet: `python -m polymarket_agent.main close-all --wallet=<old>`.
3. Sweep USDC from old → new wallet.
4. Remove old key from config.
5. Restart agent.

## Kill switch

```bash
# Activate (blocks ALL trades)
echo "manual halt: $(date -Iseconds)" > data/KILL_SWITCH

# Deactivate
rm data/KILL_SWITCH
```

Programmatically: `KillSwitch().activate("reason")` /
`KillSwitch().deactivate()`.

## Backup / restore

SQLite:
```bash
sqlite3 data/trades.db ".backup data/trades.$(date +%F).bak"
```

Postgres:
```bash
pg_dump -U polymarket polymarket > backup.sql
psql -U polymarket polymarket < backup.sql
```

## Paper → Live cutover

1. Run paper for ≥ 7 days; confirm positive PnL and Brier < 0.22.
2. Fund wallet with *small* USDC (e.g. $100).
3. Set `dry_run: false` in `config/settings.yaml`, keep `bankroll` small.
4. Monitor first 24h continuously.
