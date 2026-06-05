# Database

PolymarketTrader supports both **SQLite** (default) and **PostgreSQL** (optional).

## SQLite (default)

Everything works out of the box — trades are written to `data/trades.db`.
This is ideal for single-node development and small personal deployments.

```bash
# Nothing to configure.  Start the agent:
python -m polymarket_agent.main trade
```

Apply migrations explicitly (optional, tables auto-create at runtime):

```bash
./scripts/migrate.sh
```

## PostgreSQL

For multi-node or durable deployments, set `DATABASE_URL`:

```bash
export DATABASE_URL=postgresql+asyncpg://polymarket:polymarket@localhost:5432/polymarket
./scripts/migrate.sh
python -m polymarket_agent.main trade
```

### Docker Compose

Start Postgres alongside the trader with the `with-postgres` profile:

```bash
docker compose --profile with-postgres up -d postgres
export DATABASE_URL=postgresql://polymarket:polymarket@localhost:5432/polymarket
./scripts/migrate.sh
```

## Migrations

Alembic is used for schema migrations.  The initial migration lives at
`alembic/versions/0001_initial.py`.

```bash
alembic upgrade head          # apply migrations
alembic revision -m "change"  # create a new migration
alembic downgrade -1          # rollback one step
```

## Backup & restore

- **SQLite**: copy `data/trades.db` while the agent is stopped, or use
  `sqlite3 data/trades.db ".backup backup.db"` while running.
- **Postgres**: `pg_dump polymarket > backup.sql` / `psql polymarket < backup.sql`.
