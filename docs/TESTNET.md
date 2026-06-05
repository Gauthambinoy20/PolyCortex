# Testnet (Amoy / Mumbai)

Polymarket runs on Polygon mainnet.  There is no public Polymarket deployment
on a testnet today, so "testnet" here means running the full agent stack
against mocked APIs **and** pointing web3 at Polygon's Amoy testnet for
USDC/gas interactions.

## Setup

```bash
export POLYGON_CHAIN_ID=80002                       # Amoy
export POLYGON_RPC_URL=https://rpc-amoy.polygon.technology
export POLYMARKET_PRIVATE_KEY=0xYOUR_TEST_KEY
export TRADING_MODE=paper                           # force paper engine
```

Fund your test wallet with MATIC from the [Amoy faucet](https://faucet.polygon.technology/).

## Running

```bash
python -m polymarket_agent.main trade
```

The paper engine intercepts CLOB order calls and simulates fills based on
the live order book snapshots fetched from Polymarket (mainnet) — giving a
realistic L2 experience without risking real funds.

## Testing

Integration tests under `tests/integration/` exercise the full flow with
`aioresponses` mocking all HTTP.  They are marked `integration` and skipped
by default — run with:

```bash
pytest -m integration -v
```

## Caveats

* Order signatures verified on Amoy will **not** be accepted by the Polymarket
  exchange contract on mainnet — the exchange contract address differs.
* The CLOB REST API is mainnet-only; use the paper engine for order
  submission.
* Always assert `chain_id == 137` before sending a live order.  This is
  enforced in `scripts/preflight.py`.
