# Multi-Wallet Support

PolymarketTrader can route trades across multiple wallets to:

* Improve privacy — split activity across addresses
* Respect per-wallet position limits
* Spread counterparty/whale-detection risk

## Configuration

Set `POLYMARKET_PRIVATE_KEYS` as a comma-separated list.  Each entry may
optionally include an allocation weight as `:fraction`:

```bash
# Equal allocation
POLYMARKET_PRIVATE_KEYS=0xkey1,0xkey2,0xkey3

# Weighted allocation (70/30 split)
POLYMARKET_PRIVATE_KEYS=0xkey1:0.7,0xkey2:0.3
```

If the variable is absent, the single `POLYMARKET_PRIVATE_KEY` is used.

## Selection strategies

| Strategy          | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `round_robin`     | Cycles through wallets in order (default).                   |
| `min_utilization` | Selects the wallet with the lowest recorded USDC deployed.   |
| `weighted`        | Picks the wallet with the smallest `utilization/allocation`. |

## Privacy trade-offs

1. **On-chain correlation**: Polymarket exchange contracts publicly record
   every order and fill.  Using multiple wallets helps, but does not prevent
   a chain analyst from clustering them if funds flow between wallets.
2. **Funding sources**: If all wallets are funded from a single CEX account,
   cluster analysis is still trivial.  Consider different on-ramps.
3. **Timing correlations**: Rapid near-simultaneous activity across wallets
   is a strong clustering signal.  Stagger trades or use distinct cadence.
4. **Key custody**: More keys = more surface area.  Prefer HSM/KMS-backed
   signing (see `docs/SECRETS.md`).

## Monitoring

`WalletManager.as_dict()` returns per-wallet utilization and trade counts,
which the dashboard will surface in a future release.
