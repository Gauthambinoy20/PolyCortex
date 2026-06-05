# Architecture

## System Overview

PolymarketTrader is an autonomous neural trading agent for Polymarket prediction markets.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant Main as main.py
    participant Env as EnvValidator
    participant Geo as GeoCheck
    participant Gamma as GammaClient
    participant Features as FeatureEngine
    participant Edge as EdgeDetector
    participant Sizer as PositionSizer
    participant Risk as DrawdownController
    participant Exec as OrderExecutor
    participant CLOB as ClobClient
    participant DB as SQLite (trades.db)
    participant Metrics as Prometheus

    Main->>Env: validate_env()
    Env-->>Main: ValidationResult
    Main->>Geo: _check_geo_restriction()
    loop Every 60s
        Main->>Gamma: get_markets()
        Gamma-->>Main: markets[]
        Main->>Features: compute_features(market)
        Features-->>Main: feature_vector
        Main->>Edge: detect_edge(features)
        Edge-->>Main: EdgeResult
        Main->>Sizer: calculate_position(edge)
        Sizer-->>Main: size_usdc
        Main->>Risk: get_multiplier(bankroll)
        Risk-->>Main: multiplier
        Main->>Exec: place_order(order)
        Exec->>Exec: _assert_sufficient_balance()
        Exec->>Exec: _check_gas_price()
        Exec->>DB: persist submitted_orders
        Exec->>CLOB: post_order()
        CLOB-->>Exec: order_id
        Exec->>Metrics: inc_orders_submitted()
        Exec->>DB: record trade
    end
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Entry Point | `src/polymarket_agent/main.py` | Orchestrates the trading loop |
| Feature Engine | `src/polymarket_agent/features/engine.py` | Order book + sentiment features |
| Edge Detector | `src/polymarket_agent/models/edge.py` | Identifies mispriced markets |
| Position Sizer | `src/polymarket_agent/risk/sizer.py` | Kelly-based position sizing |
| Drawdown Controller | `src/polymarket_agent/risk/drawdown.py` | Risk management + daily loss limits |
| Order Executor | `src/polymarket_agent/execution/executor.py` | Order lifecycle + balance/gas checks |
| CLOB Client | `src/polymarket_agent/data/clob_client.py` | Polymarket order book API |
| Gamma Client | `src/polymarket_agent/data/gamma_client.py` | Polymarket markets API |
| Market Resolver | `src/polymarket_agent/tracking/resolver.py` | Closes resolved positions |
| Rate Limiter | `src/polymarket_agent/infra/rate_limiter.py` | Token-bucket for API calls |
| Polygon RPC | `src/polymarket_agent/infra/polygon_rpc.py` | RPC failover with backoff |
| Dashboard | `dashboard/app.py` | Streamlit monitoring UI (password-protected) |
| Metrics | `src/polymarket_agent/infra/metrics.py` | Prometheus exporter |
