# Diagrams

All diagrams are Mermaid (render natively on GitHub) and are derived from the
actual modules under `src/polymarket_agent/`. The trading-cycle sequence diagram
lives in [architecture.md](architecture.md); this file adds the architecture,
data-flow and entity-relationship views.

## 1. Architecture

How the subsystems connect, from market data through to execution and tracking.

```mermaid
flowchart TD
    subgraph External
        GAMMA[Gamma API]
        CLOBAPI[CLOB API]
        WS[CLOB WebSocket]
        TAVILY[Tavily News]
        CLAUDE[Claude API]
        POLY[Polygon RPC]
    end

    subgraph Data["data/"]
        GC[gamma_client]
        CC[clob_client]
        WSC[ws_client]
        HIST[history]
        NEWS[news_client]
    end

    subgraph Features["features/"]
        FE[engine]
        SENT[sentiment]
    end

    subgraph Models["models/"]
        EDGE[edge — Bayesian]
        TCN[tcn]
        CAL[calibration]
        REG[regime — HMM]
    end

    subgraph Strategies["strategies/"]
        REGS[registry]
        BT[bayesian_tcn]
        MOM[momentum]
    end

    subgraph Risk["risk/"]
        SIZER[sizer — Kelly]
        DD[drawdown]
        KILL[kill_switch]
    end

    subgraph Execution["execution/"]
        SM[state_machine]
        EXEC[executor]
        PAPER[paper_engine]
        ADV[batch / twap / trailing_stop]
    end

    subgraph Tracking["tracking/"]
        TRK[tracker]
        LRN[learner]
        RES[resolver]
        REC[reconciler]
    end

    MAIN[main — orchestrator]
    DASH[dashboard/app.py]
    DB[(SQLite / Postgres)]
    METRICS[infra/metrics — Prometheus]

    GAMMA --> GC --> MAIN
    CLOBAPI --> CC --> MAIN
    WS --> WSC --> MAIN
    TAVILY --> NEWS --> SENT
    CLAUDE --> SENT --> FE
    GC --> HIST --> FE
    CC --> FE
    FE --> EDGE
    EDGE --> TCN --> CAL --> REG --> REGS
    REGS --> BT & MOM --> MAIN
    MAIN --> SIZER --> DD --> KILL --> EXEC
    EXEC --> SM
    EXEC --> PAPER
    EXEC --> ADV
    EXEC --> CLOBAPI
    EXEC --> POLY
    MAIN --> TRK --> DB
    TRK --> LRN
    MAIN --> RES --> DB
    MAIN --> REC --> DB
    EXEC --> DB
    MAIN --> METRICS
    DB --> DASH
```

## 2. Data Flow Diagram

Sources → processes → stores → sinks for one trading cycle.

```mermaid
flowchart LR
    S1[Gamma markets]:::src
    S2[CLOB order book]:::src
    S3[Price history]:::src
    S4[News + Claude sentiment]:::src

    P1{{Feature engine}}:::proc
    P2{{Edge + TCN + calibration}}:::proc
    P3{{Regime HMM}}:::proc
    P4{{Strategy ensemble}}:::proc
    P5{{Kelly sizer + drawdown gates}}:::proc
    P6{{Order executor}}:::proc
    P7{{Performance tracker + learner}}:::proc

    D1[(trades)]:::store
    D2[(snapshots)]:::store
    D3[(submitted_orders)]:::store
    D4[(sentiment cache)]:::store
    D5[/learned_weights.json/]:::store

    K1[[CLOB exchange]]:::sink
    K2[[Prometheus / dashboard]]:::sink
    K3[[Telegram alerts]]:::sink

    S1 & S2 & S3 --> P1
    S4 --> P1
    S4 -.cache.-> D4
    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    P6 --> K1
    P6 --> D3
    P6 --> D1
    P7 --> D1 & D2
    P7 --> D5
    D1 & D2 --> K2
    P5 --> K3
    P7 --> P4

    classDef src fill:#1f6feb,stroke:#0b3,color:#fff
    classDef proc fill:#2d333b,stroke:#888,color:#fff
    classDef store fill:#3fb950,stroke:#0a0,color:#000
    classDef sink fill:#8957e5,stroke:#63a,color:#fff
```

## 3. Trading-cycle sequence

See [architecture.md](architecture.md#sequence-diagram) for the full
request-lifecycle sequence diagram (env validation → scan → feature/edge/size →
risk gates → balance/gas checks → CLOB submission → persistence → metrics).

## 4. Entity-Relationship (persistence)

The persistence layer (`infra/db.py`, `tracking/tracker.py`,
`execution/executor.py`, `features/sentiment.py`). `submitted_orders` is the
idempotency ledger linked to a `trades` row via `local_order_id`.

```mermaid
erDiagram
    trades ||--o{ submitted_orders : "local_order_id"
    trades {
        int id PK
        text market_id
        text question
        text direction
        real entry_price
        real size_usdc
        text status
        real exit_price
        real pnl
        real edge_at_entry
        real estimated_prob
        real calibrated_prob
        text regime_at_entry
        text category
        text signal_breakdown
        int is_paper
        text order_type
        text bracket_state
        text local_order_id
    }
    submitted_orders {
        text client_order_id PK
        text market_id
        text side
        real size
        text status
        text created_at
    }
    snapshots {
        int id PK
        text timestamp
        real bankroll
        real total_pnl
        int open_positions_count
        real drawdown
        real brier
        real win_rate
        real profit_factor
        real open_exposure
    }
    cache {
        text market_id PK
        real timestamp
        real score
        real confidence
        text reasoning
    }
```
