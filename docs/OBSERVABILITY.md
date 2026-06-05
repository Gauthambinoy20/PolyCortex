# Observability

## Metrics (Prometheus)

The agent exposes Prometheus metrics via `prometheus_client`.  Key series:

| Metric                               | Type      | Labels              |
|--------------------------------------|-----------|---------------------|
| `polymarket_orders_total`            | counter   | `direction`,`status`|
| `polymarket_order_latency_seconds`   | histogram | `stage`             |
| `polymarket_fills_total`             | counter   | `direction`         |
| `polymarket_drawdown_ratio`          | gauge     | —                   |
| `polymarket_daily_pnl_usdc`          | gauge     | —                   |
| `polymarket_kill_switch_active`      | gauge     | —                   |
| `polymarket_errors_total`            | counter   | `component`         |
| `polymarket_rpc_failover_total`      | counter   | `endpoint`          |

See `src/polymarket_agent/infra/metrics.py`.

## Tracing (OpenTelemetry)

Enable tracing by calling `setup_tracing("console")` at startup (or `otlp`
with `OTEL_EXPORTER_OTLP_ENDPOINT` to ship to Tempo/Jaeger).

```python
from polymarket_agent.infra.telemetry import setup_tracing, get_tracer

setup_tracing("otlp", endpoint="http://tempo:4317")
tracer = get_tracer()
with tracer.start_as_current_span("scan_cycle"):
    ...
```

## Dashboards

A Grafana dashboard is at `ops/grafana/polymarket-dashboard.json`.  Import
through Grafana → Dashboards → Import → Upload JSON.  The dashboard expects
the default Prometheus datasource.

Panels:

1. Order rate (orders/min by direction)
2. Fill rate
3. Order submission latency P50/P95/P99
4. Drawdown ratio
5. Daily P&L
6. Error rate by component
7. RPC failover events
8. Kill-switch state

## Alerts

Prometheus alert rules: `ops/prometheus/alerts.yml`.  Alerts fire for:

* High error rate
* Drawdown breach (`drawdown_stop`)
* Kill-switch active
* API latency exceeds threshold
* RPC failover triggered repeatedly

## Logs

Structured JSON logs via `polymarket_agent.utils.logger.setup_logging()`.
Context-vars `trace_id`, `market_id`, and `order_id` are auto-injected.
