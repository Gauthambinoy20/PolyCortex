from polymarket_agent.infra.metrics import get_metrics


def test_singleton_returns_same_instance():
    m1 = get_metrics()
    m2 = get_metrics()
    assert m1 is m2


def test_trading_metrics_methods_do_not_crash():
    m = get_metrics()
    m.inc_orders_submitted(market_id="abc", side="BUY")
    m.inc_orders_filled(market_id="abc", side="BUY")
    m.inc_orders_failed(market_id="abc", reason="timeout")
    m.inc_api_errors(client="gamma", endpoint="/markets")
    m.set_drawdown(12.5)


def test_start_server_is_safe_when_disabled():
    m = get_metrics()
    # Toggle off and ensure start_server doesn't raise.
    saved = m._enabled
    m._enabled = False
    try:
        m.start_server(port=0)
    finally:
        m._enabled = saved
