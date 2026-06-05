"""Entry point for the Polymarket Neural Trading Agent."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import signal
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import pandas as pd
import yaml

from polymarket_agent.data.clob_client import ClobClient
from polymarket_agent.data.gamma_client import GammaClient
from polymarket_agent.data.history import HistoryStore
from polymarket_agent.data.news_client import NewsClient
from polymarket_agent.demo import seed_demo_workspace
from polymarket_agent.events.bus import EventBus
from polymarket_agent.execution.arbitrage import ArbitrageScanner
from polymarket_agent.execution.executor import BracketSpec, OrderExecutor
from polymarket_agent.features.engine import PolymarketFeatureEngine
from polymarket_agent.features.sentiment import SentimentAnalyzer
from polymarket_agent.infra.env_validator import validate_env
from polymarket_agent.infra.health import HealthChecker
from polymarket_agent.infra.scheduler import IntervalScheduler
from polymarket_agent.infra.shutdown import GracefulShutdown
from polymarket_agent.models.calibration import CalibrationFitMetrics, ProbabilityCalibrator
from polymarket_agent.models.edge import UnifiedEdgeDetector
from polymarket_agent.models.regime import RegimeDetector
from polymarket_agent.models.tcn import PolymarketEnsemble
from polymarket_agent.risk.drawdown import DrawdownController
from polymarket_agent.risk.kill_switch import KillSwitch
from polymarket_agent.risk.sizer import UnifiedPositionSizer
from polymarket_agent.tracking.learner import SelfAdjuster
from polymarket_agent.tracking.reconciler import PositionReconciler
from polymarket_agent.tracking.tracker import PerformanceTracker
from polymarket_agent.types import Event, EventType
from polymarket_agent.utils.alerting import AlertRouter
from polymarket_agent.utils.config_validator import validate_config, validate_environment
from polymarket_agent.utils.heartbeat import HeartbeatMonitor
from polymarket_agent.utils.telegram import TelegramNotifier

logger = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


DISCLAIMER_BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  EXPERIMENTAL SOFTWARE — FINANCIAL RISK DISCLAIMER                       ║
║                                                                              ║
║  Trading prediction markets carries substantial risk of loss.                ║
║  This software is provided AS-IS with no warranty.                           ║
║  You may lose all funds. Not financial advice.                               ║
║  You are solely responsible for compliance with your local laws              ║
║  (including Polymarket's geo-restrictions on US users).                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


async def _check_geo_restriction(live_mode: bool, auto_accept: bool = False) -> None:
    """Check if running from a geo-restricted region (US) and warn accordingly."""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get("https://ipapi.co/json/", timeout=aiohttp.ClientTimeout(total=5)) as resp,
        ):
            if resp.status == 200:
                data = await resp.json()
                country = data.get("country_code", "")
                if country == "US":
                    logger.warning(
                        "\n\033[91m" + "=" * 70 + "\n  ⚠️  GEO-RESTRICTION WARNING\n"
                        "  You appear to be in the United States.\n"
                        "  Polymarket restricts access for US users per their Terms of Service.\n"
                        "  You are SOLELY responsible for compliance with applicable laws.\n" + "=" * 70 + "\033[0m"
                    )
                    if live_mode and not auto_accept:
                        raise SystemExit("Live trading from US detected. Use --i-understand-geo-risk to override.")
    except SystemExit:
        raise
    except Exception as exc:
        logger.debug("Geo check failed (non-fatal): %s", exc)


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()

    project_path = (_PROJECT_ROOT / path).resolve()
    if project_path.exists():
        return project_path

    return path.resolve()


class PolymarketTradingAgent:
    """Unified trading agent with error recovery, health checks, and graceful shutdown."""

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        self.project_root = _PROJECT_ROOT
        self.config_path = _resolve_config_path(config_path)

        with open(self.config_path) as f:
            self.config: dict = yaml.safe_load(f)

        # Validate config at startup
        warnings = validate_config(self.config)
        for w in warnings:
            logger.warning("Config warning: %s", w)
        env_issues = validate_environment()
        for issue in env_issues:
            logger.warning("Env issue: %s", issue)

        # Data clients
        self.gamma = GammaClient(
            base_url=self.config.get("gamma_api_url", "https://gamma-api.polymarket.com"),
        )
        self.clob = ClobClient(
            base_url=self.config.get("clob_api_url", "https://clob.polymarket.com"),
        )
        self.news = NewsClient(
            cache_minutes=self.config.get("sentiment_cache_minutes", 30),
        )
        self.history = HistoryStore(str(self._project_path("data", "price_history")))

        # Feature & analysis
        self.features = PolymarketFeatureEngine()
        self.sentiment = SentimentAnalyzer(
            cache_db=str(self._project_path("data", "sentiment_cache.db")),
            cache_minutes=self.config.get("sentiment_cache_minutes", 30),
            model=self.config.get("sentiment_model", "claude-haiku-4-5-20251001"),
        )
        self.regime = RegimeDetector()

        # Edge & risk
        self._bankroll: float = float(self.config.get("bankroll", 100))
        self.edge_detector = UnifiedEdgeDetector(
            self.config,
            sentiment_analyzer=self.sentiment,
        )
        self.calibrator = ProbabilityCalibrator(
            path=str(self._project_path("data", "models", "probability_calibration.json")),
            enabled=bool(self.config.get("enable_probability_calibration", True)),
            method=str(self.config.get("probability_calibration_method", "isotonic")),
            min_samples=int(self.config.get("probability_calibration_min_samples", 30)),
            refit_interval=int(self.config.get("probability_calibration_refit_interval", 25)),
        )
        self.edge_detector.set_probability_calibrator(self.calibrator)
        self.sizer = UnifiedPositionSizer(self.config)
        self.drawdown = DrawdownController(
            reduce_at=self.config.get("drawdown_reduce", 0.08),
            stop_at=self.config.get("drawdown_stop", 0.15),
            emergency_at=self.config.get("drawdown_emergency", 0.20),
            initial_bankroll=self._bankroll,
        )
        self.drawdown.daily_loss_limit_usdc = self.config.get("daily_loss_limit_usdc")
        self.kill_switch = KillSwitch(
            path=str(self._project_path("data", "KILL_SWITCH")),
        )

        # Execution
        self.executor = OrderExecutor(self.config)
        self.arbitrage = ArbitrageScanner()

        # Tracking & learning
        self.tracker = PerformanceTracker(str(self._project_path("data", "trades.db")))
        self.learner = SelfAdjuster(
            self.config,
            weights_path=str(self._project_path("data", "learned_weights.json")),
        )
        self._apply_learned_settings()

        # Monitoring & alerting
        self._telegram = TelegramNotifier()
        self._alert_router = AlertRouter(
            alerts_path=self._project_path("data", "alerts.jsonl"),
            telegram=self._telegram,
        )
        self._heartbeat = HeartbeatMonitor(
            heartbeat_path=self._project_path("data", "heartbeat.json"),
            max_stale_minutes=int(self.config.get("scan_interval_minutes", 15) * 3),
        )
        self._reconciler = PositionReconciler(
            stale_hours=float(self.config.get("stale_position_hours", 48)),
            auto_close_phantoms=bool(self.config.get("auto_close_phantoms", False)),
        )

        # Event bus for internal component communication
        self._event_bus = EventBus()

        # Graceful shutdown handler
        self._shutdown = GracefulShutdown(timeout=30.0)
        self._shutdown.register(self._shutdown_cancel_orders, name="cancel_orders")
        self._shutdown.register(self._shutdown_save_state, name="save_state")
        self._shutdown.register(self._shutdown_close_clients, name="close_clients")

        # Health checker
        self._health = HealthChecker()
        self._health.register_check("gamma_api", self._check_gamma_health)
        self._health.register_check("tracker_db", self._check_tracker_health)

        # TCN model (optional)
        self.tcn_model: PolymarketEnsemble | None = None
        try:
            import torch

            model_path = self._project_path("data", "models", "tcn_latest.pt")
            if os.path.exists(model_path):
                self.tcn_model = PolymarketEnsemble(n_features=10)
                self.tcn_model.load_state_dict(torch.load(model_path, weights_only=True))
                self.tcn_model.eval()
                self.edge_detector.set_tcn_model(self.tcn_model)
                logger.info("Loaded TCN model from %s", model_path)
        except Exception as exc:
            logger.warning("Could not load TCN model: %s", exc)

        # Regime model (optional)
        try:
            regime_path = self._project_path("data", "models", "regime_latest.pkl")
            if os.path.exists(regime_path):
                self.regime.load(str(regime_path))
                logger.info("Loaded regime model from %s", regime_path)
        except Exception as exc:
            logger.warning("Could not load regime model: %s", exc)

        if self.calibrator.load():
            logger.info(
                "Loaded probability calibrator from %s (%d samples)",
                self.calibrator.path,
                self.calibrator.sample_count,
            )

        # Periodic-job scheduler (TCN retrain, regime refit, etc.)
        self.scheduler = IntervalScheduler(self._project_path("data", "scheduler_state.json"))
        self._tcn_retrain_days = float(self.config.get("tcn_retrain_days", 0) or 0)
        self._regime_refit_days = float(self.config.get("regime_refit_days", 0) or 0)

        # Failure tracking
        self._consecutive_failures = 0
        self._max_failures = int(self.config.get("max_consecutive_failures", 5))
        self._running = True

    def _project_path(self, *parts: str) -> Path:
        return self.project_root.joinpath(*parts)

    async def _retrain_tcn(self) -> None:
        """Trigger TCN retrain via the existing scripts/train_tcn.py subprocess.

        Marks the scheduler regardless of success so we don't retry every cycle
        on persistent failure (the next refit fires after the configured window).
        On success, hot-reloads the new model into the edge detector.
        """
        script = self._project_path("scripts", "train_tcn.py")
        if not script.exists():
            logger.warning("TCN retrain skipped — script not found: %s", script)
            self.scheduler.mark_ran("tcn_retrain")
            return
        try:
            logger.info("Starting TCN retrain (this may take a while)…")
            proc = await asyncio.create_subprocess_exec(
                "python3",
                str(script),
                cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            timeout = float(self.config.get("tcn_retrain_timeout_seconds", 1800))
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("TCN retrain timed out after %.0fs", timeout)
                self.scheduler.mark_ran("tcn_retrain")
                return
            if proc.returncode == 0:
                logger.info("TCN retrain succeeded — reloading model")
                self._reload_tcn_model()
            else:
                logger.error(
                    "TCN retrain failed (rc=%s): %s",
                    proc.returncode,
                    (stderr or b"").decode(errors="replace")[-500:],
                )
        except Exception as exc:
            logger.error("TCN retrain crashed: %s", exc)
        finally:
            self.scheduler.mark_ran("tcn_retrain")

    def _reload_tcn_model(self) -> None:
        try:
            import torch

            model_path = self._project_path("data", "models", "tcn_latest.pt")
            if not os.path.exists(model_path):
                return
            new_model = PolymarketEnsemble(n_features=10)
            new_model.load_state_dict(torch.load(model_path, weights_only=True))
            new_model.eval()
            self.tcn_model = new_model
            self.edge_detector.set_tcn_model(new_model)
            logger.info("Reloaded TCN model from %s", model_path)
        except Exception as exc:
            logger.warning("Could not reload TCN model: %s", exc)

    def _apply_learned_settings(self) -> None:
        """Push learned settings into the live config and edge detector."""
        self.config["min_edge"] = self.learner.min_edge
        self.config["signal_weights"] = dict(self.learner.signal_weights)
        self.edge_detector.min_edge = self.learner.min_edge
        self.edge_detector.signal_weights = dict(self.learner.signal_weights)

    def _refresh_probability_calibration(self, metrics: dict) -> CalibrationFitMetrics | None:
        trade_count = int(metrics.get("trade_count", 0))
        if not self.calibrator.should_refit(trade_count):
            return None

        fit_metrics = self.calibrator.fit_from_trades(
            self.tracker.get_closed_trades(limit=None),
            trade_count=trade_count,
        )
        if fit_metrics is None:
            return None

        self.calibrator.save()
        logger.info(
            "Updated probability calibration on %d trades (Brier %.4f -> %.4f)",
            fit_metrics.sample_count,
            fit_metrics.brier_before,
            fit_metrics.brier_after,
        )
        return fit_metrics

    def _write_scan_summary(
        self,
        *,
        all_markets: list,
        filtered_markets: list,
        order_books: dict[str, dict],
        opportunities: list,
        arbitrage_opps: list,
    ) -> None:
        """Persist a lightweight market watch summary for the dashboard."""
        top_n = int(self.config.get("market_watch_top_n", 12))
        category_mix: dict[str, int] = {}
        for market in filtered_markets:
            category = market.category or "unknown"
            category_mix[category] = category_mix.get(category, 0) + 1

        top_markets = sorted(filtered_markets, key=lambda market: market.volume_24h, reverse=True)[:top_n]
        top_spreads = sorted(order_books.values(), key=lambda market: market.get("spread", 0.0), reverse=True)[:top_n]

        summary = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_markets": len(all_markets),
            "filtered_markets": len(filtered_markets),
            "collected_order_books": len(order_books),
            "arbitrage_count": len(arbitrage_opps),
            "category_mix": category_mix,
            "top_markets": [
                {
                    "market_id": market.condition_id,
                    "question": market.question,
                    "category": market.category,
                    "volume_24h": market.volume_24h,
                    "liquidity": market.liquidity,
                }
                for market in top_markets
            ],
            "top_spreads": [
                {
                    "market_id": market.get("market_id"),
                    "question": market.get("question"),
                    "category": market.get("category"),
                    "spread": market.get("spread"),
                    "midpoint": market.get("midpoint"),
                    "bid_depth": market.get("bid_depth"),
                    "ask_depth": market.get("ask_depth"),
                }
                for market in top_spreads
            ],
            "top_opportunities": [
                {
                    "market_id": opportunity.market_id,
                    "question": opportunity.question,
                    "category": opportunity.category,
                    "regime": opportunity.regime,
                    "direction": opportunity.direction,
                    "edge": opportunity.edge,
                    "confidence": opportunity.confidence,
                    "estimated_prob": opportunity.estimated_prob,
                    "market_price": opportunity.market_price,
                }
                for opportunity in opportunities[:top_n]
            ],
            "top_arbitrage": [
                {
                    "market_id": opp.market_id,
                    "question": opp.question,
                    "gap": opp.gap,
                    "profit_after_costs": opp.profit_after_costs,
                }
                for opp in arbitrage_opps[:top_n]
            ],
        }
        out_path = self._project_path("data", "scan_summary.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_bytes = json.dumps(summary, indent=2).encode("utf-8")
        out_path.write_bytes(json_bytes)
        # Compressed copy for archival / bandwidth-sensitive readers
        with gzip.open(out_path.with_suffix(".json.gz"), "wb") as gz:
            gz.write(json_bytes)

    def _build_bracket_spec(self) -> BracketSpec | None:
        if not self.config.get("enable_bracket_orders", True):
            return None
        return BracketSpec(
            take_profit_pct=float(self.config.get("bracket_take_profit_pct", 0.06)),
            stop_loss_pct=float(self.config.get("bracket_stop_loss_pct", 0.035)),
            ttl_minutes=int(self.config.get("gtt_ttl_minutes", 24 * 60)),
        )

    async def _reconcile_order_lifecycle(self, order_books: dict[str, dict]) -> None:
        events = await self.executor.reconcile_orders(order_books)
        for event in events:
            order = event.order_snapshot
            if order is None:
                continue
            payload = asdict(order)
            if order.order_kind == "entry":
                self.tracker.sync_entry_order(payload)
                if not order.is_paper and order.status == "filled":
                    trade_id = self.tracker.get_trade_id_for_order(order.order_id)
                    trade = self.tracker.get_trade(trade_id) if trade_id is not None else None
                    if trade is not None and float(trade.get("fee_usdc") or 0.0) == 0.0:
                        self.tracker.record_fee(
                            trade_id,
                            fee_usdc=abs(
                                (order.filled_size_usdc or order.size_usdc)
                                * float(self.config.get("taker_fee_rate", 0.02))
                            ),
                            is_maker=False,
                            gas_cost_usdc=0.0,
                        )
                if event.event_type in {"entry_partially_filled", "entry_cancelled", "entry_expired"}:
                    self._write_alert(
                        "order",
                        f"{order.market_id} entry {event.status}",
                        severity="warning" if event.status in {"cancelled", "expired"} else "info",
                    )
            else:
                self.tracker.sync_bracket_state(payload)
                if event.event_type == "exit_filled":
                    trade_id = self.tracker.get_trade_id_for_order(order.parent_order_id or order.order_id)
                    if trade_id is not None:
                        self.tracker.record_fee(
                            trade_id,
                            fee_usdc=abs(
                                (order.filled_size_usdc or order.size_usdc)
                                * float(self.config.get("taker_fee_rate", 0.02))
                            ),
                            is_maker=False,
                            gas_cost_usdc=0.0,
                        )
                    pnl = self.tracker.close_trade_for_order(
                        payload,
                        exit_price=event.fill_price,
                        exit_reason=event.message or order.order_kind,
                    )
                    if pnl != 0.0:
                        self.drawdown.record_trade_pnl(pnl)
                    self._write_alert(
                        "bracket",
                        f"{order.market_id} {order.order_kind} triggered at {event.fill_price:.3f} pnl={pnl:.2f}",
                        severity="info",
                    )
                elif event.event_type in {"gtt_expired", "oco_cancelled"}:
                    self._write_alert(
                        "bracket",
                        f"{order.market_id} {event.message}",
                        severity="warning" if event.event_type == "gtt_expired" else "info",
                    )

    async def startup_checks(self) -> bool:
        """Validate environment and connectivity before trading."""
        critical_ok = True

        # Validate environment variables
        env_result = validate_env(require_live_trading=not self.config.get("dry_run", True))
        for err in env_result.errors:
            logger.error("ENV: %s", err)
            critical_ok = False
        for warn in env_result.warnings:
            logger.warning("ENV: %s", warn)

        # Environment variable checks
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set — sentiment analysis will be unavailable")
        if not os.environ.get("TAVILY_API_KEY"):
            logger.warning("TAVILY_API_KEY not set — news fetching will be unavailable")
        if not self.config.get("dry_run", True) and not os.environ.get("POLYMARKET_PRIVATE_KEY"):
            logger.error("POLYMARKET_PRIVATE_KEY required for live trading")
            critical_ok = False

        # API connectivity check
        try:
            test_markets = await self.gamma.get_markets(limit=1)
            if test_markets:
                logger.info("Gamma API connectivity OK (%d market returned)", len(test_markets))
            else:
                logger.warning("Gamma API returned no markets — may be a network issue")
        except Exception as exc:
            logger.warning("Gamma API check failed: %s — will retry during trading", exc)

        # Wallet check for live mode
        if not self.config.get("dry_run", True) and self.executor.clob is None:
            logger.error("Live mode requires a configured CLOB wallet")
            critical_ok = False

        # Reconcile open positions on restart
        open_positions = self.tracker.get_open_positions()
        if open_positions:
            logger.info(
                "Found %d open positions from previous session — will reconcile on first cycle",
                len(open_positions),
            )

        # Startup summary
        if self.config.get("demo_seed_on_start") and not self._project_path("data", "trades.db").exists():
            seed_demo_workspace(
                self.project_root,
                reset=True,
                trade_count=int(self.config.get("demo_trade_count", 36)),
                bankroll=self._bankroll,
            )

        # Initialize drawdown controller with starting bankroll.
        # If the kill switch was previously active (emergency drawdown) but has
        # since been deactivated, reset the peak to avoid a permanent drawdown
        # loop where the old high-water mark keeps triggering emergency mode.
        ks_active, _ = self.kill_switch.is_active()
        if not ks_active:
            dd_status = self.drawdown.get_status(self._bankroll)
            if dd_status["tier"] in ("stopped", "emergency"):
                self.drawdown.reset_peak(self._bankroll)
        self.drawdown.update(self._bankroll)
        self.sizer.update_bankroll(self._bankroll)

        logger.info(
            "Startup summary: dry_run=%s, bankroll=$%.2f, min_edge=%.2f%%, "
            "tcn_loaded=%s, regime_loaded=%s, calibration_loaded=%s",
            self.config.get("dry_run", True),
            self._bankroll,
            self.learner.min_edge * 100,
            self.tcn_model is not None,
            self.regime.fitted,
            self.calibrator.is_fitted,
        )

        return critical_ok

    async def run_cycle(self) -> None:
        """Execute one full scan-analyze-trade cycle."""
        import time as _time

        cycle_start = _time.monotonic()
        cycle_errors = 0
        trades_placed = 0

        try:
            # Kill switch check
            ks_active, ks_reason = self.kill_switch.is_active()
            if ks_active:
                logger.critical("Kill switch active: %s — skipping cycle", ks_reason)
                self._write_alert("kill_switch", f"Cycle skipped: {ks_reason}", severity="critical")
                return

            # 1. SCAN
            min_liq = self.config.get("min_liquidity", 500)
            min_vol = self.config.get("min_volume_24h", 100)
            min_days = self.config.get("min_days_to_resolution", 1)
            max_days = self.config.get("max_days_to_resolution", 90)

            all_markets = await self.gamma.get_all_markets()
            now = datetime.now(UTC)
            markets = []
            for m in all_markets:
                if m.liquidity < min_liq or m.volume_24h < min_vol:
                    continue
                if m.end_date:
                    days_left = (m.end_date - now).total_seconds() / 86400
                    if days_left < min_days or days_left > max_days:
                        continue
                markets.append(m)

            logger.info("Scanned %d markets, %d pass filters", len(all_markets), len(markets))
            await self._event_bus.publish(
                Event(
                    event_type=EventType.MARKET_SCANNED,
                    payload={"total": len(all_markets), "filtered": len(markets)},
                    source="agent",
                )
            )
            market_lookup = {market.condition_id: market for market in markets}

            # 2. COLLECT order book snapshots
            collected = 0
            order_books: dict[str, dict] = {}
            market_watch_top_n = int(self.config.get("market_watch_top_n", 12))
            for m in markets[: max(50, market_watch_top_n)]:
                if not m.clob_token_ids:
                    continue
                token_id = m.clob_token_ids[0]
                try:
                    snapshot = await self.clob.get_order_book(token_id)
                    if snapshot is None:
                        continue
                    self.history.save_snapshot(m.condition_id, snapshot, m.volume_24h)
                    # NO token ask ≈ 1 - YES best_bid (complement market)
                    no_ask = None
                    if len(m.clob_token_ids) >= 2:
                        # Ideally fetch NO token book; approximate for now
                        no_ask = 1.0 - snapshot.best_bid if snapshot.best_bid > 0 else None
                    order_books[m.condition_id] = {
                        "market_id": m.condition_id,
                        "question": m.question,
                        "category": m.category,
                        "midpoint": snapshot.midpoint,
                        "spread": snapshot.spread,
                        "bid_depth": snapshot.bid_depth,
                        "ask_depth": snapshot.ask_depth,
                        "best_bid": snapshot.best_bid,
                        "best_ask": snapshot.best_ask,
                        "bids": snapshot.bids,
                        "asks": snapshot.asks,
                        "volume_24h": m.volume_24h,
                        "yes_ask": snapshot.best_ask,
                        "no_ask": no_ask,
                        "token_id": token_id,
                        "no_token_id": m.clob_token_ids[1] if len(m.clob_token_ids) >= 2 else "",
                        "end_date": m.end_date,
                    }
                    collected += 1
                except Exception as exc:
                    logger.warning("Failed to collect %s: %s", m.condition_id, exc)
                await asyncio.sleep(0.3)

            logger.info("Collected %d order book snapshots", collected)
            await self._reconcile_order_lifecycle(order_books)

            # 3. ARBITRAGE scan
            arb_data = list(order_books.values())
            arb_opps = self.arbitrage.scan(arb_data)
            if arb_opps:
                for opp in arb_opps[:5]:
                    logger.info(
                        "Arbitrage opportunity: %s gap=%.4f profit=%.4f",
                        opp.question[:60],
                        opp.gap,
                        opp.profit_after_costs,
                    )
                    self._write_alert("arbitrage", f"{opp.question}: profit={opp.profit_after_costs:.4f}")

            # 4. NEWS — fetch for top markets by volume
            sorted_by_vol = sorted(markets, key=lambda m: m.volume_24h, reverse=True)
            news_top_n = max(1, int(self.config.get("news_top_n", 20)))
            for m in sorted_by_vol[:news_top_n]:
                try:
                    async with asyncio.timeout(float(self.config.get("api_timeout", 10))):
                        await self.news.search_news(m.question)
                except TimeoutError:
                    logger.warning("News fetch timed out for %s", m.condition_id)
                except Exception as exc:
                    logger.warning("News fetch failed for %s: %s", m.condition_id, exc)

            # 5. SENTIMENT — batch analyze uncached markets
            uncached: list[tuple[str, str, str]] = []
            for m in sorted_by_vol[:news_top_n]:
                cached = await asyncio.to_thread(self.sentiment.get_cached, m.condition_id)
                if cached is None:
                    uncached.append((m.condition_id, m.question, m.description))
            if uncached:
                try:
                    async with asyncio.timeout(float(self.config.get("llm_timeout", 30))):
                        await self.sentiment.analyze_batch(uncached[:news_top_n])
                    logger.info("Analyzed sentiment for %d markets", len(uncached[:news_top_n]))
                except TimeoutError:
                    logger.warning("Sentiment batch analysis timed out — continuing with stale data")
                except Exception as exc:
                    logger.warning("Sentiment batch analysis failed: %s — continuing with stale data", exc)

            # 6. REGIME — fit from recent history if not fitted, or on schedule
            regime_due = not self.regime.fitted or (
                self._regime_refit_days > 0 and self.scheduler.due("regime_refit", self._regime_refit_days)
            )
            if regime_due:
                all_condition_ids = self.history.list_markets()
                combined_frames = []
                for cid in all_condition_ids[:50]:
                    hist = self.history.load_history(cid, lookback_hours=7 * 24)
                    if hist is not None and len(hist) >= 20:
                        combined_frames.append(hist)
                if combined_frames:
                    combined = pd.concat(combined_frames, ignore_index=True)
                    if len(combined) >= 100:
                        self.regime.fit_from_history(combined)
                        self.regime.save(str(self._project_path("data", "models", "regime_latest.pkl")))
                        self.scheduler.mark_ran("regime_refit")
                        logger.info(
                            "Regime model fitted and saved (refit_days=%s)",
                            self._regime_refit_days,
                        )

            # 6b. TCN retrain — invoke training script as a subprocess on schedule
            if self._tcn_retrain_days > 0 and self.scheduler.due("tcn_retrain", self._tcn_retrain_days):
                await self._retrain_tcn()

            precomputed_cross = self.edge_detector.precompute_cross_signals(markets, self.history)

            # 7. EDGE — calculate for each market
            opportunities = []
            current_min_edge = self.learner.min_edge
            for cid, mdata in order_books.items():
                hist = self.history.load_history(cid, lookback_hours=48)
                features = None
                if hist is not None and len(hist) >= 20:
                    features = self.features.create_features(hist, resolution_date=mdata.get("end_date"))
                market_obj = market_lookup.get(cid)
                if market_obj is not None:
                    mdata["related_markets"] = precomputed_cross.get(cid, [])

                # Regime prediction
                regime = "stable"
                if hist is not None and len(hist) >= 2:
                    vol = hist["midpoint"].pct_change().std()
                    vol_ratio = hist["volume"].iloc[-1] / (hist["volume"].mean() + 1e-8)
                    spread_val = hist["spread"].iloc[-1]
                    regime, _ = self.regime.predict(vol, vol_ratio, spread_val)

                # News volume
                news_results = self.news.get_cached(mdata["question"])
                if news_results is None:
                    news_results = await self.news.search_news(mdata["question"])
                news_count, news_ratio = self.news.get_news_volume(news_results)
                mdata["news_volume_ratio"] = news_ratio

                # Price rate of change (actual 24h lookback)
                if hist is not None and len(hist) >= 2:
                    ts_col = pd.to_datetime(hist["timestamp"], utc=True) if "timestamp" in hist.columns else hist.index
                    cutoff_24h = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=24)
                    recent = hist[ts_col >= cutoff_24h]
                    if len(recent) >= 2:
                        mdata["price_roc_24h"] = float(recent["midpoint"].iloc[-1] - recent["midpoint"].iloc[0])
                    else:
                        mdata["price_roc_24h"] = 0.0
                else:
                    mdata["price_roc_24h"] = 0.0

                edge_result = await self.edge_detector.estimate_edge(mdata, features, regime)
                if edge_result.edge >= current_min_edge:
                    opportunities.append(edge_result)

            logger.info("Found %d edges >= %.2f%%", len(opportunities), current_min_edge * 100)
            self._write_scan_summary(
                all_markets=all_markets,
                filtered_markets=markets,
                order_books=order_books,
                opportunities=opportunities,
                arbitrage_opps=arb_opps,
            )

            # 8. SORT by edge * confidence descending
            opportunities.sort(key=lambda e: e.edge * e.confidence, reverse=True)

            # 9. SIZE & 10. EXECUTE — top 5 opportunities
            open_positions = self.tracker.get_open_positions()
            self.drawdown.update(self._bankroll)
            dd_mult = self.drawdown.get_multiplier(self._bankroll)

            if self.drawdown.should_close_all(self._bankroll):
                logger.critical("EMERGENCY: drawdown threshold breached — closing all positions")
                await self.executor.cancel_all_open_orders()
                self.kill_switch.activate("Emergency drawdown threshold breached")
                self._write_alert(
                    "drawdown",
                    "Emergency close-all triggered — kill switch activated",
                    severity="critical",
                )
                return

            opportunity_top_n = int(self.config.get("opportunity_top_n", 10))
            for edge_result in opportunities[: min(5, opportunity_top_n)]:
                edge_dict = {
                    "raw_estimated_prob": edge_result.raw_estimated_prob,
                    "estimated_prob": edge_result.estimated_prob,
                    "calibrated_prob": edge_result.calibrated_prob,
                    "market_price": edge_result.market_price,
                    "edge": edge_result.edge,
                    "direction": edge_result.direction,
                    "confidence": edge_result.confidence,
                    "regime": edge_result.regime,
                    "category": edge_result.category,
                    "market_id": edge_result.market_id,
                    "question": edge_result.question,
                    "signal_breakdown": edge_result.signal_breakdown,
                }

                size = self.sizer.calculate_position(edge_dict, open_positions, dd_mult)
                if size <= 0:
                    continue

                # Find token_id from order_books — use NO token for NO direction
                book_data = order_books.get(edge_result.market_id, {})
                if edge_result.direction == "NO" and book_data.get("no_token_id"):
                    token_id = book_data["no_token_id"]
                else:
                    token_id = book_data.get("token_id", "")

                if (not self.config.get("dry_run", True)) and book_data.get("bids") and book_data.get("asks"):
                    from polymarket_agent.execution.slippage import check_slippage_ok

                    ok, estimate = check_slippage_ok(
                        size,
                        "BUY",
                        book_data["bids"],
                        book_data["asks"],
                        float(self.config.get("max_slippage_bps", 50)),
                    )
                    if not ok:
                        logger.warning(
                            "Skipping %s due to slippage %.1f bps",
                            edge_result.market_id,
                            estimate.slippage_bps,
                        )
                        continue

                order = await self.executor.place_order(
                    market_id=edge_result.market_id,
                    direction=edge_result.direction,
                    size_usdc=size,
                    price=edge_result.market_price,
                    token_id=token_id,
                    bracket_spec=self._build_bracket_spec(),
                    market_book=book_data,
                )

                if order is not None:
                    order_payload = asdict(order)
                    # 11. TRACK
                    if order.is_paper:
                        self.tracker.record_paper_trade(edge_dict, size, order=order_payload)
                    else:
                        self.tracker.record_trade(edge_dict, size, order_payload)
                    trade_id = self.tracker.get_trade_id_for_order(order.order_id)
                    if trade_id is not None and (order.is_paper or order.status == "filled"):
                        self.tracker.record_fee(
                            trade_id,
                            fee_usdc=size * float(self.config.get("taker_fee_rate", 0.02)),
                            is_maker=False,
                            gas_cost_usdc=0.0,
                        )

                    # Deduct position cost only for instant paper fills.
                    # Live orders are deducted via the bankroll recalculation
                    # below (initial + realized_pnl - open_capital) once the
                    # tracker records them, avoiding double-counting.
                    if order.status == "filled":
                        self._bankroll = max(0.0, self._bankroll - size)
                        self.sizer.update_bankroll(self._bankroll)

                    logger.info(
                        "Placed %s %s $%.2f @ %.3f edge=%.2f%% conf=%.2f regime=%s bankroll=$%.2f | %s",
                        edge_result.direction,
                        "PAPER" if order.is_paper else "LIVE",
                        size,
                        edge_result.market_price,
                        edge_result.edge * 100,
                        edge_result.confidence,
                        edge_result.regime,
                        self._bankroll,
                        edge_result.question[:60],
                    )
                    await self._event_bus.publish(
                        Event(
                            event_type=EventType.ORDER_PLACED,
                            payload={
                                "market_id": edge_result.market_id,
                                "direction": edge_result.direction,
                                "size": size,
                            },
                            source="agent",
                        )
                    )

                    # Refresh open positions for next sizing
                    open_positions = self.tracker.get_open_positions()
                    trades_placed += 1

            # Update bankroll from realized PnL on closed trades
            metrics = self.tracker.get_metrics()
            realized_pnl = metrics.get("total_pnl", 0.0)
            initial = float(self.config.get("bankroll", 100))
            # Bankroll = initial + realized PnL - capital in open positions
            open_capital = sum(p.get("size_usdc", 0) for p in open_positions)
            self._bankroll = max(0.0, initial + realized_pnl - open_capital)
            if self._bankroll <= 0:
                logger.critical("Bankroll depleted ($%.2f) — pausing new trades", self._bankroll)
            self.sizer.update_bankroll(self._bankroll)
            self.drawdown.update(self._bankroll)

            # Record portfolio snapshot
            dd = self.drawdown.get_drawdown(self._bankroll)
            self.tracker.record_snapshot(self._bankroll, dd)

            # 12. LEARN
            calibration_update = self._refresh_probability_calibration(metrics)
            if calibration_update is not None:
                self._write_alert(
                    "calibration",
                    (
                        "Refit probability calibration "
                        f"(Brier {calibration_update.brier_before:.4f} -> "
                        f"{calibration_update.brier_after:.4f})"
                    ),
                    severity="info",
                )

            previous_min_edge = self.learner.min_edge
            previous_weights = dict(self.learner.signal_weights)
            adjustments = self.learner.check_and_adjust(metrics)
            self._apply_learned_settings()
            if adjustments["min_edge"] != previous_min_edge or adjustments["signal_weights"] != previous_weights:
                self._write_alert(
                    "learner",
                    f"Updated min_edge={adjustments['min_edge']:.3f} and refreshed signal weights",
                    severity="info",
                )
            if metrics["trade_count"] > 0:
                self.tracker.log_summary()

            # 13. ALERTS
            if metrics.get("brier_score") is not None and metrics["brier_score"] > 0.30:
                self._write_alert("calibration", f"High Brier score: {metrics['brier_score']:.4f}", severity="warning")

            # Cancel expired orders
            expired_ids = await self.executor.cancel_expired_orders()
            for order_id in expired_ids:
                order = self.executor.get_order(order_id)
                if order is None:
                    continue
                payload = asdict(order)
                if order.order_kind == "entry":
                    self.tracker.sync_entry_order(payload)
                else:
                    self.tracker.sync_bracket_state(payload)

            # 14. RECONCILE positions
            recon_issues = self._reconciler.reconcile(self.tracker, self.executor)
            for issue in recon_issues:
                self._write_alert(
                    f"recon:{issue.issue_type}",
                    issue.description,
                    severity=issue.severity,
                )

            self._consecutive_failures = 0

        except asyncio.CancelledError:
            logger.info("Cycle cancelled — shutting down")
            raise
        except (OSError, ConnectionError) as exc:
            self._consecutive_failures += 1
            cycle_errors += 1
            logger.error(
                "Network error in cycle (consecutive failures: %d): %s",
                self._consecutive_failures,
                exc,
            )
            self._write_alert("health", f"Network error: {exc}", severity="error")
        except Exception:
            self._consecutive_failures += 1
            cycle_errors += 1
            logger.exception("Cycle failed (consecutive failures: %d)", self._consecutive_failures)
            self._write_alert("health", f"Cycle failed ({self._consecutive_failures} consecutive)", severity="error")
        finally:
            await self._event_bus.process()
            # Heartbeat — always record, even on failure
            cycle_duration = _time.monotonic() - cycle_start
            self._heartbeat.beat(
                cycle_metrics={
                    "duration_seconds": round(cycle_duration, 2),
                    "trades_placed": trades_placed,
                    "errors": cycle_errors,
                    "consecutive_failures": self._consecutive_failures,
                }
            )

            # Auto-activate kill switch on too many consecutive failures
            if self._consecutive_failures >= self._max_failures:
                self.kill_switch.activate(f"Auto-triggered: {self._consecutive_failures} consecutive failures")

    async def run_forever(self, interval_minutes: int | None = None) -> None:
        """Run the trading loop indefinitely with health checks."""
        if not await self.startup_checks():
            logger.critical("Startup checks failed — aborting")
            return

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)
        self._shutdown.install(loop)

        interval = interval_minutes or self.config.get("scan_interval_minutes", 15)
        logger.info("Starting trading loop with %d-minute interval", interval)

        while self._running:
            if self._consecutive_failures >= self._max_failures:
                logger.critical(
                    "Hit %d consecutive failures — pausing for 1 hour",
                    self._max_failures,
                )
                self._write_alert(
                    "health",
                    f"{self._max_failures} consecutive failures — pausing 1 hour",
                    severity="critical",
                )
                await asyncio.sleep(3600)
                self._consecutive_failures = 0
            else:
                await self.run_cycle()
                if self._running:
                    await asyncio.sleep(interval * 60)

        await self.cleanup()

    async def run_once(self) -> None:
        """Run a single trading cycle and always release shared resources."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)
        try:
            await self.run_cycle()
        finally:
            await self.cleanup()

    def _handle_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False

    async def _shutdown_cancel_orders(self) -> None:
        """Shutdown hook: cancel all open orders."""
        cancelled = await self.executor.cancel_all_open_orders()
        logger.info("Shutdown: cancelled %d open orders", cancelled)

    async def _shutdown_save_state(self) -> None:
        """Shutdown hook: save tracker and learner state."""
        self.tracker.close()
        logger.info("Shutdown: saved tracker state")

    async def _shutdown_close_clients(self) -> None:
        """Shutdown hook: close API client sessions."""
        await self.gamma.close()
        await self.clob.close()
        logger.info("Shutdown: closed API sessions")

    async def _check_gamma_health(self) -> bool:
        """Health check: can we reach the Gamma API?"""
        try:
            markets = await self.gamma.get_markets(limit=1)
            return markets is not None and len(markets) > 0
        except Exception:
            return False

    async def _check_tracker_health(self) -> bool:
        """Health check: is the trades database accessible?"""
        try:
            self.tracker.get_metrics()
            return True
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cancel open orders, flush state, and close API sessions."""
        logger.info("Starting graceful shutdown...")
        self._write_alert("health", "Agent shutting down", severity="info")
        results = await self._shutdown.execute()
        failed = [k for k, v in results.items() if v != "ok"]
        if failed:
            logger.warning("Shutdown completed with failures: %s", failed)
        else:
            logger.info("Agent shutdown complete — clean exit")

    async def scan_only(self) -> list:
        """Fetch and display active markets (for CLI scan command)."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)
        markets = await self.gamma.get_all_markets()
        markets.sort(key=lambda m: m.volume_24h, reverse=True)
        logger.info("=" * 80)
        logger.info("%-50s %6s %6s %10s %10s", "Market", "YES", "NO", "Vol24h", "Liquidity")
        logger.info("=" * 80)
        for m in markets[:30]:
            logger.info(
                "%-50s %6.3f %6.3f %10.0f %10.0f",
                m.question[:50],
                m.yes_price,
                m.no_price,
                m.volume_24h,
                m.liquidity,
            )
        logger.info("Total active markets: %d", len(markets))
        await self.gamma.close()
        return markets

    def _write_alert(self, alert_type: str, message: str, severity: str = "info") -> None:
        """Route alert through the unified alert system (JSONL + Telegram)."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._alert_router.alert(alert_type, message, severity))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
        except RuntimeError:
            # No running loop — fall back to sync JSONL write
            alert = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": alert_type,
                "message": message,
                "severity": severity,
            }
            alerts_path = self._project_path("data", "alerts.jsonl")
            alerts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(alerts_path, "a") as f:
                f.write(json.dumps(alert) + "\n")

        webhook_url = self.config.get("alert_webhook_url")
        webhook_severities = self.config.get("alert_webhook_severities", ["critical", "error"])
        if webhook_url and severity in webhook_severities:
            alert = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": alert_type,
                "message": message,
                "severity": severity,
            }
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._dispatch_webhook(webhook_url, alert))
            except RuntimeError:
                pass  # No event loop — skip webhook

    async def _dispatch_webhook(self, url: str, alert: dict) -> None:
        """Fire-and-forget async webhook POST. Never raises."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                logger.warning("Skipping webhook with unsupported scheme: %s", url)
                return
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    url,
                    json=alert,
                    timeout=aiohttp.ClientTimeout(total=5),
                ),
            ):
                pass
        except Exception as exc:
            logger.warning("Webhook dispatch failed: %s", exc)


def cli():
    """CLI entry point."""
    print(DISCLAIMER_BANNER)
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Neural Trading Agent")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--i-understand-geo-risk",
        action="store_true",
        help="Acknowledge geo-restriction risk for live trading from restricted regions.",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("scan", help="Scan and display markets")
    demo_p = sub.add_parser("demo-seed", help="Seed demo trades, alerts, and dashboard data")
    demo_p.add_argument("--trades", type=int, default=36)
    demo_p.add_argument("--no-reset", action="store_true")

    trade_p = sub.add_parser("trade", help="Run trading loop")
    trade_g = trade_p.add_mutually_exclusive_group()
    trade_g.add_argument("--dry-run", action="store_true")
    trade_g.add_argument("--live", action="store_true")
    trade_p.add_argument("--once", action="store_true", help="Single cycle")

    # Kill switch commands
    ks_p = sub.add_parser("kill-switch", help="Manage the emergency kill switch")
    ks_sub = ks_p.add_subparsers(dest="ks_action")
    ks_on = ks_sub.add_parser("on", help="Activate kill switch")
    ks_on.add_argument("--reason", default="Manual activation", help="Reason for activation")
    ks_sub.add_parser("off", help="Deactivate kill switch")
    ks_sub.add_parser("status", help="Check kill switch status")

    args = parser.parse_args()

    from polymarket_agent.utils.logger import setup_logging

    setup_logging(level="DEBUG" if args.verbose else "INFO", json_output=not args.verbose)

    if not args.command:
        parser.print_help()
        return

    if args.command == "demo-seed":
        config_path = _resolve_config_path(args.config)
        result = seed_demo_workspace(
            config_path.parent.parent,
            reset=not getattr(args, "no_reset", False),
            trade_count=getattr(args, "trades", 36),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "kill-switch":
        ks = KillSwitch(path=str(_PROJECT_ROOT / "data" / "KILL_SWITCH"))
        if args.ks_action == "on":
            ks.activate(args.reason)
            print(f"🚨 Kill switch ACTIVATED: {args.reason}")
        elif args.ks_action == "off":
            ks.deactivate()
            print("✅ Kill switch deactivated")
        else:
            active, reason = ks.is_active()
            if active:
                print(f"🚨 ACTIVE — {reason}")
            else:
                print("✅ Kill switch is OFF")
        return

    agent = PolymarketTradingAgent(args.config)
    if args.command == "scan":
        asyncio.run(agent.scan_only())
    elif args.command == "trade":
        if hasattr(args, "dry_run") and args.dry_run:
            agent.config["dry_run"] = True
        elif hasattr(args, "live") and args.live:
            agent.config["dry_run"] = False
        live_mode = not agent.config.get("dry_run", True)
        asyncio.run(_check_geo_restriction(live_mode, auto_accept=args.i_understand_geo_risk))
        if hasattr(args, "once") and args.once:
            asyncio.run(agent.run_once())
        else:
            asyncio.run(agent.run_forever())


if __name__ == "__main__":
    cli()
