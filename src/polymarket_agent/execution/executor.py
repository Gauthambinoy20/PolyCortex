import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import ceil
from uuid import uuid4

from polymarket_agent.execution.slippage import check_slippage_ok
from polymarket_agent.risk.kill_switch import KillSwitch

logger = logging.getLogger(__name__)

FINAL_ORDER_STATUSES = {"filled", "cancelled", "expired"}


class InsufficientBalanceError(Exception):
    """Raised when USDC balance is insufficient for an order."""


class DuplicateOrderError(Exception):
    """Raised when a duplicate client_order_id is detected."""


@dataclass(frozen=True)
class BracketSpec:
    take_profit_pct: float
    stop_loss_pct: float
    ttl_minutes: int = 24 * 60


@dataclass
class Order:
    order_id: str
    market_id: str
    token_id: str
    direction: str  # 'YES' or 'NO'
    side: str  # 'BUY'
    size_usdc: float
    price: float
    status: str  # placed, partially_filled, filled, armed, cancelled, expired
    timestamp: datetime
    chunks: int
    is_paper: bool
    exchange_order_ids: list[str] = field(default_factory=list)
    filled_size_usdc: float = 0.0
    remaining_size_usdc: float = 0.0
    average_fill_price: float | None = None
    last_fill_at: datetime | None = None
    order_type: str = "limit"
    order_kind: str = "entry"
    reduce_only: bool = False
    parent_order_id: str | None = None
    oco_group_id: str | None = None
    trigger_price: float | None = None
    trigger_condition: str | None = None
    take_profit_price: float | None = None
    stop_loss_price: float | None = None
    bracket_order_ids: list[str] = field(default_factory=list)
    bracket_state: str = "inactive"
    gtt_expiry_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.remaining_size_usdc <= 0.0 and self.status not in FINAL_ORDER_STATUSES and self.status != "armed":
            self.remaining_size_usdc = self.size_usdc - self.filled_size_usdc
        if self.status == "filled":
            self.remaining_size_usdc = 0.0
            if self.filled_size_usdc <= 0.0:
                self.filled_size_usdc = self.size_usdc
            if self.average_fill_price is None:
                self.average_fill_price = self.price
        elif self.status == "armed":
            self.remaining_size_usdc = self.size_usdc


@dataclass
class OrderLifecycleEvent:
    event_type: str
    order_id: str
    market_id: str
    status: str
    message: str = ""
    fill_price: float | None = None
    fill_size_usdc: float = 0.0
    parent_order_id: str | None = None
    sibling_order_id: str | None = None
    order_snapshot: Order | None = None


class OrderExecutor:
    def __init__(self, config: dict) -> None:
        self.dry_run: bool = config.get("dry_run", True)
        self.split_orders_above: float = config.get("split_orders_above", 50)
        self.order_expiry_minutes: int = config.get("order_expiry_minutes", 60)
        self.enable_bracket_orders: bool = config.get("enable_bracket_orders", True)
        self.bracket_take_profit_pct: float = config.get("bracket_take_profit_pct", 0.06)
        self.bracket_stop_loss_pct: float = config.get("bracket_stop_loss_pct", 0.035)
        self.gtt_ttl_minutes: int = config.get("gtt_ttl_minutes", 24 * 60)
        self.paper_fill_on_place: bool = config.get("paper_fill_on_place", True)
        self.paper_allow_partial_fills: bool = config.get("paper_allow_partial_fills", True)
        self.paper_partial_fill_ratio: float = config.get("paper_partial_fill_ratio", 0.5)
        self.paper_fill_tolerance: float = config.get("paper_fill_tolerance", 0.002)
        self.live_reconcile_enabled: bool = config.get("live_reconcile_enabled", True)
        self.max_single_order_usdc: float = config.get("max_single_order_usdc", 500.0)
        self.dedup_window_seconds: int = config.get("dedup_window_seconds", 300)
        self.orders: dict[str, Order] = {}
        self._recent_placements: dict[tuple[str, str, str], datetime] = {}
        self._order_lock = asyncio.Lock()
        self.kill_switch: KillSwitch = KillSwitch(config.get("kill_switch_path", "data/KILL_SWITCH"))
        self.clob = None
        self.config: dict = config
        self._db_path: str = config.get("db_path", "data/trades.db")
        self._init_submitted_orders_db()

        if not self.dry_run:
            key = os.environ.get("POLYMARKET_PRIVATE_KEY")
            if key:
                try:
                    from py_clob_client.client import ClobClient

                    host = config.get("host", "https://clob.polymarket.com")
                    self.clob = ClobClient(host, key=key, chain_id=137)
                    logger.info("Live CLOB client initialized")
                except Exception as exc:
                    raise RuntimeError(
                        f"Live trading requires a working CLOB client, but initialization failed: {exc}"
                    ) from exc
            else:
                raise RuntimeError(
                    "POLYMARKET_PRIVATE_KEY environment variable is required for live trading (dry_run=False)"
                )

    def _next_order_id(self) -> str:
        return uuid4().hex[:12]

    def _init_submitted_orders_db(self) -> None:
        db_dir = os.path.dirname(self._db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS submitted_orders (
                        client_order_id TEXT PRIMARY KEY,
                        market_id TEXT NOT NULL,
                        side TEXT NOT NULL,
                        size REAL NOT NULL,
                        status TEXT DEFAULT 'pending',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to initialize submitted_orders DB: %s", exc)

    def _persist_order_intent(self, client_order_id: str, market_id: str, side: str, size: float) -> bool:
        """Persist order intent before submission. Returns False if already exists (duplicate)."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "INSERT INTO submitted_orders (client_order_id, market_id, side, size, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (client_order_id, market_id, side, size, datetime.now(UTC).isoformat()),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            logger.warning("Duplicate order detected for client_order_id=%s — skipping", client_order_id)
            return False
        except Exception as exc:
            logger.warning("Failed to persist order intent: %s", exc)
            return True  # Fail-open to avoid blocking trading on DB issues

    def _update_order_status_db(self, client_order_id: str, status: str) -> None:
        """Update the status of a submitted order in the DB."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "UPDATE submitted_orders SET status=? WHERE client_order_id=?",
                    (status, client_order_id),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to update order status in DB: %s", exc)

    async def _assert_sufficient_balance(self, order_size_usdc: float) -> None:
        """Check on-chain USDC balance before submitting a live order."""
        if self.dry_run:
            return
        try:
            import aiohttp

            polygon_rpc = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
            usdc_contract = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            wallet = os.environ.get("POLYMARKET_WALLET_ADDRESS", "")
            if not wallet:
                logger.warning("POLYMARKET_WALLET_ADDRESS not set, skipping balance check")
                return
            fn_selector = "0x70a08231"
            padded_addr = wallet.lower().replace("0x", "").zfill(64)
            data = fn_selector + padded_addr
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": usdc_contract, "data": data}, "latest"],
                "id": 1,
            }
            async with (
                aiohttp.ClientSession() as session,
                session.post(polygon_rpc, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp,
            ):
                result = await resp.json()
            hex_balance = result.get("result", "0x0")
            balance_raw = int(hex_balance, 16)
            balance_usdc = balance_raw / 1e6
            if balance_usdc < order_size_usdc:
                raise InsufficientBalanceError(
                    f"Insufficient USDC balance: have ${balance_usdc:.2f}, need ${order_size_usdc:.2f}"
                )
            logger.info(
                "Balance check passed: $%.2f USDC available (need $%.2f)",
                balance_usdc,
                order_size_usdc,
            )
        except InsufficientBalanceError:
            raise
        except Exception as exc:
            logger.warning("Balance check failed (non-fatal): %s", exc)

    async def _check_gas_price(self) -> None:
        """Abort if Polygon gas price exceeds max_gas_gwei threshold."""
        if self.dry_run:
            return
        max_gas_gwei = float(os.environ.get("MAX_GAS_GWEI", "200"))
        try:
            import aiohttp

            polygon_rpc = os.environ.get("POLYGON_RPC_URL", "https://polygon-rpc.com")
            payload = {"jsonrpc": "2.0", "method": "eth_gasPrice", "params": [], "id": 1}
            async with (
                aiohttp.ClientSession() as session,
                session.post(polygon_rpc, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as resp,
            ):
                result = await resp.json()
            hex_gas = result.get("result", "0x0")
            gas_wei = int(hex_gas, 16)
            gas_gwei = gas_wei / 1e9
            if gas_gwei > max_gas_gwei:
                raise RuntimeError(
                    f"Gas price too high: {gas_gwei:.1f} gwei (max: {max_gas_gwei:.1f} gwei). "
                    "Aborting order to prevent excessive gas costs."
                )
            logger.info("Gas price OK: %.1f gwei (max: %.1f)", gas_gwei, max_gas_gwei)
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Gas price check failed (non-fatal): %s", exc)

    @staticmethod
    def _contract_price(direction: str, market_price: float) -> float:
        return market_price if direction == "YES" else (1.0 - market_price)

    @staticmethod
    def _clamp_price(price: float) -> float:
        return max(0.01, min(float(price), 0.99))

    def _validate_order(
        self,
        *,
        market_id: str,
        token_id: str,
        direction: str,
        size_usdc: float,
        price: float,
    ) -> str | None:
        """Validate an order before placement.

        Returns:
            ``None`` if valid, otherwise a human-readable rejection reason.
        """
        # Kill switch
        active, reason = self.kill_switch.is_active()
        if active:
            return f"Kill switch is active: {reason}"

        # Price bounds
        if not (0.01 <= price <= 0.99):
            return f"Price {price:.4f} outside allowed range [0.01, 0.99]"

        # Size bounds
        if size_usdc < 0.50:
            return f"Size ${size_usdc:.2f} below minimum $0.50"
        if size_usdc > self.max_single_order_usdc:
            return f"Size ${size_usdc:.2f} exceeds max ${self.max_single_order_usdc:.2f}"

        # Direction
        if direction not in ("YES", "NO"):
            return f"Invalid direction '{direction}' — must be 'YES' or 'NO'"

        # Required identifiers
        if not token_id:
            if not self.dry_run:
                return "token_id must be non-empty"
            logger.warning("token_id is empty — acceptable in dry-run mode only")
        if not market_id:
            return "market_id must be non-empty"

        return None

    def _cleanup_recent_placements(self) -> None:
        """Remove expired entries from the deduplication cache."""
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=self.dedup_window_seconds)
        expired = [k for k, ts in self._recent_placements.items() if ts < cutoff]
        for k in expired:
            del self._recent_placements[k]
        # Hard cap to prevent unbounded growth
        if len(self._recent_placements) > 10000:
            oldest = sorted(self._recent_placements, key=self._recent_placements.get)[:5000]
            for k in oldest:
                del self._recent_placements[k]

    def _is_duplicate(self, market_id: str, direction: str, order_kind: str = "entry") -> bool:
        """Check if this (market_id, direction, order_kind) was placed recently.

        Returns:
            ``True`` if a duplicate is detected (should skip).
        """
        self._cleanup_recent_placements()
        key = (market_id, direction, order_kind)
        if key in self._recent_placements:
            last = self._recent_placements[key]
            logger.warning(
                "Duplicate order blocked: %s %s was placed at %s (within %ds window)",
                direction,
                market_id,
                last.isoformat(),
                self.dedup_window_seconds,
            )
            return True
        return False

    def _default_bracket_spec(self) -> BracketSpec | None:
        if not self.enable_bracket_orders:
            return None
        return BracketSpec(
            take_profit_pct=self.bracket_take_profit_pct,
            stop_loss_pct=self.bracket_stop_loss_pct,
            ttl_minutes=self.gtt_ttl_minutes,
        )

    def _build_entry_order(
        self,
        *,
        market_id: str,
        direction: str,
        size_usdc: float,
        price: float,
        token_id: str,
        chunks: int,
        is_paper: bool,
        timestamp: datetime,
    ) -> Order:
        return Order(
            order_id=self._next_order_id(),
            market_id=market_id,
            token_id=token_id,
            direction=direction,
            side="BUY",
            size_usdc=size_usdc,
            price=price,
            status="placed",
            timestamp=timestamp,
            chunks=chunks,
            is_paper=is_paper,
            remaining_size_usdc=size_usdc,
            order_type="limit",
            order_kind="entry",
        )

    def _apply_fill(
        self,
        order: Order,
        *,
        fill_size_usdc: float,
        fill_price: float,
        now: datetime,
    ) -> str:
        fill_size = max(0.0, min(fill_size_usdc, order.remaining_size_usdc))
        if fill_size <= 0.0:
            return order.status

        previous_filled = order.filled_size_usdc
        new_filled = previous_filled + fill_size
        if previous_filled > 0.0 and order.average_fill_price is not None:
            order.average_fill_price = (
                (order.average_fill_price * previous_filled) + (fill_price * fill_size)
            ) / new_filled
        else:
            order.average_fill_price = fill_price

        order.filled_size_usdc = round(new_filled, 8)
        order.remaining_size_usdc = round(max(order.size_usdc - order.filled_size_usdc, 0.0), 8)
        order.last_fill_at = now
        if order.remaining_size_usdc <= 1e-8:
            order.remaining_size_usdc = 0.0
            order.status = "filled"
        else:
            order.status = "partially_filled"
        return order.status

    def _build_bracket_levels(self, entry_price: float, bracket_spec: BracketSpec) -> tuple[float, float]:
        tp = self._clamp_price(entry_price * (1.0 + bracket_spec.take_profit_pct))
        sl = self._clamp_price(entry_price * (1.0 - bracket_spec.stop_loss_pct))
        return tp, sl

    def _sync_child_sizes(self, parent_order: Order) -> None:
        active_size = parent_order.filled_size_usdc or parent_order.size_usdc
        for child_id in parent_order.bracket_order_ids:
            child = self.orders.get(child_id)
            if child is None or child.status != "armed":
                continue
            child.size_usdc = active_size
            child.remaining_size_usdc = active_size

    def _arm_bracket_orders(
        self,
        parent_order: Order,
        *,
        bracket_spec: BracketSpec,
        now: datetime,
    ) -> list[OrderLifecycleEvent]:
        if parent_order.order_kind != "entry":
            return []
        active_price = parent_order.average_fill_price or parent_order.price
        tp_price, sl_price = self._build_bracket_levels(active_price, bracket_spec)
        parent_order.take_profit_price = tp_price
        parent_order.stop_loss_price = sl_price

        if parent_order.bracket_order_ids:
            parent_order.bracket_state = "armed"
            self._sync_child_sizes(parent_order)
            return [
                OrderLifecycleEvent(
                    event_type="brackets_armed",
                    order_id=parent_order.order_id,
                    market_id=parent_order.market_id,
                    status=parent_order.bracket_state,
                    message="Bracket orders refreshed",
                    parent_order_id=parent_order.order_id,
                    order_snapshot=parent_order,
                )
            ]

        oco_group_id = uuid4().hex[:12]
        expiry_at = now + timedelta(minutes=bracket_spec.ttl_minutes)
        child_size = parent_order.filled_size_usdc or parent_order.size_usdc
        take_profit = Order(
            order_id=self._next_order_id(),
            market_id=parent_order.market_id,
            token_id=parent_order.token_id,
            direction=parent_order.direction,
            side="SELL",
            size_usdc=child_size,
            price=tp_price,
            status="armed",
            timestamp=now,
            chunks=1,
            is_paper=parent_order.is_paper,
            remaining_size_usdc=child_size,
            order_type="gtt",
            order_kind="take_profit",
            reduce_only=True,
            parent_order_id=parent_order.order_id,
            oco_group_id=oco_group_id,
            trigger_price=tp_price,
            trigger_condition="gte",
            gtt_expiry_at=expiry_at,
        )
        stop_loss = Order(
            order_id=self._next_order_id(),
            market_id=parent_order.market_id,
            token_id=parent_order.token_id,
            direction=parent_order.direction,
            side="SELL",
            size_usdc=child_size,
            price=sl_price,
            status="armed",
            timestamp=now,
            chunks=1,
            is_paper=parent_order.is_paper,
            remaining_size_usdc=child_size,
            order_type="gtt",
            order_kind="stop_loss",
            reduce_only=True,
            parent_order_id=parent_order.order_id,
            oco_group_id=oco_group_id,
            trigger_price=sl_price,
            trigger_condition="lte",
            gtt_expiry_at=expiry_at,
        )
        self.orders[take_profit.order_id] = take_profit
        self.orders[stop_loss.order_id] = stop_loss
        parent_order.bracket_order_ids = [take_profit.order_id, stop_loss.order_id]
        parent_order.bracket_state = "armed"

        return [
            OrderLifecycleEvent(
                event_type="brackets_armed",
                order_id=parent_order.order_id,
                market_id=parent_order.market_id,
                status=parent_order.bracket_state,
                message="Bracket orders armed",
                parent_order_id=parent_order.order_id,
                order_snapshot=parent_order,
            ),
            OrderLifecycleEvent(
                event_type="gtt_created",
                order_id=take_profit.order_id,
                market_id=take_profit.market_id,
                status=take_profit.status,
                message="Take-profit armed",
                parent_order_id=parent_order.order_id,
                order_snapshot=take_profit,
            ),
            OrderLifecycleEvent(
                event_type="gtt_created",
                order_id=stop_loss.order_id,
                market_id=stop_loss.market_id,
                status=stop_loss.status,
                message="Stop-loss armed",
                parent_order_id=parent_order.order_id,
                order_snapshot=stop_loss,
            ),
        ]

    async def place_order(
        self,
        market_id: str,
        direction: str,
        size_usdc: float,
        price: float,
        token_id: str = "",
        *,
        bracket_spec: BracketSpec | None = None,
        market_book: dict | None = None,
    ) -> Order | None:
        async with self._order_lock:
            return await self._place_order_unlocked(
                market_id=market_id,
                direction=direction,
                size_usdc=size_usdc,
                price=price,
                token_id=token_id,
                bracket_spec=bracket_spec,
                market_book=market_book,
            )

    async def _place_order_unlocked(
        self,
        market_id: str,
        direction: str,
        size_usdc: float,
        price: float,
        token_id: str = "",
        *,
        bracket_spec: BracketSpec | None = None,
        market_book: dict | None = None,
    ) -> Order | None:
        # --- Pre-flight validation ---
        rejection = self._validate_order(
            market_id=market_id,
            token_id=token_id,
            direction=direction,
            size_usdc=size_usdc,
            price=price,
        )
        if rejection is not None:
            logger.error("Order rejected: %s (market=%s)", rejection, market_id)
            return None

        if market_book is not None and not self.dry_run:
            bids = market_book.get("bids") or []
            asks = market_book.get("asks") or []
            if bids and asks:
                max_slippage_bps = float(self.config.get("max_slippage_bps", 50))
                ok, est = check_slippage_ok(size_usdc, "BUY", bids, asks, max_slippage_bps)
                if not ok:
                    logger.error(
                        "Slippage too high for %s (%s): %.1f bps > %.1f bps",
                        market_id,
                        direction,
                        est.slippage_bps,
                        max_slippage_bps,
                    )
                    return None

        # --- Deduplication ---
        if self._is_duplicate(market_id, direction):
            return None

        limit_price = price - 0.001 if direction == "YES" else (1 - price) - 0.001
        limit_price = self._clamp_price(limit_price)

        if size_usdc > self.split_orders_above:
            n_chunks = min(ceil(size_usdc / 20), 15)  # Cap at CLOB batch limit
            chunk_size = size_usdc / n_chunks
        else:
            n_chunks = 1
            chunk_size = size_usdc

        now = datetime.now(UTC)
        entry_order = self._build_entry_order(
            market_id=market_id,
            direction=direction,
            size_usdc=size_usdc,
            price=limit_price,
            token_id=token_id,
            chunks=n_chunks,
            is_paper=self.dry_run or self.clob is None,
            timestamp=now,
        )
        self.orders[entry_order.order_id] = entry_order
        self._recent_placements[(market_id, direction, "entry")] = now
        bracket = bracket_spec or self._default_bracket_spec()

        if entry_order.is_paper:
            logger.info(
                "PAPER order: %s %s $%.2f @ %.3f (%d chunks) market=%s",
                direction,
                "BUY",
                size_usdc,
                limit_price,
                n_chunks,
                market_id,
            )
            if self.paper_fill_on_place:
                self._apply_fill(entry_order, fill_size_usdc=size_usdc, fill_price=limit_price, now=now)
                if bracket is not None:
                    self._arm_bracket_orders(entry_order, bracket_spec=bracket, now=now)
            return entry_order

        try:
            is_live_trading = not self.dry_run
            if is_live_trading:
                logger.warning("🚨 LIVE TRADING MODE — submitting real order for market %s", market_id)
                client_order_id = str(uuid4())
                if not self._persist_order_intent(client_order_id, market_id, direction, size_usdc):
                    entry_order.status = "cancelled"
                    return None
                await self._assert_sufficient_balance(size_usdc)
                await self._check_gas_price()
            exchange_ids: list[str] = []
            for i in range(n_chunks):
                logger.info(
                    "LIVE chunk %d/%d: %s BUY $%.2f @ %.3f market=%s token=%s",
                    i + 1,
                    n_chunks,
                    direction,
                    chunk_size,
                    limit_price,
                    market_id,
                    token_id,
                )
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.clob.create_and_post_order,
                        {
                            "tokenID": token_id,
                            "price": limit_price,
                            "size": chunk_size,
                            "side": "BUY",
                        },
                    ),
                    timeout=30.0,
                )
                if isinstance(response, dict) and response.get("orderID"):
                    exchange_ids.append(response["orderID"])
                elif isinstance(response, dict) and response.get("order_id"):
                    exchange_ids.append(response["order_id"])
                else:
                    logger.warning(
                        "CLOB chunk %d/%d returned no order ID: %s",
                        i + 1,
                        n_chunks,
                        str(response)[:200],
                    )
                if i < n_chunks - 1:
                    await asyncio.sleep(2)

            entry_order.exchange_order_ids = exchange_ids
            if bracket is not None:
                entry_order.bracket_state = "pending_fill"
            return entry_order
        except TimeoutError:
            logger.error(
                "Timeout placing live order: %s %s $%.2f market=%s",
                direction,
                "BUY",
                size_usdc,
                market_id,
            )
            entry_order.status = "cancelled"
            return None
        except (OSError, ConnectionError) as exc:
            logger.error(
                "Network error placing live order: %s %s $%.2f market=%s: %s",
                direction,
                "BUY",
                size_usdc,
                market_id,
                exc,
            )
            entry_order.status = "cancelled"
            return None
        except Exception:
            logger.exception(
                "Unexpected error placing live order: %s %s $%.2f market=%s",
                direction,
                "BUY",
                size_usdc,
                market_id,
            )
            entry_order.status = "cancelled"
            return None

    def get_order(self, order_id: str) -> Order | None:
        return self.orders.get(order_id)

    async def cancel_order(self, order_id: str) -> bool:
        order = self.orders.get(order_id)
        if order is None:
            logger.warning("Order %s not found", order_id)
            return False
        if order.status in FINAL_ORDER_STATUSES:
            return True

        if order.is_paper or self.clob is None or order.order_kind != "entry":
            order.status = "cancelled"
            if order.order_kind == "entry":
                order.bracket_state = "cancelled"
                for child_id in order.bracket_order_ids:
                    child = self.orders.get(child_id)
                    if child is not None and child.status not in FINAL_ORDER_STATUSES:
                        child.status = "cancelled"
            logger.info("PAPER/LOCAL cancel: %s", order_id)
            return True

        try:
            for eid in order.exchange_order_ids:
                await asyncio.wait_for(
                    asyncio.to_thread(self.clob.cancel, eid),
                    timeout=15.0,
                )
            order.status = "cancelled"
            order.bracket_state = "cancelled"
            logger.info("LIVE cancel: %s (exchange_ids=%s)", order_id, order.exchange_order_ids)
            return True
        except TimeoutError:
            logger.error("Timeout cancelling order %s", order_id)
            return False
        except (OSError, ConnectionError) as exc:
            logger.error("Network error cancelling order %s: %s", order_id, exc)
            return False
        except Exception:
            logger.exception("Unexpected error cancelling order %s", order_id)
            return False

    async def get_order_status(self, order_id: str) -> str:
        order = self.orders.get(order_id)
        if order is None:
            return "unknown"
        return order.status

    async def cancel_all_open_orders(self) -> int:
        cancelled = 0
        for oid, order in list(self.orders.items()):
            if order.status in ("placed", "partially_filled", "armed") and await self.cancel_order(oid):
                cancelled += 1
        logger.info("Cancelled %d open orders", cancelled)
        return cancelled

    async def cancel_expired_orders(self) -> list[str]:
        now = datetime.now(UTC)
        cancelled_ids: list[str] = []
        for oid, order in list(self.orders.items()):
            if order.status not in ("placed", "partially_filled", "armed"):
                continue
            if order.status == "armed" and order.gtt_expiry_at is not None and now > order.gtt_expiry_at:
                order.status = "expired"
                cancelled_ids.append(oid)
                continue
            if order.status in ("placed", "partially_filled"):
                age_minutes = (now - order.timestamp).total_seconds() / 60
                if age_minutes > self.order_expiry_minutes and await self.cancel_order(oid):
                    cancelled_ids.append(oid)
        if cancelled_ids:
            logger.info("Cancelled %d expired orders: %s", len(cancelled_ids), cancelled_ids)
        return cancelled_ids

    async def close_position(
        self,
        market_id: str,
        token_id: str,
        direction: str,
        size_usdc: float,
        price: float,
    ) -> Order | None:
        """Place a SELL order to close (reduce) an existing position.

        On the CLOB this sends a SELL for the same token_id that was originally
        bought.  In paper mode the fill is applied immediately.
        """
        if not (0.0 < price < 1.0):
            logger.error("close_position: invalid price %.4f", price)
            return None
        if size_usdc <= 0:
            logger.error("close_position: invalid size $%.2f", size_usdc)
            return None

        limit_price = self._clamp_price(price)
        now = datetime.now(UTC)

        sell_order = Order(
            order_id=self._next_order_id(),
            market_id=market_id,
            token_id=token_id,
            direction=direction,
            side="SELL",
            size_usdc=size_usdc,
            price=limit_price,
            status="placed",
            timestamp=now,
            chunks=1,
            is_paper=self.dry_run or self.clob is None,
            remaining_size_usdc=size_usdc,
            order_type="limit",
            order_kind="exit",
            reduce_only=True,
        )
        self.orders[sell_order.order_id] = sell_order

        if sell_order.is_paper:
            logger.info(
                "PAPER SELL: %s $%.2f @ %.3f market=%s",
                direction,
                size_usdc,
                limit_price,
                market_id,
            )
            self._apply_fill(sell_order, fill_size_usdc=size_usdc, fill_price=limit_price, now=now)
            return sell_order

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.clob.create_and_post_order,
                    {
                        "tokenID": token_id,
                        "price": limit_price,
                        "size": size_usdc,
                        "side": "SELL",
                    },
                ),
                timeout=30.0,
            )
            eid = None
            if isinstance(response, dict):
                eid = response.get("orderID") or response.get("order_id")
            if eid:
                sell_order.exchange_order_ids = [eid]
            else:
                logger.warning("close_position: CLOB returned no order ID: %s", str(response)[:200])
            return sell_order
        except TimeoutError:
            logger.error("Timeout placing SELL order: %s $%.2f market=%s", direction, size_usdc, market_id)
            sell_order.status = "cancelled"
            return None
        except (OSError, ConnectionError) as exc:
            logger.error("Network error placing SELL: %s: %s", market_id, exc)
            sell_order.status = "cancelled"
            return None
        except Exception:
            logger.exception("Unexpected error placing SELL: %s $%.2f market=%s", direction, size_usdc, market_id)
            sell_order.status = "cancelled"
            return None

    async def reconcile_orders(self, market_books: dict[str, dict]) -> list[OrderLifecycleEvent]:
        async with self._order_lock:
            return await self._reconcile_orders_unlocked(market_books)

    async def _reconcile_orders_unlocked(self, market_books: dict[str, dict]) -> list[OrderLifecycleEvent]:
        events: list[OrderLifecycleEvent] = []
        now = datetime.now(UTC)

        for order in list(self.orders.values()):
            if order.order_kind == "entry":
                if order.status in FINAL_ORDER_STATUSES:
                    continue
                if not order.is_paper and self.live_reconcile_enabled and self.clob is not None:
                    events.extend(await self._reconcile_live_entry(order, now))
                elif order.is_paper and order.status in ("placed", "partially_filled"):
                    market_data = market_books.get(order.market_id)
                    if market_data is None:
                        continue
                    contract_price = self._contract_price(
                        order.direction, float(market_data.get("midpoint", order.price))
                    )
                    if contract_price <= (order.price + self.paper_fill_tolerance):
                        fill_size = order.remaining_size_usdc
                        if self.paper_allow_partial_fills and order.remaining_size_usdc > 0.0:
                            fill_size = min(
                                order.remaining_size_usdc, max(order.size_usdc * self.paper_partial_fill_ratio, 0.0)
                            )
                        previous_status = order.status
                        self._apply_fill(order, fill_size_usdc=fill_size, fill_price=contract_price, now=now)
                        event_type = "entry_filled" if order.status == "filled" else "entry_partially_filled"
                        events.append(
                            OrderLifecycleEvent(
                                event_type=event_type,
                                order_id=order.order_id,
                                market_id=order.market_id,
                                status=order.status,
                                message=f"Entry {order.status}",
                                fill_price=contract_price,
                                fill_size_usdc=fill_size,
                                order_snapshot=order,
                            )
                        )
                        if order.status != previous_status and self.enable_bracket_orders:
                            bracket = self._default_bracket_spec()
                            if bracket is not None:
                                events.extend(self._arm_bracket_orders(order, bracket_spec=bracket, now=now))

                if (
                    order.bracket_state in ("pending_fill", "armed")
                    and order.filled_size_usdc > 0.0
                    and self.enable_bracket_orders
                ):
                    bracket = self._default_bracket_spec()
                    if bracket is not None and not order.bracket_order_ids:
                        events.extend(self._arm_bracket_orders(order, bracket_spec=bracket, now=now))

            elif order.order_kind in ("take_profit", "stop_loss"):
                if order.status != "armed":
                    continue
                market_data = market_books.get(order.market_id)
                if market_data is None:
                    continue
                if order.gtt_expiry_at is not None and now > order.gtt_expiry_at:
                    order.status = "expired"
                    parent = self.orders.get(order.parent_order_id or "")
                    if parent is not None:
                        parent.bracket_state = "expired"
                    events.append(
                        OrderLifecycleEvent(
                            event_type="gtt_expired",
                            order_id=order.order_id,
                            market_id=order.market_id,
                            status=order.status,
                            message="Trigger order expired",
                            parent_order_id=order.parent_order_id,
                            order_snapshot=order,
                        )
                    )
                    continue

                # Use best_bid for take-profit (what we'd actually get selling)
                # and best_ask for stop-loss (worst-case price scenario)
                if order.order_kind == "take_profit":
                    eval_price = float(market_data.get("best_bid", market_data.get("midpoint", order.price)))
                elif order.order_kind == "stop_loss":
                    eval_price = float(market_data.get("best_ask", market_data.get("midpoint", order.price)))
                else:
                    eval_price = float(market_data.get("midpoint", order.price))
                contract_price = self._contract_price(order.direction, eval_price)
                should_trigger = False
                if order.trigger_condition == "gte" and order.trigger_price is not None:
                    should_trigger = contract_price >= order.trigger_price
                elif order.trigger_condition == "lte" and order.trigger_price is not None:
                    should_trigger = contract_price <= order.trigger_price

                if not should_trigger:
                    continue

                self._apply_fill(
                    order,
                    fill_size_usdc=order.remaining_size_usdc or order.size_usdc,
                    fill_price=contract_price,
                    now=now,
                )
                parent = self.orders.get(order.parent_order_id or "")
                if parent is not None:
                    parent.bracket_state = "closed"

                sibling_id: str | None = None
                if order.oco_group_id:
                    for candidate in self.orders.values():
                        if (
                            candidate.oco_group_id == order.oco_group_id
                            and candidate.order_id != order.order_id
                            and candidate.status == "armed"
                        ):
                            candidate.status = "cancelled"
                            sibling_id = candidate.order_id
                            events.append(
                                OrderLifecycleEvent(
                                    event_type="oco_cancelled",
                                    order_id=candidate.order_id,
                                    market_id=candidate.market_id,
                                    status=candidate.status,
                                    message="Sibling bracket order cancelled",
                                    parent_order_id=candidate.parent_order_id,
                                    sibling_order_id=order.order_id,
                                    order_snapshot=candidate,
                                )
                            )

                events.append(
                    OrderLifecycleEvent(
                        event_type="exit_filled",
                        order_id=order.order_id,
                        market_id=order.market_id,
                        status=order.status,
                        message=f"{order.order_kind} triggered",
                        fill_price=contract_price,
                        fill_size_usdc=order.filled_size_usdc,
                        parent_order_id=order.parent_order_id,
                        sibling_order_id=sibling_id,
                        order_snapshot=order,
                    )
                )

        return events

    async def _reconcile_live_entry(self, order: Order, now: datetime) -> list[OrderLifecycleEvent]:
        events: list[OrderLifecycleEvent] = []
        status_rows = []
        for exchange_order_id in order.exchange_order_ids:
            row = await self._fetch_live_exchange_status(exchange_order_id)
            if row is not None:
                status_rows.append(row)
        if not status_rows:
            return events

        total_filled = sum(float(row.get("filled_size_usdc", 0.0)) for row in status_rows)
        last_price = next(
            (
                float(row.get("average_fill_price", order.average_fill_price or order.price))
                for row in reversed(status_rows)
                if row.get("average_fill_price") is not None
            ),
            order.average_fill_price or order.price,
        )
        previous_filled = order.filled_size_usdc
        delta_filled = max(total_filled - previous_filled, 0.0)

        if delta_filled > 0.0:
            self._apply_fill(order, fill_size_usdc=delta_filled, fill_price=last_price, now=now)
            events.append(
                OrderLifecycleEvent(
                    event_type="entry_filled" if order.status == "filled" else "entry_partially_filled",
                    order_id=order.order_id,
                    market_id=order.market_id,
                    status=order.status,
                    message="Live fill reconciliation",
                    fill_price=last_price,
                    fill_size_usdc=delta_filled,
                    order_snapshot=order,
                )
            )

        statuses = {str(row.get("status", "")).lower() for row in status_rows}
        if order.status not in ("filled", "partially_filled"):
            if statuses & {"cancelled", "canceled"}:
                order.status = "cancelled"
                events.append(
                    OrderLifecycleEvent(
                        event_type="entry_cancelled",
                        order_id=order.order_id,
                        market_id=order.market_id,
                        status=order.status,
                        message="Live order cancelled",
                        order_snapshot=order,
                    )
                )
            elif statuses & {"expired"}:
                order.status = "expired"
                events.append(
                    OrderLifecycleEvent(
                        event_type="entry_expired",
                        order_id=order.order_id,
                        market_id=order.market_id,
                        status=order.status,
                        message="Live order expired",
                        order_snapshot=order,
                    )
                )

        if order.filled_size_usdc > 0.0 and self.enable_bracket_orders:
            bracket = self._default_bracket_spec()
            if bracket is not None:
                events.extend(self._arm_bracket_orders(order, bracket_spec=bracket, now=now))
        return events

    async def _fetch_live_exchange_status(self, exchange_order_id: str) -> dict | None:
        if self.clob is None:
            return None

        methods = [
            "get_order",
            "get_order_status",
        ]
        for method_name in methods:
            method = getattr(self.clob, method_name, None)
            if method is None:
                continue
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(method, exchange_order_id),
                    timeout=15.0,
                )
            except TypeError:
                continue
            except TimeoutError:
                logger.warning("Timeout reconciling live order %s via %s", exchange_order_id, method_name)
                return None
            except Exception as exc:
                logger.warning("Failed to reconcile live order %s via %s: %s", exchange_order_id, method_name, exc)
                return None
            parsed = self._parse_live_status_response(response)
            if parsed is not None:
                return parsed
        return None

    # Expected CLOB API response field names (primary → fallbacks)
    _STATUS_FIELDS = ("status", "state", "order_status")
    _FILLED_FIELDS = ("size_matched", "filled_size", "filled", "filled_size_usdc")
    _PRICE_FIELDS = ("associate_trades",)  # computed from trades when available
    _AVG_PRICE_FIELDS = ("avg_price", "average_price", "price")

    def _parse_live_status_response(self, response: object) -> dict | None:
        if not isinstance(response, dict):
            logger.debug("Non-dict CLOB response: %s", type(response).__name__)
            return None

        # Extract status — require at least one recognized field
        status = None
        for key in self._STATUS_FIELDS:
            if key in response:
                status = response[key]
                break
        if status is None:
            logger.warning(
                "CLOB response missing status field (expected one of %s), keys: %s",
                self._STATUS_FIELDS,
                list(response.keys()),
            )
            return None

        # Extract filled size
        filled_raw = None
        for key in self._FILLED_FIELDS:
            if key in response and response[key] is not None:
                filled_raw = response[key]
                break
        try:
            filled_size = float(filled_raw) if filled_raw is not None else 0.0
        except (TypeError, ValueError):
            logger.warning("CLOB filled size not numeric: %r", filled_raw)
            filled_size = 0.0

        # Extract average fill price
        avg_price = None
        for key in self._AVG_PRICE_FIELDS:
            if key in response and response[key] is not None:
                avg_price = response[key]
                break
        try:
            average_fill_price = float(avg_price) if avg_price is not None else None
        except (TypeError, ValueError):
            average_fill_price = None

        return {
            "status": str(status).lower(),
            "filled_size_usdc": filled_size,
            "average_fill_price": average_fill_price,
        }
