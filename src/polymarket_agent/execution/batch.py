"""Batch order optimization: group up to 15 orders per CLOB API call.

Polymarket CLOB supports batch order placement for efficiency.
This module collects orders into batches and submits them together,
reducing API calls and latency.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 15


@dataclass
class PendingOrder:
    """A single order waiting to be batched."""

    token_id: str
    price: float
    size: float
    side: str  # 'BUY' or 'SELL'
    market_id: str = ""
    direction: str = "YES"
    order_type: str = "GTC"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class BatchResult:
    """Result of a batch submission."""

    batch_id: int
    submitted_count: int
    success_count: int
    failed_count: int
    exchange_order_ids: list[str]
    errors: list[str]
    elapsed_ms: float


class BatchOrderManager:
    """Collects orders and submits them in batches of up to 15.

    Usage:
        batcher = BatchOrderManager(clob_client=clob)
        batcher.add(PendingOrder(token_id='abc', price=0.55, size=10, side='BUY'))
        batcher.add(PendingOrder(token_id='def', price=0.40, size=20, side='BUY'))
        results = await batcher.flush()
    """

    def __init__(
        self,
        clob_client=None,
        max_batch_size: int = MAX_BATCH_SIZE,
        dry_run: bool = True,
    ) -> None:
        self.clob = clob_client
        self.max_batch_size = min(max_batch_size, MAX_BATCH_SIZE)
        self.dry_run = dry_run
        self._pending: list[PendingOrder] = []
        self._batch_counter: int = 0
        self._total_submitted: int = 0
        self._total_succeeded: int = 0
        self._total_failed: int = 0

    @property
    def pending_count(self) -> int:
        """Number of orders waiting to be submitted."""
        return len(self._pending)

    @property
    def stats(self) -> dict:
        """Cumulative batch submission statistics."""
        return {
            "batches_sent": self._batch_counter,
            "total_submitted": self._total_submitted,
            "total_succeeded": self._total_succeeded,
            "total_failed": self._total_failed,
            "pending": self.pending_count,
        }

    def add(self, order: PendingOrder) -> int:
        """Add an order to the pending queue.

        Returns:
            Current pending count after adding.
        """
        self._pending.append(order)
        logger.debug(
            "Queued order: %s %s %.0f @ %.3f (pending=%d)",
            order.side,
            order.token_id[:8],
            order.size,
            order.price,
            len(self._pending),
        )
        return len(self._pending)

    def clear(self) -> int:
        """Clear all pending orders without submitting.

        Returns:
            Number of orders cleared.
        """
        count = len(self._pending)
        self._pending.clear()
        logger.info("Cleared %d pending orders", count)
        return count

    async def flush(self) -> list[BatchResult]:
        """Submit all pending orders in batches of max_batch_size.

        Returns:
            List of BatchResult, one per batch submitted.
        """
        if not self._pending:
            return []

        results: list[BatchResult] = []

        while self._pending:
            batch = self._pending[: self.max_batch_size]
            self._pending = self._pending[self.max_batch_size :]

            result = await self._submit_batch(batch)
            results.append(result)

            # Small delay between batches to avoid rate limiting
            if self._pending:
                await asyncio.sleep(0.5)

        return results

    async def flush_if_full(self) -> list[BatchResult]:
        """Submit only if we've accumulated max_batch_size orders."""
        if len(self._pending) >= self.max_batch_size:
            return await self.flush()
        return []

    async def _submit_batch(self, batch: list[PendingOrder]) -> BatchResult:
        """Submit a single batch to the CLOB API."""
        self._batch_counter += 1
        batch_id = self._batch_counter
        start = asyncio.get_event_loop().time()

        logger.info(
            "Submitting batch #%d with %d orders (dry_run=%s)",
            batch_id,
            len(batch),
            self.dry_run,
        )

        if self.dry_run or self.clob is None:
            # Paper mode: simulate success
            exchange_ids = [f"paper-batch{batch_id}-{i}" for i in range(len(batch))]
            elapsed = (asyncio.get_event_loop().time() - start) * 1000

            self._total_submitted += len(batch)
            self._total_succeeded += len(batch)

            logger.info(
                "PAPER batch #%d: %d orders simulated in %.1fms",
                batch_id,
                len(batch),
                elapsed,
            )
            return BatchResult(
                batch_id=batch_id,
                submitted_count=len(batch),
                success_count=len(batch),
                failed_count=0,
                exchange_order_ids=exchange_ids,
                errors=[],
                elapsed_ms=round(elapsed, 1),
            )

        # Live mode: use CLOB batch API
        order_dicts = []
        for order in batch:
            order_dicts.append(
                {
                    "tokenID": order.token_id,
                    "price": order.price,
                    "size": order.size,
                    "side": order.side,
                }
            )

        exchange_ids_live: list[str] = []
        errors: list[str] = []

        try:
            # py-clob-client doesn't have a batch endpoint in all versions,
            # so we fall back to individual submission if batch fails
            if hasattr(self.clob, "create_and_post_batch_order"):
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.clob.create_and_post_batch_order, order_dicts),
                    timeout=30.0,
                )
                if isinstance(response, dict):
                    for oid in response.get("orderIDs", []):
                        exchange_ids_live.append(oid)
                elif isinstance(response, list):
                    for item in response:
                        if isinstance(item, dict) and item.get("orderID"):
                            exchange_ids_live.append(item["orderID"])
            else:
                # Fallback: submit individually with small delays
                logger.warning("Batch API not available, falling back to sequential submission")
                for i, od in enumerate(order_dicts):
                    try:
                        resp = await asyncio.wait_for(
                            asyncio.to_thread(self.clob.create_and_post_order, od),
                            timeout=15.0,
                        )
                        if isinstance(resp, dict):
                            oid = resp.get("orderID") or resp.get("order_id", "")
                            if oid:
                                exchange_ids_live.append(oid)
                    except Exception as exc:
                        errors.append(f"Order {i}: {exc}")
                    if i < len(order_dicts) - 1:
                        await asyncio.sleep(0.2)
        except TimeoutError:
            errors.append(f"Batch #{batch_id} timed out after 30s")
            logger.error("Batch #%d timed out", batch_id)
        except Exception as exc:
            errors.append(f"Batch #{batch_id}: {exc}")
            logger.exception("Batch #%d failed", batch_id)

        elapsed = (asyncio.get_event_loop().time() - start) * 1000
        success = len(exchange_ids_live)
        failed = len(batch) - success

        self._total_submitted += len(batch)
        self._total_succeeded += success
        self._total_failed += failed

        logger.info(
            "Batch #%d: %d/%d succeeded, %d failed in %.1fms",
            batch_id,
            success,
            len(batch),
            failed,
            elapsed,
        )

        return BatchResult(
            batch_id=batch_id,
            submitted_count=len(batch),
            success_count=success,
            failed_count=failed,
            exchange_order_ids=exchange_ids_live,
            errors=errors,
            elapsed_ms=round(elapsed, 1),
        )
