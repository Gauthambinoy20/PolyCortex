"""Multi-wallet manager with per-wallet allocation and selection strategies."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WalletEntry:
    address: str
    private_key: str
    allocation: float  # fraction of capital allocated to this wallet (0-1)
    utilization: float = 0.0
    trade_count: int = 0
    metadata: dict = field(default_factory=dict)


def _derive_address(private_key: str) -> str:
    try:
        from eth_account import Account
    except ImportError as exc:  # pragma: no cover
        raise ImportError("eth-account required") from exc
    return str(Account.from_key(private_key).address)


class WalletManager:
    """Manage multiple wallets with selectable routing strategies.

    Selection strategies:

    * ``round_robin`` — cycles through wallets in order.
    * ``min_utilization`` — always returns the least-utilized wallet (by
      recorded USDC utilization).
    * ``weighted`` — picks the wallet whose ``(utilization / allocation)``
      ratio is smallest, respecting configured allocations.
    """

    def __init__(self, private_keys: list[str], allocations: list[float] | None = None) -> None:
        if not private_keys:
            raise ValueError("at least one private key required")
        if allocations is None:
            allocations = [1.0 / len(private_keys)] * len(private_keys)
        if len(allocations) != len(private_keys):
            raise ValueError("allocations length must match private_keys")
        total = sum(allocations)
        if total <= 0:
            raise ValueError("total allocation must be > 0")
        allocations = [a / total for a in allocations]

        self._wallets: list[WalletEntry] = []
        for pk, alloc in zip(private_keys, allocations, strict=True):
            pk_clean = pk.strip()
            if not pk_clean:
                continue
            addr = _derive_address(pk_clean)
            self._wallets.append(WalletEntry(address=addr, private_key=pk_clean, allocation=alloc))
        if not self._wallets:
            raise ValueError("no valid private keys after filtering")

        self._rr_index = 0

    @classmethod
    def from_env(cls, env_var: str = "POLYMARKET_PRIVATE_KEYS") -> WalletManager:
        """Load wallets from a comma-separated env var.

        Each entry may optionally include an allocation weight, e.g.
        ``0xkey1:0.7,0xkey2:0.3``.  If allocations are omitted they default to
        equal weights.
        """
        raw = os.environ.get(env_var, "")
        if not raw:
            # Fallback to single-key env var for backwards compatibility
            single = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
            if not single:
                raise ValueError(f"{env_var} and POLYMARKET_PRIVATE_KEY are empty")
            return cls([single])
        keys: list[str] = []
        allocs: list[float] = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                pk, alloc = token.split(":", 1)
                keys.append(pk.strip())
                try:
                    allocs.append(float(alloc))
                except ValueError:
                    allocs.append(1.0)
            else:
                keys.append(token)
                allocs.append(1.0)
        return cls(keys, allocs)

    @property
    def wallets(self) -> list[WalletEntry]:
        return list(self._wallets)

    @property
    def addresses(self) -> list[str]:
        return [w.address for w in self._wallets]

    def __len__(self) -> int:
        return len(self._wallets)

    def get_wallet(self, strategy: str = "round_robin") -> str:
        """Return an address chosen via *strategy*."""
        if strategy == "round_robin":
            entry = self._wallets[self._rr_index % len(self._wallets)]
            self._rr_index += 1
            return entry.address
        if strategy == "min_utilization":
            entry = min(self._wallets, key=lambda w: w.utilization)
            return entry.address
        if strategy == "weighted":

            def _ratio(w: WalletEntry) -> float:
                return (w.utilization + 1e-9) / max(w.allocation, 1e-9)

            entry = min(self._wallets, key=_ratio)
            return entry.address
        raise ValueError(f"unknown strategy: {strategy}")

    def get_private_key(self, address: str) -> str:
        for w in self._wallets:
            if w.address.lower() == address.lower():
                return w.private_key
        raise KeyError(f"wallet not registered: {address}")

    def record_utilization(self, address: str, amount: float) -> None:
        for w in self._wallets:
            if w.address.lower() == address.lower():
                w.utilization += float(amount)
                w.trade_count += 1
                return
        logger.warning("record_utilization: unknown wallet %s", address)

    def reset_utilization(self) -> None:
        for w in self._wallets:
            w.utilization = 0.0
            w.trade_count = 0

    def as_dict(self) -> list[dict]:
        return [
            {
                "address": w.address,
                "allocation": w.allocation,
                "utilization": w.utilization,
                "trade_count": w.trade_count,
            }
            for w in self._wallets
        ]
