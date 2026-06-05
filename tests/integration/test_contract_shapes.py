"""Contract tests: ensure order/tx shapes match py-clob-client expectations."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_signature_types_match_clob_spec():
    """Polymarket CLOB spec: EOA=0, MAGIC=1, GNOSIS=2."""
    from polymarket_agent.constants import (
        SIGNATURE_TYPE_EOA,
        SIGNATURE_TYPE_GNOSIS,
        SIGNATURE_TYPE_MAGIC,
    )

    assert SIGNATURE_TYPE_EOA == 0
    assert SIGNATURE_TYPE_MAGIC == 1
    assert SIGNATURE_TYPE_GNOSIS == 2


def test_usdc_integer_units():
    """All USDC amounts crossing the wire must be integers in 1e-6 units."""
    from polymarket_agent.constants import USDC_DECIMALS

    def to_wire(amount_usdc: float) -> int:
        return int(round(amount_usdc * 10**USDC_DECIMALS))

    assert to_wire(1.0) == 1_000_000
    assert to_wire(0.5) == 500_000
    assert to_wire(123.456789) == 123_456_789


def test_chain_id_is_polygon_mainnet():
    from polymarket_agent.constants import POLYGON_CHAIN_ID

    assert POLYGON_CHAIN_ID == 137


def test_exchange_addresses_are_checksummed():
    # eth_utils (bundled with eth-account) provides the same EIP-55 check as
    # web3.Web3.is_checksum_address without the heavy web3 dependency.
    from eth_utils import is_checksum_address

    from polymarket_agent.constants import POLYMARKET_EXCHANGE, POLYMARKET_NEG_RISK_EXCHANGE

    assert is_checksum_address(POLYMARKET_EXCHANGE)
    assert is_checksum_address(POLYMARKET_NEG_RISK_EXCHANGE)


def test_order_metadata_shape():
    """The Order dataclass must expose fields used by downstream accounting."""
    from polymarket_agent.execution.executor import Order

    fields = {f for f in Order.__dataclass_fields__}
    required = {
        "order_id",
        "market_id",
        "token_id",
        "direction",
        "size_usdc",
        "price",
        "status",
    }
    assert required.issubset(fields)
