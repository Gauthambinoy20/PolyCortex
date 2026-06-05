"""Tests for polymarket_agent.constants."""

from __future__ import annotations

from polymarket_agent import constants


def test_usdc_decimals():
    """Ensure USDC uses 6 decimals, not 18."""
    assert constants.USDC_DECIMALS == 6
    one_usdc_raw = 1_000_000
    assert one_usdc_raw == 10**constants.USDC_DECIMALS


def test_signature_types():
    assert constants.SIGNATURE_TYPE_EOA == 0
    assert constants.SIGNATURE_TYPE_MAGIC == 1
    assert constants.SIGNATURE_TYPE_GNOSIS == 2
    # All distinct
    assert (
        len(
            {
                constants.SIGNATURE_TYPE_EOA,
                constants.SIGNATURE_TYPE_MAGIC,
                constants.SIGNATURE_TYPE_GNOSIS,
            }
        )
        == 3
    )


def test_polymarket_addresses_are_checksummed():
    # eth_utils ships with eth-account (a runtime dependency) and provides the
    # same EIP-55 check as web3.Web3.is_checksum_address, so we avoid pulling in
    # the heavy web3 package just for string validation.
    from eth_utils import is_checksum_address

    assert is_checksum_address(constants.USDC_ADDRESS)
    assert is_checksum_address(constants.POLYMARKET_EXCHANGE)
    assert is_checksum_address(constants.POLYMARKET_NEG_RISK_EXCHANGE)
    assert is_checksum_address(constants.CONDITIONAL_TOKENS)


def test_chain_ids():
    assert constants.POLYGON_CHAIN_ID == 137
    assert constants.MUMBAI_CHAIN_ID == 80001
    assert constants.AMOY_CHAIN_ID == 80002
