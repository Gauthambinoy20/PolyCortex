"""Environment variable validation and secrets management.

Validates required environment variables at startup, provides
clear error messages for missing or invalid values, and ensures
sensitive values aren't accidentally logged.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EnvVar:
    """Definition of an environment variable."""

    name: str
    required: bool = False
    default: str | None = None
    sensitive: bool = False
    description: str = ""
    validator: str | None = None  # regex pattern


@dataclass
class ValidationResult:
    """Result of environment validation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    loaded: dict[str, str] = field(default_factory=dict)


# Standard env var definitions for PolymarketTrader
STANDARD_ENV_VARS = [
    EnvVar(
        name="POLYMARKET_PRIVATE_KEY",
        required=False,  # Only required for live trading
        sensitive=True,
        description="Private key for Polymarket CLOB trading",
        validator=r"^(0x)?[a-fA-F0-9]{64}$",
    ),
    EnvVar(
        name="POLYMARKET_CLOB_URL",
        default="https://clob.polymarket.com",
        description="Polymarket CLOB API endpoint",
    ),
    EnvVar(
        name="POLYMARKET_GAMMA_URL",
        default="https://gamma-api.polymarket.com",
        description="Polymarket Gamma API endpoint",
    ),
    EnvVar(
        name="POLYGON_RPC_URL",
        default="https://polygon-rpc.com",
        description="Polygon RPC endpoint",
    ),
    EnvVar(
        name="ANTHROPIC_API_KEY",
        sensitive=True,
        description="Anthropic API key for Claude sentiment analysis",
    ),
    EnvVar(
        name="TAVILY_API_KEY",
        sensitive=True,
        description="Tavily API key for news search",
    ),
    EnvVar(
        name="TRADING_MODE",
        default="paper",
        description="Trading mode: paper or live",
        validator=r"^(paper|live)$",
    ),
    EnvVar(
        name="LOG_LEVEL",
        default="INFO",
        description="Logging level",
        validator=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    ),
    EnvVar(
        name="DB_PATH",
        default="data/trades.db",
        description="SQLite database path",
    ),
]


def mask_value(value: str, show_chars: int = 4) -> str:
    """Mask a sensitive value, showing only the first few characters."""
    if len(value) <= show_chars:
        return "***"
    return value[:show_chars] + "***" + value[-2:]


def validate_env(
    env_vars: list[EnvVar] | None = None,
    require_live_trading: bool = False,
) -> ValidationResult:
    """Validate environment variables against definitions.

    Args:
        env_vars: List of EnvVar definitions. Uses STANDARD_ENV_VARS if None.
        require_live_trading: If True, POLYMARKET_PRIVATE_KEY becomes required.

    Returns:
        ValidationResult with status, errors, warnings, and loaded values.
    """
    if env_vars is None:
        env_vars = STANDARD_ENV_VARS

    errors: list[str] = []
    warnings: list[str] = []
    loaded: dict[str, str] = {}

    for var in env_vars:
        value = os.environ.get(var.name, "")
        is_required = var.required

        # Special case: live trading needs private key
        if var.name == "POLYMARKET_PRIVATE_KEY" and require_live_trading:
            is_required = True

        if not value:
            if is_required:
                errors.append(f"Missing required: {var.name} — {var.description}")
            elif var.default is not None:
                loaded[var.name] = var.default
                logger.debug(
                    "Using default for %s: %s",
                    var.name,
                    mask_value(var.default) if var.sensitive else var.default,
                )
            else:
                warnings.append(f"Optional not set: {var.name} — {var.description}")
            continue

        # Validate format
        if var.validator and not re.match(var.validator, value):
            errors.append(f"Invalid format for {var.name}: expected pattern {var.validator}")
            continue

        loaded[var.name] = value
        if var.sensitive:
            logger.info("Loaded %s: %s", var.name, mask_value(value))
        else:
            logger.info("Loaded %s: %s", var.name, value)

    valid = len(errors) == 0

    if errors:
        logger.error("Environment validation FAILED with %d error(s):", len(errors))
        for e in errors:
            logger.error("  ✗ %s", e)
    if warnings:
        for w in warnings:
            logger.warning("  ⚠ %s", w)
    if valid:
        logger.info("Environment validation passed (%d variables loaded)", len(loaded))

    return ValidationResult(valid=valid, errors=errors, warnings=warnings, loaded=loaded)


def get_env(name: str, default: str | None = None, sensitive: bool = False) -> str:
    """Get an environment variable with logging.

    Args:
        name: Variable name.
        default: Default value if not set.
        sensitive: If True, mask the value in logs.

    Returns:
        The environment variable value.

    Raises:
        RuntimeError: If variable is not set and no default provided.
    """
    value = os.environ.get(name, "")
    if not value:
        if default is not None:
            return default
        raise RuntimeError(f"Required environment variable not set: {name}")

    if sensitive:
        logger.debug("Read %s: %s", name, mask_value(value))

    return value


def validate_private_key(key: str) -> str:
    """Validate a private key string and return the cleaned hex (no 0x prefix).

    Raises:
        ValueError: If the key is invalid.
    """
    cleaned = key.strip().removeprefix("0x")
    if len(cleaned) != 64:
        raise ValueError(f"POLYMARKET_PRIVATE_KEY must be 64 hex chars (got {len(cleaned)}). Check your .env file.")
    if not all(c in "0123456789abcdefABCDEF" for c in cleaned):
        raise ValueError("POLYMARKET_PRIVATE_KEY contains non-hex characters. Check your .env file.")
    try:
        from eth_account import Account

        acct = Account.from_key(bytes.fromhex(cleaned))
        addr_suffix = acct.address[-4:]
        logger.info("Private key validated. Wallet address ends in: ...%s", addr_suffix)
    except ImportError:
        logger.warning("eth_account not installed; skipping address derivation check")
    except Exception as exc:
        raise ValueError(f"Failed to derive address from POLYMARKET_PRIVATE_KEY: {exc}") from exc
    return cleaned
