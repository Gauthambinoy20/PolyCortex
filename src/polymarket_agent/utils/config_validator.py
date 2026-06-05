"""Config validation and API response sanitization for production safety."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, cast


class ConfigValidationError(Exception):
    """Raised when trading configuration has critical issues."""

    pass


def validate_config(config: dict) -> list[str]:
    """Validate trading config. Returns list of warnings. Raises ConfigValidationError for critical issues."""
    errors: list[str] = []
    warnings: list[str] = []

    # ── Critical checks (raise) ──

    bankroll = config.get("bankroll")
    if bankroll is None or not isinstance(bankroll, (int, float)) or bankroll <= 0:
        errors.append("bankroll must be > 0")

    kelly = config.get("kelly_fraction")
    if kelly is None or not isinstance(kelly, (int, float)) or not (0 < kelly <= 1.0):
        errors.append("kelly_fraction must be 0 < x <= 1.0")

    max_pos = config.get("max_position_pct")
    if max_pos is None or not isinstance(max_pos, (int, float)) or not (0 < max_pos <= 1.0):
        errors.append("max_position_pct must be 0 < x <= 1.0")

    max_port = config.get("max_portfolio_pct")
    if max_port is None or not isinstance(max_port, (int, float)) or not (0 < max_port <= 1.0):
        errors.append("max_portfolio_pct must be 0 < x <= 1.0")

    scan_interval = config.get("scan_interval_minutes")
    if scan_interval is None or not isinstance(scan_interval, (int, float)) or scan_interval < 1:
        errors.append("scan_interval_minutes must be >= 1")

    api_timeout = config.get("api_timeout")
    if api_timeout is None or not isinstance(api_timeout, (int, float)) or api_timeout <= 0:
        errors.append("api_timeout must be > 0")

    # Drawdown thresholds: all must be 0 < x < 1.0 and ordered
    dd_reduce = config.get("drawdown_reduce")
    dd_stop = config.get("drawdown_stop")
    dd_emergency = config.get("drawdown_emergency")

    for name, val in [
        ("drawdown_reduce", dd_reduce),
        ("drawdown_stop", dd_stop),
        ("drawdown_emergency", dd_emergency),
    ]:
        if val is None or not isinstance(val, (int, float)) or not (0 < val < 1.0):
            errors.append(f"{name} must be 0 < x < 1.0")

    if (
        isinstance(dd_reduce, (int, float))
        and isinstance(dd_stop, (int, float))
        and isinstance(dd_emergency, (int, float))
        and 0 < dd_reduce < 1.0
        and 0 < dd_stop < 1.0
        and 0 < dd_emergency < 1.0
        and not (dd_reduce < dd_stop < dd_emergency)
    ):
        errors.append(
            "drawdown_reduce < drawdown_stop < drawdown_emergency required "
            f"(got {dd_reduce} < {dd_stop} < {dd_emergency})"
        )

    if errors:
        raise ConfigValidationError("Critical config errors:\n" + "\n".join(f"  - {e}" for e in errors))

    # ── Warning checks (return) ──

    if isinstance(bankroll, (int, float)) and bankroll > 10000:
        warnings.append("Large bankroll detected, verify intentional")

    if isinstance(kelly, (int, float)) and kelly > 0.5:
        warnings.append("Aggressive Kelly fraction")

    if config.get("dry_run") is False:
        warnings.append("LIVE TRADING MODE ENABLED")

    if isinstance(max_pos, (int, float)) and max_pos > 0.10:
        warnings.append("Large position limit")

    return warnings


def validate_environment() -> list[str]:
    """Check required env vars. Returns list of issues."""
    issues: list[str] = []

    pk = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        issues.append("POLYMARKET_PRIVATE_KEY is not set")
    elif not pk.startswith("0x") or len(pk) != 66:
        issues.append("POLYMARKET_PRIVATE_KEY format invalid (should start with 0x and be 66 chars)")

    anthropic = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic:
        issues.append("ANTHROPIC_API_KEY is not set")
    elif not anthropic.startswith("sk-ant-"):
        issues.append("ANTHROPIC_API_KEY format invalid (should start with sk-ant-)")

    tavily = os.environ.get("TAVILY_API_KEY", "")
    if not tavily:
        issues.append("TAVILY_API_KEY is not set")

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    if tg_token and not tg_chat:
        issues.append("TELEGRAM_BOT_TOKEN set but TELEGRAM_CHAT_ID is missing")
    if tg_chat and not tg_token:
        issues.append("TELEGRAM_CHAT_ID set but TELEGRAM_BOT_TOKEN is missing")

    return issues


_SECRET_PATTERNS = re.compile(
    r"(password|api_key|api_secret|passphrase|private_key|secret_key|"
    r"credential|auth_token|bearer|mnemonic)",
    re.IGNORECASE,
)

_MAX_STRING_LEN = 10000
_MAX_LIST_LEN = 1000


def sanitize_api_response(data: dict | list | None, max_depth: int = 10) -> dict | list | None:
    """Recursively sanitize API response data.

    - Removes keys that look like secrets
    - Truncates strings > 10000 chars
    - Caps list lengths at 1000 items
    - Prevents deeply nested structures (max_depth)
    - Returns sanitized copy (does not mutate input)
    """
    if data is None:
        return None

    return cast("dict | list | None", _sanitize(data, max_depth, current_depth=0))


def _sanitize(data: object, max_depth: int, current_depth: int) -> object:
    if current_depth >= max_depth:
        return "<truncated: max depth exceeded>"

    if isinstance(data, dict):
        result: dict[Any, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and _SECRET_PATTERNS.search(key):
                result[key] = "***REDACTED***"
            else:
                result[key] = _sanitize(value, max_depth, current_depth + 1)
        return result

    if isinstance(data, list):
        truncated = data[:_MAX_LIST_LEN]
        return [_sanitize(item, max_depth, current_depth + 1) for item in truncated]

    if isinstance(data, str):
        if len(data) > _MAX_STRING_LEN:
            return data[:_MAX_STRING_LEN] + f"...<truncated, total {len(data)} chars>"
        return data

    # Primitives (int, float, bool, None nested) pass through
    return copy.deepcopy(data) if isinstance(data, (dict, list)) else data
