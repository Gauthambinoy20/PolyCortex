"""Structured JSON logging configuration with file rotation."""

import contextvars
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import MutableMapping
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

__all__ = [
    "setup_logging",
    "JSONFormatter",
    "CycleLoggerAdapter",
    "SensitiveFilter",
    "generate_cycle_id",
    "set_trace_id",
    "set_market_id",
    "set_order_id",
    "get_trace_id",
]

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_market_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("market_id", default="")
_order_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("order_id", default="")


def set_trace_id(value: str) -> None:
    _trace_id_var.set(value)


def set_market_id(value: str) -> None:
    _market_id_var.set(value)


def set_order_id(value: str) -> None:
    _order_id_var.set(value)


def get_trace_id() -> str:
    return _trace_id_var.get()


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("cycle_id", "market_id", "order_id", "component"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        # Inject context-var values when not already set on record
        for ctx_key, ctx_var in (
            ("trace_id", _trace_id_var),
            ("market_id", _market_id_var),
            ("order_id", _order_id_var),
        ):
            if ctx_key not in log_entry:
                ctx_val = ctx_var.get()
                if ctx_val:
                    log_entry[ctx_key] = ctx_val
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


class CycleLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that injects cycle_id and optional context into every log record.

    Usage:
        base_logger = logging.getLogger(__name__)
        cycle_logger = CycleLoggerAdapter(base_logger, cycle_id="a1b2c3d4")
        cycle_logger.info("Processing market", extra={"market_id": "0x123"})
        # Output: {"ts": "...", "level": "INFO", ..., "cycle_id": "a1b2c3d4", "market_id": "0x123"}
    """

    def __init__(self, logger: logging.Logger, cycle_id: str, **kwargs: str) -> None:
        super().__init__(logger, {"cycle_id": cycle_id, **kwargs})

    def process(self, msg: Any, kwargs: MutableMapping[str, Any]) -> tuple[Any, MutableMapping[str, Any]]:
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


class SensitiveFilter(logging.Filter):
    """Filter that redacts potential private keys and API secrets from log output."""

    _PATTERNS = [
        (re.compile(r"0x[a-fA-F0-9]{64}"), "0x***REDACTED_KEY***"),
        (re.compile(r"(?i)(api[_-]?key|secret|passphrase|password|token)\s*[:=]\s*\S+"), r"\1=***REDACTED***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, replacement in self._PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        return True


def generate_cycle_id() -> str:
    """Generate a short unique ID for correlating logs within a single trading cycle."""
    return uuid.uuid4().hex[:8]


def setup_logging(
    *,
    level: str | None = None,
    json_output: bool = True,
    log_dir: str | None = None,
    log_format: str | None = None,
) -> None:
    """Configure root logger with optional JSON output and file rotation.

    Args:
        level: Log level string. Falls back to LOG_LEVEL env var, then INFO.
        json_output: Whether to use JSON formatting (ignored if log_format is set).
        log_dir: Directory for log files. If set, adds a rotating file handler.
        log_format: Either "json" or "text". Overrides json_output if provided.
    """
    if log_format is not None:
        json_output = log_format.lower() == "json"
    resolved_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved_level, logging.INFO))

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    formatter: logging.Formatter
    if json_output:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler (stdout)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    sensitive_filter = SensitiveFilter()
    console.addFilter(sensitive_filter)
    root.addHandler(console)

    # File handler (rotating) if log_dir specified
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / "agent.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        root.addHandler(file_handler)

    # Quiet noisy libraries
    for name in ("aiohttp", "urllib3", "httpx", "hmmlearn"):
        logging.getLogger(name).setLevel(logging.WARNING)
