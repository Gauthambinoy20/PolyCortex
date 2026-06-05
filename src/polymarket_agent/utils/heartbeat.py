"""Heartbeat / dead-man's-switch for the trading loop."""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """Writes and reads a heartbeat JSON file so external watchers can
    detect a stalled trading process.

    No method ever raises; errors are logged and swallowed.
    """

    def __init__(
        self,
        heartbeat_path: Path,
        max_stale_minutes: int = 30,
    ) -> None:
        self._path = heartbeat_path
        self._max_stale_minutes = max_stale_minutes

        # Ensure parent directory exists.
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def beat(self, cycle_metrics: dict | None = None) -> None:
        """Record a heartbeat, optionally including cycle metrics.

        Args:
            cycle_metrics: Arbitrary dict with keys such as
                ``duration_seconds``, ``markets_scanned``,
                ``trades_placed``, ``errors``.
        """
        record = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "pid": os.getpid(),
            "cycle_metrics": cycle_metrics or {},
        }
        try:
            self._path.write_text(json.dumps(record, default=str), encoding="utf-8")
        except Exception:
            logger.error("Failed to write heartbeat", exc_info=True)

    def is_stale(self) -> tuple[bool, float]:
        """Check whether the last heartbeat is older than the threshold.

        Returns:
            A ``(is_stale, minutes_since_last_beat)`` tuple.  If the
            heartbeat file is missing or unreadable the beat is considered
            stale and minutes is ``float('inf')``.
        """
        data = self.get_last_beat()
        if data is None:
            return True, float("inf")

        try:
            ts_str: str = data["timestamp"]
            last_ts = datetime.fromisoformat(ts_str)
            elapsed_min = (datetime.now(tz=UTC) - last_ts).total_seconds() / 60.0
            return elapsed_min > self._max_stale_minutes, elapsed_min
        except Exception:
            logger.error("Failed to parse heartbeat timestamp", exc_info=True)
            return True, float("inf")

    def get_last_beat(self) -> dict | None:
        """Read and return the most recent heartbeat record, or ``None``."""
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except FileNotFoundError:
            return None
        except Exception:
            logger.error("Failed to read heartbeat file", exc_info=True)
            return None
