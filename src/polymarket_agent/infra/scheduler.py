"""Lightweight persistent interval scheduler for periodic background jobs.

Keeps last-run timestamps in a JSON file so that intervals survive process
restarts. Intentionally tiny — for anything fancier use cron/systemd timers.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class IntervalScheduler:
    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self._state: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            self._state = json.loads(self.state_path.read_text() or "{}")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load scheduler state %s: %s", self.state_path, exc)
            self._state = {}

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(self._state, indent=2))
        except OSError as exc:
            logger.warning("Could not save scheduler state %s: %s", self.state_path, exc)

    def last_run(self, job_name: str) -> datetime | None:
        ts = self._state.get(job_name)
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None

    def due(self, job_name: str, interval_days: float) -> bool:
        if interval_days <= 0:
            return False
        last = self.last_run(job_name)
        if last is None:
            return True
        return datetime.now(UTC) - last >= timedelta(days=interval_days)

    def mark_ran(self, job_name: str) -> None:
        self._state[job_name] = datetime.now(UTC).isoformat()
        self._save()
