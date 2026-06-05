"""File-based kill switch for emergency trading halt.

When the kill switch is active (file exists at the configured path),
ALL trading operations must be halted immediately.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_KILL_SWITCH_PATH = "data/KILL_SWITCH"


class KillSwitch:
    """File-based kill switch that halts all trading when active.

    The switch is backed by a simple file on disk.  When the file exists,
    ``is_active()`` returns ``(True, <reason>)``.  Any component that
    places orders should check ``is_active()`` before proceeding.
    """

    def __init__(self, path: str = DEFAULT_KILL_SWITCH_PATH) -> None:
        self.path: Path = Path(path)

    def activate(self, reason: str) -> None:
        """Create the kill-switch file with a timestamp and reason.

        Args:
            reason: Human-readable explanation for the halt.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).isoformat()
        self.path.write_text(f"{timestamp} | {reason}\n", encoding="utf-8")
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def deactivate(self) -> None:
        """Remove the kill-switch file, allowing trading to resume."""
        try:
            self.path.unlink(missing_ok=True)
            logger.warning("Kill switch deactivated — trading may resume")
        except OSError as exc:
            logger.error("Failed to deactivate kill switch: %s", exc)

    def is_active(self) -> tuple[bool, str]:
        """Check whether the kill switch is currently engaged.

        Returns:
            A ``(active, reason)`` tuple.  *reason* is the file contents
            when active, or an empty string when inactive.
        """
        if not self.path.exists():
            return False, ""
        try:
            content = self.path.read_text(encoding="utf-8").strip()
            return True, content
        except OSError as exc:
            logger.error("Error reading kill switch file: %s", exc)
            # Fail-safe: if we can't read the file, assume active
            return True, f"unreadable kill switch file ({exc})"

    def activate_on_emergency_drawdown(self) -> None:
        """Auto-activate due to emergency drawdown threshold breach."""
        self.activate("Emergency drawdown threshold breached")

    def activate_on_consecutive_failures(self, failure_count: int) -> None:
        """Auto-activate after *failure_count* consecutive order failures.

        Args:
            failure_count: Number of consecutive failures that triggered the halt.
        """
        self.activate(f"{failure_count} consecutive order failures")

    def activate_on_manual_trigger(self, operator: str = "unknown") -> None:
        """Auto-activate via manual / operator trigger.

        Args:
            operator: Identifier of the person or system that triggered the halt.
        """
        self.activate(f"Manual trigger by {operator}")
