#!/usr/bin/env python3
"""Container healthcheck: verify trading agent is alive and responsive.

Exit 0 = healthy, Exit 1 = unhealthy.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", "/app/data"))
HEARTBEAT_FILE = DATA_DIR / "heartbeat.json"
TRADES_DB = DATA_DIR / "trades.db"
MAX_HEARTBEAT_AGE_SECONDS = 30 * 60  # 30 minutes
MIN_DISK_FREE_MB = 100


def check_heartbeat() -> str | None:
    """Check heartbeat file exists and is fresh (< 30 min old)."""
    if not HEARTBEAT_FILE.exists():
        return f"Heartbeat file missing: {HEARTBEAT_FILE}"

    try:
        with open(HEARTBEAT_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return f"Cannot read heartbeat: {exc}"

    ts = data.get("timestamp")
    if ts is None:
        return "Heartbeat file missing 'timestamp' field"

    age = time.time() - ts
    if age > MAX_HEARTBEAT_AGE_SECONDS:
        return f"Heartbeat stale: {age:.0f}s old (limit {MAX_HEARTBEAT_AGE_SECONDS}s)"

    # Check process is alive
    pid = data.get("pid")
    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return f"Process {pid} from heartbeat is not running"
        except PermissionError:
            pass  # Process exists but we can't signal it — acceptable

    return None


def check_database() -> str | None:
    """Verify trades.db is not corrupted."""
    if not TRADES_DB.exists():
        return None  # DB may not exist yet on first run

    try:
        conn = sqlite3.connect(str(TRADES_DB), timeout=5)
        cursor = conn.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        conn.close()
        if result and result[0] != "ok":
            return f"Database integrity check failed: {result[0]}"
    except sqlite3.Error as exc:
        return f"Database error: {exc}"

    return None


def check_disk_space() -> str | None:
    """Ensure > 100MB free disk space."""
    try:
        usage = shutil.disk_usage(DATA_DIR if DATA_DIR.exists() else "/")
        free_mb = usage.free / (1024 * 1024)
        if free_mb < MIN_DISK_FREE_MB:
            return f"Low disk space: {free_mb:.0f}MB free (need {MIN_DISK_FREE_MB}MB)"
    except OSError as exc:
        return f"Cannot check disk space: {exc}"

    return None


def main() -> int:
    checks = [
        ("heartbeat", check_heartbeat),
        ("database", check_database),
        ("disk_space", check_disk_space),
    ]

    failures: list[str] = []
    for name, check_fn in checks:
        result = check_fn()
        if result:
            failures.append(f"[{name}] {result}")

    if failures:
        for f in failures:
            print(f"UNHEALTHY: {f}", file=sys.stderr)
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
