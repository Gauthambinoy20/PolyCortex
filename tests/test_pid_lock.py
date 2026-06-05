"""Tests for infra.pid_lock."""

from __future__ import annotations

import os
import sys

import pytest

from polymarket_agent.infra.pid_lock import PidLock


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fcntl not available on Windows")
def test_acquires_and_releases(tmp_path):
    lock_path = tmp_path / "lock"
    lock = PidLock(str(lock_path))
    assert lock.acquire() is True
    assert lock_path.exists()
    content = lock_path.read_text().strip()
    assert int(content) == os.getpid()
    lock.release()
    assert not lock_path.exists()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="fcntl not available on Windows")
def test_second_instance_fails(tmp_path):
    lock_path = tmp_path / "lock"
    a = PidLock(str(lock_path))
    b = PidLock(str(lock_path))
    assert a.acquire() is True
    assert b.acquire() is False
    a.release()
    # After release, b can acquire
    assert b.acquire() is True
    b.release()


def test_context_manager(tmp_path):
    lock_path = tmp_path / "lock"
    with PidLock(str(lock_path)) as lock:
        assert lock_path.exists()
    assert not lock_path.exists()


def test_context_manager_raises_if_held(tmp_path):
    lock_path = tmp_path / "lock"
    a = PidLock(str(lock_path))
    assert a.acquire()
    try:
        with pytest.raises(RuntimeError), PidLock(str(lock_path)):
            pass
    finally:
        a.release()
