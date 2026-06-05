"""Tests for IntervalScheduler."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from polymarket_agent.infra.scheduler import IntervalScheduler


def test_first_run_is_due(tmp_path):
    sched = IntervalScheduler(tmp_path / "state.json")
    assert sched.due("foo", interval_days=1) is True


def test_zero_or_negative_interval_never_due(tmp_path):
    sched = IntervalScheduler(tmp_path / "state.json")
    assert sched.due("foo", interval_days=0) is False
    assert sched.due("foo", interval_days=-1) is False


def test_mark_ran_persists_and_blocks_until_interval(tmp_path):
    state_path = tmp_path / "state.json"
    sched = IntervalScheduler(state_path)
    sched.mark_ran("foo")
    assert sched.due("foo", interval_days=1) is False
    # New instance must read persisted state
    sched2 = IntervalScheduler(state_path)
    assert sched2.due("foo", interval_days=1) is False
    assert sched2.last_run("foo") is not None


def test_due_after_interval_elapsed(tmp_path):
    state_path = tmp_path / "state.json"
    past = (datetime.now(UTC) - timedelta(days=10)).isoformat()
    state_path.write_text(json.dumps({"foo": past}))
    sched = IntervalScheduler(state_path)
    assert sched.due("foo", interval_days=1) is True
    assert sched.due("foo", interval_days=30) is False


def test_corrupt_state_file_is_ignored(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{not valid json")
    sched = IntervalScheduler(state_path)
    assert sched.due("foo", interval_days=1) is True
