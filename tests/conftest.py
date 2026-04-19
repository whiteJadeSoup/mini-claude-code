"""Shared fixtures for the mini-cc test suite."""
from pathlib import Path

import pytest

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.state import tasks


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-session")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


@pytest.fixture
def fresh_tasks(monkeypatch):
    fresh = tasks.TaskManager(persist_path=None)
    monkeypatch.setattr(tasks, "_tasks", fresh)
    return fresh
