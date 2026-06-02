"""Surface ② — agent-initiated memory op display (classify + collapse).
See docs/superpowers/specs/2026-06-02-memory-op-display-design.md.
"""
import os
from pathlib import Path

import pytest

from mini_cc import config


@pytest.fixture
def memdir(tmp_path, monkeypatch):
    """Point the session memdir at a tmp dir and set CWD to tmp_path.

    Patches the PACKAGE attribute mini_cc.memdir.get_auto_mem_path, because
    is_memory_path / safe_path resolve it lazily through the package at call
    time (mirrors tests/test_file_read_memory.py:_point_memdir_at). realpath
    so it matches safe_path's realpath'd `resolved`.
    """
    monkeypatch.setattr(config, "CWD", os.path.realpath(tmp_path))
    d = tmp_path / "mem"
    d.mkdir()
    real = Path(os.path.realpath(d))
    monkeypatch.setattr("mini_cc.memdir.get_auto_mem_path", lambda: real)
    return real


def test_is_memory_path_true_for_file_in_memdir(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(str(memdir / "user_nickname.md")) is True


def test_is_memory_path_false_for_project_file(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(os.path.join(config.CWD, "src", "foo.py")) is False


def test_is_memory_path_true_for_memdir_itself(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(str(memdir)) is True


def test_is_memory_path_true_for_nested_file(memdir):
    from mini_cc.memdir import is_memory_path
    assert is_memory_path(str(memdir / "sub" / "deep.md")) is True


def test_is_memory_path_false_for_adjacent_dir(memdir):
    from mini_cc.memdir import is_memory_path
    # /tmp/mem_extra must not be admitted: it only shares a prefix with the
    # memdir, not the /sep boundary — the + os.sep guard exists for this.
    adjacent = str(memdir) + "_extra"
    assert is_memory_path(os.path.join(adjacent, "x.md")) is False
