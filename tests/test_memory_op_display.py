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


# ---- classify_memory_op (T5-T8) ----

def test_classify_read_write_search(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    p = str(memdir / "m.md")
    assert classify_memory_op("file_read", {"path": p}) == "read"
    assert classify_memory_op("file_edit", {"path": p}) == "write"
    assert classify_memory_op("file_write", {"path": p}) == "write"
    assert classify_memory_op("grep", {"pattern": "x", "path": str(memdir)}) == "search"
    assert classify_memory_op("glob", {"pattern": "*.md", "path": str(memdir)}) == "search"


def test_classify_none_for_project_file(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    p = os.path.join(config.CWD, "src", "a.py")
    assert classify_memory_op("file_read", {"path": p}) is None


def test_classify_none_for_search_without_path(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    assert classify_memory_op("grep", {"pattern": "x"}) is None


def test_classify_none_for_escape_path(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    # safe_path rejects cwd+memdir escapes → ValueError → not a memory op
    assert classify_memory_op("file_read", {"path": "../../../../etc/passwd"}) is None


# ---- memory_run_summary (T9-T11) ----

def test_summary_multi_kind_join():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(2, 1, 0, active=False) == "Recalled 2 memories · wrote 1 memory"


def test_summary_search_has_no_count():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(0, 0, 1, active=True) == "Searching memories"


def test_summary_singular():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(1, 0, 0, active=False) == "Recalled 1 memory"


def test_summary_empty_is_blank():
    from mini_cc.consumers.tui.memory_ops import memory_run_summary
    assert memory_run_summary(0, 0, 0, active=False) == ""


# ---- MemoryRun state machine (T12-T14, T16) ----

def test_run_collapses_consecutive_reads(memdir):
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    p = str(memdir / "m.md")
    for _ in range(3):
        assert r.absorb("file_read", {"path": p}) is True
    assert r.is_open is True
    assert r.flush() == "Recalled 3 memories"
    assert r.is_open is False
    assert r.flush() is None  # idempotent close


def test_run_breaks_on_non_memory_tool(memdir):
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    p = str(memdir / "m.md")
    r.absorb("file_read", {"path": p})
    r.absorb("file_read", {"path": p})
    assert r.absorb("execute_command", {"command": "ls"}) is False
    assert r.flush() == "Recalled 2 memories"
    assert r.absorb("file_read", {"path": p}) is True
    assert r.flush() == "Recalled 1 memory"


def test_run_merges_across_steps(memdir):
    # No non-memory tool between reads → one run regardless of LLM step boundary.
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    p = str(memdir / "m.md")
    for _ in range(3):
        r.absorb("file_read", {"path": p})
    assert r.flush() == "Recalled 3 memories"


def test_run_counts_op_regardless_of_later_error(memdir):
    # absorb happens at add-time; a later failed result doesn't change the count.
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    r.absorb("file_read", {"path": str(memdir / "m.md")})
    assert r.flush() == "Recalled 1 memory"


def test_run_live_summary_present_tense(memdir):
    from mini_cc.consumers.tui.memory_ops import MemoryRun
    r = MemoryRun()
    assert r.live_summary() is None            # no open run
    p = str(memdir / "m.md")
    r.absorb("file_read", {"path": p})
    r.absorb("file_read", {"path": p})
    assert r.live_summary() == "Recalling 2 memories"


def test_classify_none_for_unknown_tool(memdir):
    from mini_cc.consumers.tui.memory_ops import classify_memory_op
    # Unknown tool names short-circuit to None before any path handling.
    assert classify_memory_op("plan_todos", {"path": str(memdir / "m.md")}) is None


# ---- ToolStatus wiring (T12 end-to-end, T15 sub-tool) ----

def _make_tool_status():
    """Instantiate ToolStatus unmounted; stub post_message to capture flushes.
    The exercised paths (add_tool / _flush_mem_run) only touch post_message —
    no Textual screen needed."""
    from mini_cc.consumers.tui.app import ToolStatus
    ts = ToolStatus()
    captured = []
    ts.post_message = lambda m: captured.append(m)
    return ts, captured


def test_toolstatus_flushes_one_memory_row_when_broken(memdir):
    from mini_cc.consumers.tui.app import ToolFlushed
    ts, captured = _make_tool_status()
    p = str(memdir / "m.md")
    ts.add_tool("c0", "file_read", {"path": p}, "", "a0", None)
    ts.add_tool("c1", "file_read", {"path": p}, "", "a1", None)
    assert ts._mem_run.is_open is True
    assert captured == []  # nothing flushed mid-run
    # A non-memory top-level tool breaks the run → exactly one collapsed row.
    ts.add_tool("c2", "execute_command", {"command": "ls"}, "", "a2", None)
    flushed = [m for m in captured if isinstance(m, ToolFlushed)]
    assert len(flushed) == 1
    assert "Recalled 2 memories" in flushed[0].markup
    assert ts._mem_run.is_open is False


def test_toolstatus_subtool_memory_read_not_folded(memdir):
    ts, captured = _make_tool_status()
    # parent_id set → sub-tool path; must NOT open a memory run.
    ts.add_tool("s1", "file_read", {"path": str(memdir / "m.md")},
                "  ", "a1", "missing-parent")
    assert ts._mem_run.is_open is False
    assert captured == []


def test_toolstatus_flushes_memory_row_on_turn_end(memdir):
    from mini_cc.consumers.tui.app import ToolFlushed
    ts, captured = _make_tool_status()
    # end_turn() calls update()/remove_class() that assume a mounted widget;
    # stub them so we can exercise the flush path in isolation.
    ts.update = lambda *a, **k: None
    ts.remove_class = lambda *a, **k: None
    p = str(memdir / "m.md")
    ts.add_tool("c0", "file_read", {"path": p}, "", "a0", None)
    ts.add_tool("c1", "file_read", {"path": p}, "", "a1", None)
    assert ts._mem_run.is_open is True
    ts.end_turn()
    flushed = [m for m in captured if isinstance(m, ToolFlushed)]
    assert len(flushed) == 1
    assert "Recalled 2 memories" in flushed[0].markup
    assert ts._mem_run.is_open is False
