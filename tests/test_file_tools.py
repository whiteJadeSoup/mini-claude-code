"""Tests for file_read / file_write / file_edit tools and the file_read_state cache.

Covers Goals G1-G7 from the design plan, three-path matrix (Happy/Edge/Error) per Goal.
Uses real filesystem (tmp_path fixture) — file_read_state._state is the system under
test, mocking it would defeat the purpose. mtime-based staleness is exercised with
real os.utime calls (T15) for true Windows-mtime-jitter coverage.
"""
import asyncio
import os
from pathlib import Path

import pytest

from mini_cc import config
from mini_cc.state import file_read_state
from mini_cc.tools._utils import _sub_agent_scope
from mini_cc.tools.base import (
    FileEditOutput,
    FileReadOutput,
    FileWriteOutput,
    ToolErrorOutput,
    get_tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workdir(tmp_path: Path, monkeypatch):
    """Sandbox CWD = tmp_path; reset file_read_state before/after each test."""
    monkeypatch.setattr(config, "CWD", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    file_read_state._state.clear()
    yield tmp_path
    file_read_state._state.clear()


@pytest.fixture
def fr():
    return get_tool("file_read")


@pytest.fixture
def fw():
    return get_tool("file_write")


@pytest.fixture
def fe():
    return get_tool("file_edit")


def _arun(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# G1: line-numbered paginated read
# ---------------------------------------------------------------------------

def test_T01_read_simple_full_file(workdir, fr):
    (workdir / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 51)), encoding="utf-8")
    out = _arun(fr.execute(path="a.txt"))
    assert isinstance(out, FileReadOutput)
    assert out.total_lines == 50
    assert out.returned_lines == 50
    assert out.start_line == 1
    assert out.truncated_by_limit is False
    assert "     1\tline1" in out.content
    assert "    50\tline50" in out.content


def test_T02_read_offset_limit(workdir, fr):
    (workdir / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 51)), encoding="utf-8")
    out = _arun(fr.execute(path="a.txt", offset=10, limit=20))
    assert isinstance(out, FileReadOutput)
    assert out.start_line == 10
    assert out.returned_lines == 20
    assert out.total_lines == 50
    assert "    10\tline10" in out.content
    assert "    29\tline29" in out.content
    assert "line30" not in out.content
    assert "line9" not in out.content


def test_T03_read_truncated_by_limit(workdir, fr):
    (workdir / "big.txt").write_text("\n".join(f"L{i}" for i in range(1, 5001)), encoding="utf-8")
    out = _arun(fr.execute(path="big.txt"))
    assert isinstance(out, FileReadOutput)
    assert out.returned_lines == 2000
    assert out.total_lines == 5000
    assert out.truncated_by_limit is True


def test_T04_read_offset_beyond_end(workdir, fr):
    (workdir / "a.txt").write_text("\n".join(f"line{i}" for i in range(1, 51)), encoding="utf-8")
    out = _arun(fr.execute(path="a.txt", offset=999, limit=10))
    assert isinstance(out, FileReadOutput)
    assert out.returned_lines == 0
    api = out.to_api_str()
    assert "50" in api
    assert "999" in api
    assert "beyond the end" in api


def test_T05_read_long_line_truncation(workdir, fr):
    (workdir / "long.txt").write_text("x" * 3000 + "\nshort", encoding="utf-8")
    out = _arun(fr.execute(path="long.txt"))
    first_line = out.content.split("\n")[0]
    assert "...[line truncated]" in first_line
    # The line content portion (after `<num>\t`) is truncated to MAX_LINE_CHARS=2000
    content_part = first_line.split("\t", 1)[1]
    assert len(content_part) == 2000 + len("...[line truncated]")


def test_T06_read_binary_rejected(workdir, fr):
    (workdir / "binary.bin").write_bytes(b"\x00\x01\xff\xfe\xfd")
    out = _arun(fr.execute(path="binary.bin"))
    assert isinstance(out, ToolErrorOutput)
    assert "not UTF-8" in out.message
    assert "execute_command" in out.message  # actionable next-step


# ---------------------------------------------------------------------------
# G2: uniqueness gate (file_edit)
# ---------------------------------------------------------------------------

def test_T07_edit_unique_match(workdir, fr, fe):
    (workdir / "a.py").write_text("foo bar baz", encoding="utf-8")
    _arun(fr.execute(path="a.py"))
    out = _arun(fe.execute(path="a.py", old_string="bar", new_string="BAR"))
    assert isinstance(out, FileEditOutput)
    assert out.replaced is True
    assert out.replace_count == 1
    assert (workdir / "a.py").read_text(encoding="utf-8") == "foo BAR baz"


def test_T08_edit_replace_all(workdir, fr, fe):
    (workdir / "a.py").write_text("foo bar foo baz foo", encoding="utf-8")
    _arun(fr.execute(path="a.py"))
    out = _arun(fe.execute(path="a.py", old_string="foo", new_string="FOO", replace_all=True))
    assert isinstance(out, FileEditOutput)
    assert out.replaced is True
    assert out.replace_count == 3
    assert "All occurrences" in out.to_api_str()
    assert (workdir / "a.py").read_text(encoding="utf-8") == "FOO bar FOO baz FOO"


def test_T09_edit_multi_match_rejected_with_linenos(workdir, fr, fe):
    (workdir / "m.py").write_text(
        "def helper():\n    pass\n\ndef helper():\n    pass\n\ndef helper():\n    pass\n",
        encoding="utf-8",
    )
    _arun(fr.execute(path="m.py"))
    out = _arun(fe.execute(path="m.py", old_string="def helper():", new_string="def renamed():"))
    assert isinstance(out, ToolErrorOutput)
    assert "Found 3 occurrences" in out.message
    assert "lines [1, 4, 7]" in out.message
    assert "replace_all=true" in out.message
    assert "extend old_string" in out.message  # actionable second option
    # Disk untouched
    assert (workdir / "m.py").read_text(encoding="utf-8").count("def helper():") == 3


def test_T10_edit_old_string_not_found(workdir, fr, fe):
    (workdir / "a.py").write_text("foo bar baz", encoding="utf-8")
    _arun(fr.execute(path="a.py"))
    out = _arun(fe.execute(path="a.py", old_string="QQQ", new_string="X"))
    assert isinstance(out, ToolErrorOutput)
    assert "not found" in out.message
    assert "whitespace" in out.message  # one of two recovery hints
    assert "edited" in out.message  # second hint about external edits


def test_T11_edit_old_eq_new(workdir, fr, fe):
    (workdir / "a.py").write_text("hello", encoding="utf-8")
    _arun(fr.execute(path="a.py"))
    out = _arun(fe.execute(path="a.py", old_string="X", new_string="X"))
    assert isinstance(out, ToolErrorOutput)
    assert "identical" in out.message


# ---------------------------------------------------------------------------
# G3: content-aware staleness
# ---------------------------------------------------------------------------

def test_T12_read_then_edit_passes_gate(workdir, fr, fe):
    (workdir / "a.py").write_text("foo bar", encoding="utf-8")
    _arun(fr.execute(path="a.py"))
    out = _arun(fe.execute(path="a.py", old_string="foo", new_string="FOO"))
    assert isinstance(out, FileEditOutput) and out.replaced
    # Entry should now be the post-edit self-record
    p = config.safe_path("a.py")
    entry = file_read_state._state.get(p)
    assert entry is not None
    assert entry.offset is None and entry.limit is None
    assert entry.content == "FOO bar"


def test_T13_edit_unread_rejected(workdir, fe):
    # File exists on disk but never went through file_read
    (workdir / "a.py").write_text("foo", encoding="utf-8")
    out = _arun(fe.execute(path="a.py", old_string="foo", new_string="FOO"))
    assert isinstance(out, ToolErrorOutput)
    assert "has not been read yet" in out.message
    assert "file_read('a.py')" in out.message  # actionable
    # Disk untouched
    assert (workdir / "a.py").read_text() == "foo"


def test_T14_external_change_rejects_edit(workdir, fr, fe):
    p = workdir / "shared.py"
    p.write_text("v1", encoding="utf-8")
    _arun(fr.execute(path="shared.py"))
    # Push mtime forward AND change content
    import time
    time.sleep(0.05)
    p.write_text("externally changed", encoding="utf-8")
    out = _arun(fe.execute(path="shared.py", old_string="changed", new_string="STUFF"))
    assert isinstance(out, ToolErrorOutput)
    assert "modified since you last read" in out.message
    # Disk preserves the external write
    assert p.read_text() == "externally changed"


def test_T15_mtime_jitter_content_unchanged_passes(workdir, fr, fe):
    """B2 content fallback: mtime advanced but bytes identical → allow edit."""
    p = workdir / "a.py"
    p.write_text("hello world", encoding="utf-8")
    _arun(fr.execute(path="a.py"))

    # Push mtime forward by 1 hour without touching content (simulates Windows
    # cloud-sync / antivirus / CI artifact-scanner mtime jitter).
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime + 3600))

    out = _arun(fe.execute(path="a.py", old_string="hello", new_string="HI"))
    assert isinstance(out, FileEditOutput) and out.replaced
    assert p.read_text() == "HI world"


def test_T16_chained_edits_self_record(workdir, fr, fe):
    """B3: post-edit self-record means the second edit doesn't fail staleness."""
    p = workdir / "chain.py"
    p.write_text("alpha bravo charlie", encoding="utf-8")
    _arun(fr.execute(path="chain.py"))
    out1 = _arun(fe.execute(path="chain.py", old_string="alpha", new_string="A"))
    assert isinstance(out1, FileEditOutput) and out1.replaced
    out2 = _arun(fe.execute(path="chain.py", old_string="bravo", new_string="B"))
    assert isinstance(out2, FileEditOutput) and out2.replaced
    out3 = _arun(fe.execute(path="chain.py", old_string="charlie", new_string="C"))
    assert isinstance(out3, FileEditOutput) and out3.replaced
    assert p.read_text() == "A B C"


# ---------------------------------------------------------------------------
# G4: sub-agent isolation
# ---------------------------------------------------------------------------

def test_T17_sub_agent_scope_isolates_read_state(workdir, fr, fe):
    """Main reads foo → enter sub-scope (state empty) → sub edit fails read-gate
    → sub does its own read+edit (modifies state internally) → exit scope →
    main's state restored exactly as before sub.

    Note: sub modified disk, so main's pre-sub entry is now stale relative to disk.
    To test isolation cleanly, we assert main's entry IDENTITY is restored;
    edit-after-restore would correctly fail staleness (which is desired behavior).
    """
    p = workdir / "shared.py"
    p.write_text("alpha beta", encoding="utf-8")
    _arun(fr.execute(path="shared.py"))
    abs_p = config.safe_path("shared.py")
    main_entry_before = file_read_state._state.get(abs_p)
    assert main_entry_before is not None
    main_entry_id = id(main_entry_before)

    with _sub_agent_scope("test"):
        # Sub starts with empty state
        assert file_read_state._state.get(abs_p) is None

        # Sub: edit without read → reject
        out_sub_unread = _arun(fe.execute(
            path="shared.py", old_string="alpha", new_string="A",
        ))
        assert isinstance(out_sub_unread, ToolErrorOutput)
        assert "has not been read yet" in out_sub_unread.message

        # Sub: read its own entry into the (sub) state
        _arun(fr.execute(path="shared.py"))
        assert file_read_state._state.get(abs_p) is not None

    # Back in main: state is restored. The exact same Entry object is in place.
    main_entry_after = file_read_state._state.get(abs_p)
    assert main_entry_after is not None
    assert id(main_entry_after) == main_entry_id


# ---------------------------------------------------------------------------
# G5: token-budget gates
# ---------------------------------------------------------------------------

def test_T18_file_too_large_pre_filter(workdir, fr):
    """Layer 1: stat-time size > 256 KB → reject."""
    (workdir / "big.txt").write_text("x" * (300 * 1024), encoding="utf-8")
    out = _arun(fr.execute(path="big.txt"))
    assert isinstance(out, ToolErrorOutput)
    assert "File too large" in out.message or "limit: 256 KB" in out.message
    assert "offset" in out.message and "limit" in out.message  # next-step hint


def test_T19_file_chars_over_budget(workdir, fr):
    """Layer 2: file < 256 KB stat but content > 100k chars → reject."""
    (workdir / "med.txt").write_text("x" * 105_000, encoding="utf-8")
    out = _arun(fr.execute(path="med.txt"))
    assert isinstance(out, ToolErrorOutput)
    assert "exceed" in out.message
    assert "token budget" in out.message
    assert "offset" in out.message and "limit" in out.message


# ---------------------------------------------------------------------------
# G6: read dedup (FILE_UNCHANGED_STUB)
# ---------------------------------------------------------------------------

def test_T20_dedup_hit_on_repeat_read(workdir, fr):
    (workdir / "a.txt").write_text("hello", encoding="utf-8")
    out1 = _arun(fr.execute(path="a.txt"))
    assert isinstance(out1, FileReadOutput) and not out1.unchanged
    out2 = _arun(fr.execute(path="a.txt"))
    assert isinstance(out2, FileReadOutput) and out2.unchanged
    assert "File unchanged since last read" in out2.to_api_str()


def test_T21_dedup_misses_on_different_limit(workdir, fr):
    (workdir / "a.txt").write_text("\n".join(f"L{i}" for i in range(1, 11)), encoding="utf-8")
    _arun(fr.execute(path="a.txt", limit=5))
    out2 = _arun(fr.execute(path="a.txt", limit=8))
    assert isinstance(out2, FileReadOutput) and not out2.unchanged
    assert out2.returned_lines == 8


def test_T22_dedup_misses_on_mtime_change(workdir, fr):
    p = workdir / "a.txt"
    p.write_text("v1", encoding="utf-8")
    _arun(fr.execute(path="a.txt"))

    import time
    time.sleep(0.05)
    p.write_text("v2", encoding="utf-8")  # changes mtime + content

    out2 = _arun(fr.execute(path="a.txt"))
    assert isinstance(out2, FileReadOutput) and not out2.unchanged
    assert "v2" in out2.content


def test_T23_dedup_skips_after_edit_self_record(workdir, fr, fe):
    """Edit/Write self-records use offset=None; dedup must skip such entries."""
    (workdir / "a.txt").write_text("alpha bravo", encoding="utf-8")
    _arun(fr.execute(path="a.txt"))
    _arun(fe.execute(path="a.txt", old_string="alpha", new_string="A"))
    out = _arun(fr.execute(path="a.txt"))
    assert isinstance(out, FileReadOutput) and not out.unchanged
    assert "A bravo" in out.content


# ---------------------------------------------------------------------------
# G7: full replacement of write_file/edit_file
# ---------------------------------------------------------------------------

def test_T24_old_write_file_module_gone():
    import importlib
    with pytest.raises(ImportError):
        importlib.import_module("mini_cc.tools.write_file")
    with pytest.raises(ImportError):
        importlib.import_module("mini_cc.tools.edit_file")


def test_T25_new_tools_registered_and_old_names_absent():
    assert get_tool("file_read") is not None
    assert get_tool("file_write") is not None
    assert get_tool("file_edit") is not None
    assert get_tool("write_file") is None
    assert get_tool("edit_file") is None


def test_T26_llm_module_loads_with_new_sub_tools():
    import mini_cc.llm as llm_mod
    sub_names = {t.name for t in llm_mod.SUB_TOOLS}
    assert {"file_read", "file_write", "file_edit"} <= sub_names
    assert "write_file" not in sub_names
    assert "edit_file" not in sub_names
    main_names = {t.name for t in llm_mod.MAIN_TOOLS}
    assert {"task", "run_skill"} <= main_names
    assert sub_names <= main_names


# ---------------------------------------------------------------------------
# Sandbox + rich data
# ---------------------------------------------------------------------------

def test_T27_sandbox_violation_on_all_three_tools(workdir, fr, fw, fe):
    out_r = _arun(fr.execute(path="../../../etc/passwd"))
    out_w = _arun(fw.execute(path="../../../tmp/escape.py", content="x"))
    out_e = _arun(fe.execute(path="../escape.py", old_string="x", new_string="y"))
    for out in (out_r, out_w, out_e):
        assert isinstance(out, ToolErrorOutput)
        assert "outside the working directory" in out.message


def test_T28_edit_preserves_rich_data(workdir, fr, fe):
    (workdir / "a.py").write_text("hello world", encoding="utf-8")
    _arun(fr.execute(path="a.py"))
    out = _arun(fe.execute(path="a.py", old_string="hello", new_string="HI"))
    assert isinstance(out, FileEditOutput)
    # Rich data preserved for v2+ UI; not in to_api_str
    assert out.original_content == "hello world"
    assert out.old_string == "hello"
    assert out.new_string == "HI"
    # to_api_str does NOT leak rich data
    api = out.to_api_str()
    assert "hello" not in api  # old_string not exposed
    assert "world" not in api  # original_content not exposed


def test_T28b_write_preserves_rich_data(workdir, fr, fw):
    (workdir / "a.py").write_text("first version", encoding="utf-8")
    _arun(fr.execute(path="a.py"))
    out = _arun(fw.execute(path="a.py", content="second version"))
    assert isinstance(out, FileWriteOutput)
    assert out.operation == "update"
    assert out.original_content == "first version"
    assert out.content == "second version"
    api = out.to_api_str()
    assert "version" not in api  # rich content not exposed
