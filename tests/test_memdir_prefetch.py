"""Layer 2 派生 + surfacing 纯函数单测。"""
import pytest

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine.messages import (
    AssistantMessage, RelevantMemoryMessage, SurfacedMemory, TextBlock,
    ToolResultMessage, ToolUseBlock, UserMessage,
)
from mini_cc.memdir import prefetch as pf_mod
from mini_cc.memdir.prefetch import (
    MAX_MEMORY_LINES, collect_recent_successful_tools, collect_surfaced,
    read_and_truncate, should_prefetch, surface_relevant,
)
from mini_cc.memdir.types import MemoryHeader, MemoryType
from mini_cc.tools.base import ToolOutput


@pytest.fixture(autouse=True)
def _stable_session(monkeypatch):
    monkeypatch.setattr(persistence, "SESSION_ID", "test-session")
    monkeypatch.setattr(config, "CWD", "/test/cwd")


def _sm(filename="a.md", content="x"):
    return SurfacedMemory(filename=filename, path=f"/abs/{filename}",
                          content=content, mtime_ms=1, line_count=1,
                          header=f"Memory: {filename}:")


def _asst_tool(call_id, name):
    return AssistantMessage(turn_id="t", model="m",
                            content=ToolUseBlock(call_id=call_id, name=name, args={}))


def _result(call_id, *, error):
    return ToolResultMessage(content="r", tool_call_id=call_id,
                             output=ToolOutput(is_error=error))


def test_should_prefetch_skips_single_word():
    assert should_prefetch("hi", 0) is False
    assert should_prefetch("recommend a package manager", 0) is True


def test_should_prefetch_skips_when_budget_full():
    assert should_prefetch("a real query here", 60 * 1024) is False


def test_collect_surfaced_paths_and_bytes():
    msgs = [RelevantMemoryMessage(memories=[_sm("a.md", "AAA"), _sm("b.md", "BB")])]
    names, total = collect_surfaced(msgs)
    assert names == frozenset({"a.md", "b.md"})
    assert total == len("AAA") + len("BB")


def test_collect_surfaced_empty():
    assert collect_surfaced([UserMessage(content="hi", source="user")]) == (frozenset(), 0)


def test_recent_tools_success_minus_failed():
    msgs = [
        UserMessage(id="u1", content="prev", source="user"),
        _asst_tool("c1", "file_edit"), _result("c1", error=False),
        _asst_tool("c2", "execute_command"), _result("c2", error=True),
        AssistantMessage(turn_id="t", model="m", content=TextBlock(text="done")),
        UserMessage(id="u2", content="cur", source="user"),
    ]
    assert collect_recent_successful_tools(msgs, current_user_id="u2") == ("file_edit",)


def test_recent_tools_excludes_tool_that_ever_errored():
    msgs = [UserMessage(id="u1", content="p", source="user"),
            _asst_tool("c1", "file_edit"), _result("c1", error=False),
            _asst_tool("c3", "file_edit"), _result("c3", error=True),
            UserMessage(id="u2", content="c", source="user")]
    assert collect_recent_successful_tools(msgs, current_user_id="u2") == ()


def test_recent_tools_stops_at_previous_user_turn():
    msgs = [_asst_tool("cX", "grep"), _result("cX", error=False),
            UserMessage(id="u0", content="old", source="user"),
            UserMessage(id="u1", content="p", source="user"),
            _asst_tool("c1", "file_edit"), _result("c1", error=False),
            UserMessage(id="u2", content="c", source="user")]
    assert collect_recent_successful_tools(msgs, current_user_id="u2") == ("file_edit",)


def test_read_and_truncate_under_cap(tmp_path):
    f = tmp_path / "m.md"
    f.write_text("line1\nline2\n", encoding="utf-8")
    h = MemoryHeader(filename="m.md", file_path=f, mtime_ms=1.0,
                     description="d", type=MemoryType.USER)
    sm = read_and_truncate(h)
    assert sm is not None and "line1" in sm.content and "truncated" not in sm.content
    assert sm.path == str(f) and sm.filename == "m.md"
    assert sm.header.endswith("Memory: m.md:") and "days old" in sm.header


def test_read_and_truncate_over_line_cap(tmp_path):
    f = tmp_path / "big.md"
    f.write_text("\n".join(f"l{i}" for i in range(MAX_MEMORY_LINES + 50)), encoding="utf-8")
    h = MemoryHeader(filename="big.md", file_path=f, mtime_ms=1.0, description=None, type=None)
    sm = read_and_truncate(h)
    assert sm is not None and "truncated" in sm.content
    assert sm.line_count == MAX_MEMORY_LINES


def test_read_and_truncate_missing_file_returns_none(tmp_path):
    h = MemoryHeader(filename="gone.md", file_path=tmp_path / "gone.md",
                     mtime_ms=1.0, description=None, type=None)
    assert read_and_truncate(h) is None


async def test_surface_relevant_reads_selected(tmp_path, monkeypatch):
    f = tmp_path / "user_role.md"
    f.write_text("likes uv", encoding="utf-8")
    h = MemoryHeader(filename="user_role.md", file_path=f, mtime_ms=1.0,
                     description="d", type=MemoryType.USER)

    async def _fake_find(query, memdir, *, recent_tools=(), already_surfaced=frozenset()):
        return [h]
    monkeypatch.setattr(pf_mod, "find_relevant_memories", _fake_find)

    out = await surface_relevant("q", tmp_path)
    assert len(out) == 1
    assert out[0].filename == "user_role.md" and "likes uv" in out[0].content
    assert out[0].path == str(f)
    assert out[0].header.endswith("user_role.md:")
