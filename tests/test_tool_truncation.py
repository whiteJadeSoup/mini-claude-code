"""Tests for oversized tool-result offload (tools/base.truncate_tool_content).

- Below threshold: content passes through verbatim.
- Above threshold: full content lands on disk, API content is a short
  marker + on-disk path + first PREVIEW_CHARS of the original.
- AgentLoop integration: ToolResultMessage.content is truncated,
  ToolResultMessage.output is NOT touched.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine import agent_loop as agent_loop_mod
from mini_cc.engine.agent_loop import AgentLoop
from mini_cc.engine.messages import ToolResultMessage
from mini_cc.tools.base import (
    CommandOutput,
    TOOL_CONTENT_MAX_CHARS,
    TOOL_CONTENT_PREVIEW_CHARS,
    truncate_tool_content,
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-trunc")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


# ---------------------------------------------------------------------------
# truncate_tool_content — unit
# ---------------------------------------------------------------------------


class TestTruncateToolContent:
    def test_below_threshold_passes_through(self, isolated_home):
        content = "x" * (TOOL_CONTENT_MAX_CHARS - 1)
        assert truncate_tool_content(content, "tc-1") == content

    def test_at_threshold_passes_through(self, isolated_home):
        content = "x" * TOOL_CONTENT_MAX_CHARS
        assert truncate_tool_content(content, "tc-1") == content

    def test_above_threshold_writes_to_disk(self, isolated_home):
        content = "y" * (TOOL_CONTENT_MAX_CHARS + 10_000)
        truncate_tool_content(content, "tc-1")
        path = persistence.tool_result_path("tc-1")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == content

    def test_above_threshold_returns_marker_with_path_and_preview(
        self, isolated_home
    ):
        content = "y" * (TOOL_CONTENT_MAX_CHARS + 10_000)
        out = truncate_tool_content(content, "tc-1")
        path = persistence.tool_result_path("tc-1")
        # Result is much smaller than the original.
        assert len(out) < len(content) / 10
        # Marker advertises size + the on-disk path + the preview cap.
        assert f"{len(content):,} chars" in out
        assert str(path) in out
        assert f"First {TOOL_CONTENT_PREVIEW_CHARS:,} chars" in out
        # The preview itself is present (first N chars of original).
        assert content[:TOOL_CONTENT_PREVIEW_CHARS] in out

    def test_separate_tool_call_ids_get_separate_files(self, isolated_home):
        big = "z" * (TOOL_CONTENT_MAX_CHARS + 1)
        truncate_tool_content(big, "tc-A")
        truncate_tool_content(big, "tc-B")
        assert persistence.tool_result_path("tc-A").exists()
        assert persistence.tool_result_path("tc-B").exists()


# ---------------------------------------------------------------------------
# AgentLoop integration: content truncated, output preserved
# ---------------------------------------------------------------------------


class _ScriptedLLM:
    """Returns one AIMessage (with one tool_call) then a plain AIMessage."""

    def __init__(self, first: AIMessage, second: AIMessage):
        self._queue = [first, second]

    def bind_tools(self, tools):  # noqa: ARG002
        return self

    async def astream(self, messages):  # noqa: ARG002
        msg = self._queue.pop(0)
        yield msg


class _HugeBashTool:
    """Stub tool_fn returning a CommandOutput well above the truncate cap."""

    name = "execute_command"

    async def ainvoke(self, args):  # noqa: ARG002
        return CommandOutput(
            stdout="Q" * (TOOL_CONTENT_MAX_CHARS + 5_000),
            returncode=0,
        )


class TestAgentLoopTruncation:
    @pytest.mark.asyncio
    async def test_tool_result_content_truncated_but_output_preserved(
        self, isolated_home, monkeypatch
    ):
        # Ensure the real MiniTool registry answers get_tool("execute_command")
        # with a stub whose to_api_content just delegates to output.to_api_str.
        # The shipped execute_command tool already does this, so we don't need
        # to monkeypatch get_tool — just verify end-to-end behaviour with the
        # real tool module imported below.
        import mini_cc.tools.builtins  # noqa: F401 — triggers registry

        tc = {"id": "tc-integration", "name": "execute_command", "args": {"command": "x"}}
        first = AIMessage(
            content="",
            tool_calls=[tc],
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            response_metadata={"id": "resp-1"},
        )
        second = AIMessage(
            content="done",
            tool_calls=[],
            usage_metadata={"input_tokens": 12, "output_tokens": 1, "total_tokens": 13},
            response_metadata={"id": "resp-2"},
        )
        loop = AgentLoop(
            bound_llm=_ScriptedLLM(first, second),
            tools_by_name={"execute_command": _HugeBashTool()},
            model_name="stub",
        )

        messages_cb_calls = []

        async def get_msgs():
            messages_cb_calls.append(1)
            return []

        emitted = []
        async for m in loop.run(get_messages=get_msgs, parent_id=None, source="agent"):
            emitted.append(m)

        tool_results = [m for m in emitted if isinstance(m, ToolResultMessage)]
        assert len(tool_results) == 1
        tr = tool_results[0]

        # API-facing content is truncated (≫ 10x smaller than the raw stdout).
        assert len(tr.content) < (TOOL_CONTENT_MAX_CHARS + 5_000) / 10
        assert "truncated" in tr.content.lower()
        # UI-facing output is the full CommandOutput with the original stdout.
        assert isinstance(tr.output, CommandOutput)
        assert len(tr.output.stdout) == TOOL_CONTENT_MAX_CHARS + 5_000
        # Full content is on disk, addressable by tool_call_id.
        path = persistence.tool_result_path("tc-integration")
        assert path.exists()
        assert len(path.read_text(encoding="utf-8")) == TOOL_CONTENT_MAX_CHARS + 5_000
