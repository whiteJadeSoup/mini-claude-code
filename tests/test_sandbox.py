"""Unit + integration tests for engine/sandbox.py.

Unit tests exercise each Sandbox implementation against synthetic MiniTool
instances. The integration tests wire a stub through AgentLoop end-to-end
and verify (a) Allow lets the tool run, (b) Deny short-circuits with a
synthetic tool_result the LLM can read, and (c) Allow(updated_args=...)
rewrites the call.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine.agent_loop import AgentLoop
from mini_cc.engine.messages import ToolResultMessage
from mini_cc.engine.sandbox import (
    Allow,
    AllowAll,
    Deny,
    ReadOnlySandbox,
    StaticSandbox,
    allow_all,
)
from mini_cc.tools import base as tools_base
from mini_cc.tools.base import CommandOutput, MiniTool, ToolOutput


# ---------------------------------------------------------------------------
# Synthetic MiniTool stubs (unit tests pass instances directly to check();
# no registry involvement)
# ---------------------------------------------------------------------------


class _StubReadTool(MiniTool):
    name = "stub_read"
    description = "test read tool"
    prompt = ""
    is_read_only = True

    async def _run(self) -> ToolOutput:  # pragma: no cover — unit tests don't invoke
        raise RuntimeError("not invoked in unit tests")


class _StubWriteTool(MiniTool):
    name = "stub_write"
    description = "test write tool"
    prompt = ""
    is_read_only = False

    async def _run(self) -> ToolOutput:  # pragma: no cover
        raise RuntimeError("not invoked in unit tests")


# ---------------------------------------------------------------------------
# AllowAll
# ---------------------------------------------------------------------------


class TestAllowAll:
    @pytest.mark.asyncio
    async def test_allows_read_tool(self):
        assert isinstance(await AllowAll().check(_StubReadTool(), {}), Allow)

    @pytest.mark.asyncio
    async def test_allows_write_tool(self):
        assert isinstance(await AllowAll().check(_StubWriteTool(), {}), Allow)

    def test_allow_all_factory_returns_shared_singleton(self):
        # The default-kwarg path in AgentLoop.run / run_loop relies on this
        # being safe to share — verify it is, in fact, shared.
        assert allow_all() is allow_all()


# ---------------------------------------------------------------------------
# StaticSandbox — declarative name lists
# ---------------------------------------------------------------------------


class TestStaticSandbox:
    @pytest.mark.asyncio
    async def test_no_allow_list_permits_anything_not_denied(self):
        sb = StaticSandbox(name="t", deny=frozenset({"stub_write"}))
        assert isinstance(await sb.check(_StubReadTool(), {}), Allow)
        d = await sb.check(_StubWriteTool(), {})
        assert isinstance(d, Deny)
        assert "stub_write" in d.reason

    @pytest.mark.asyncio
    async def test_allow_list_restricts_to_named_tools(self):
        sb = StaticSandbox(name="t", allow=frozenset({"stub_read"}))
        assert isinstance(await sb.check(_StubReadTool(), {}), Allow)
        d = await sb.check(_StubWriteTool(), {})
        assert isinstance(d, Deny)
        assert "allow list" in d.reason

    @pytest.mark.asyncio
    async def test_empty_allow_list_denies_everything(self):
        # frozenset() != None: explicit "nothing is allowed".
        sb = StaticSandbox(name="t", allow=frozenset())
        assert isinstance(await sb.check(_StubReadTool(), {}), Deny)
        assert isinstance(await sb.check(_StubWriteTool(), {}), Deny)

    @pytest.mark.asyncio
    async def test_deny_wins_over_allow_when_both_match(self):
        sb = StaticSandbox(
            name="t",
            allow=frozenset({"stub_read"}),
            deny=frozenset({"stub_read"}),
        )
        d = await sb.check(_StubReadTool(), {})
        assert isinstance(d, Deny)
        assert "deny list" in d.reason


# ---------------------------------------------------------------------------
# ReadOnlySandbox — uses MiniTool.is_read_only
# ---------------------------------------------------------------------------


class TestReadOnlySandbox:
    @pytest.mark.asyncio
    async def test_allows_read_only_tool(self):
        assert isinstance(
            await ReadOnlySandbox().check(_StubReadTool(), {}), Allow
        )

    @pytest.mark.asyncio
    async def test_denies_non_read_only(self):
        d = await ReadOnlySandbox().check(_StubWriteTool(), {})
        assert isinstance(d, Deny)
        assert "not read-only" in d.reason

    @pytest.mark.asyncio
    async def test_extra_allow_permits_named_non_read_only_tools(self):
        sb = ReadOnlySandbox(extra_allow=frozenset({"stub_write"}))
        assert isinstance(await sb.check(_StubWriteTool(), {}), Allow)


# ---------------------------------------------------------------------------
# Integration: AgentLoop honors sandbox decisions
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-sandbox")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


class _StubRegistered(MiniTool):
    """Stand-in for a real MiniTool in the global registry.

    AgentLoop uses get_tool(name) to fetch this for the sandbox check and
    to call `to_api_content` on the result. The actual tool execution
    happens through the LangChain bridge stub (`_RecordingBridge` below)
    that AgentLoop finds in `tools_by_name` — so this MiniTool's `_run`
    is never invoked; only its `name`, `is_read_only`, and the inherited
    `to_api_content` matter.
    """
    name = "_sandbox_test_write"
    description = "stub write tool used by sandbox integration tests"
    prompt = ""
    is_read_only = False

    async def _run(self, **kwargs: Any) -> ToolOutput:  # pragma: no cover
        raise RuntimeError("execution path goes through the bridge, not _run")


class _RecordingBridge:
    """LangChain-bridge-shaped stub: records what AgentLoop hands it."""
    name = _StubRegistered.name

    def __init__(self) -> None:
        self.invocations: list[dict] = []

    async def ainvoke(self, args: dict) -> ToolOutput:
        self.invocations.append(args)
        # Any ToolOutput will do — agent loop just calls .to_api_str().
        return CommandOutput(stdout=f"ran with {args}", returncode=0)


class _ScriptedLLM:
    """Two-step LLM: one tool_call, then a plain text answer."""

    def __init__(self, *responses: AIMessage) -> None:
        self._queue = list(responses)

    def bind_tools(self, tools):  # noqa: ARG002
        return self

    async def astream(self, messages):  # noqa: ARG002
        yield self._queue.pop(0)


@pytest.fixture
def registered_stub(monkeypatch):
    """Inject _StubRegistered into the global MiniTool registry; auto-remove
    on teardown. monkeypatch.setitem deletes the key if it wasn't present
    before, which is what we want — the stub name shouldn't collide with
    any real tool."""
    tool = _StubRegistered()
    monkeypatch.setitem(tools_base._REGISTRY, tool.name, tool)
    return tool


def _ai_calling(name: str, args: dict, call_id: str = "tc-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": call_id, "name": name, "args": args}],
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        response_metadata={"id": "r1"},
    )


def _ai_text(text: str = "done") -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[],
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        response_metadata={"id": "r2"},
    )


async def _empty_messages() -> list:
    return []


class TestAgentLoopSandbox:
    @pytest.mark.asyncio
    async def test_default_sandbox_is_allow_all(
        self, isolated_home, registered_stub
    ):
        # No sandbox kwarg → AgentLoop should not gate; tool runs normally.
        bridge = _RecordingBridge()
        loop = AgentLoop(
            bound_llm=_ScriptedLLM(
                _ai_calling(registered_stub.name, {"x": 1}),
                _ai_text(),
            ),
            tools_by_name={registered_stub.name: bridge},
            model_name="stub",
        )
        emitted = [m async for m in loop.run(get_messages=_empty_messages)]
        assert bridge.invocations == [{"x": 1}]
        tr = next(m for m in emitted if isinstance(m, ToolResultMessage))
        assert "sandbox" not in tr.content.lower()

    @pytest.mark.asyncio
    async def test_allow_lets_tool_run(self, isolated_home, registered_stub):
        bridge = _RecordingBridge()
        loop = AgentLoop(
            bound_llm=_ScriptedLLM(
                _ai_calling(registered_stub.name, {"x": 1}),
                _ai_text(),
            ),
            tools_by_name={registered_stub.name: bridge},
            model_name="stub",
        )
        # Static sandbox with the stub in the allow list.
        sandbox = StaticSandbox(
            name="permits", allow=frozenset({registered_stub.name})
        )
        emitted = [
            m async for m in loop.run(
                get_messages=_empty_messages, sandbox=sandbox,
            )
        ]
        assert bridge.invocations == [{"x": 1}]
        tr = next(m for m in emitted if isinstance(m, ToolResultMessage))
        assert "denied" not in tr.content.lower()

    @pytest.mark.asyncio
    async def test_deny_short_circuits_without_invoking_tool(
        self, isolated_home, registered_stub
    ):
        bridge = _RecordingBridge()
        loop = AgentLoop(
            bound_llm=_ScriptedLLM(
                _ai_calling(registered_stub.name, {"x": 1}),
                _ai_text(),
            ),
            tools_by_name={registered_stub.name: bridge},
            model_name="stub",
        )
        sandbox = StaticSandbox(
            name="blocks", deny=frozenset({registered_stub.name})
        )
        emitted = [
            m async for m in loop.run(
                get_messages=_empty_messages, sandbox=sandbox,
            )
        ]
        # Bridge was never called — the gate intercepted upstream.
        assert bridge.invocations == []
        # The synthetic tool_result names the sandbox and the tool so the
        # LLM can read why it was rejected and adjust on the next turn.
        tr = next(m for m in emitted if isinstance(m, ToolResultMessage))
        assert "[sandbox:blocks]" in tr.content
        assert registered_stub.name in tr.content

    @pytest.mark.asyncio
    async def test_allow_updated_args_rewrites_input(
        self, isolated_home, registered_stub
    ):
        bridge = _RecordingBridge()
        loop = AgentLoop(
            bound_llm=_ScriptedLLM(
                _ai_calling(registered_stub.name, {"x": 1}),
                _ai_text(),
            ),
            tools_by_name={registered_stub.name: bridge},
            model_name="stub",
        )

        class _Rewriter:
            name = "rewriter"

            async def check(self, tool, args):
                # Canonicalize: bump x to 999 regardless of what the LLM passed.
                return Allow(updated_args={"x": 999})

        emitted = [
            m async for m in loop.run(
                get_messages=_empty_messages, sandbox=_Rewriter(),
            )
        ]
        # The bridge saw the rewritten args, not the LLM's original.
        assert bridge.invocations == [{"x": 999}]
