"""Unit tests for QueryEngine — dispatch, query, run_sidechain."""
from pathlib import Path
from typing import List

import pytest
from langchain_core.messages import AIMessage

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine import query_engine
from mini_cc.engine.messages import (
    AssistantMessage,
    Message,
    StatusMessage,
    TextBlock,
    ToolResultMessage,
    UserMessage,
)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(persistence, "SESSION_ID", "test-qe")
    monkeypatch.setattr(config, "CWD", "/test/cwd")
    return tmp_path


class _ScriptedLLM:
    """Yields a preset sequence of responses, one per astream call."""

    def __init__(self, responses: List[AIMessage]):
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        assert self._responses, "no scripted response left"
        resp = self._responses.pop(0)
        self.calls += 1
        yield resp

    async def ainvoke(self, messages):
        return self._responses.pop(0)


class _CollectingConsumer:
    def __init__(self):
        self.messages: list[Message] = []

    async def on_message(self, msg):
        self.messages.append(msg)


class _RaisingConsumer:
    def __init__(self):
        self.count = 0

    async def on_message(self, msg):
        self.count += 1
        raise RuntimeError("consumer boom")


def _ai(text="hello", tool_calls=None):
    return AIMessage(
        content=text,
        tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        response_metadata={"id": "turn-1"},
    )


def _make_engine(llm):
    return query_engine.QueryEngine(
        llm_base=llm, main_tools=[], sub_tools=[],
        model_name="stub", system_prompt_builder=lambda: "sys",
    )


# ---------------------------------------------------------------------------


class TestQuery:
    @pytest.mark.asyncio
    async def test_text_only_turn(self, isolated_home, fresh_tasks):
        llm = _ScriptedLLM([_ai(text="hi there")])
        eng = _make_engine(llm)
        collected = []
        c = _CollectingConsumer()
        eng.subscribe(c)
        await eng.query("hello")
        collected = c.messages

        # Expected sequence: Status(llm_request), AssistantMessage(Text)
        assert any(isinstance(m, StatusMessage) for m in collected)
        asst = [m for m in collected if isinstance(m, AssistantMessage)]
        assert len(asst) == 1
        assert isinstance(asst[0].content, TextBlock)

    @pytest.mark.asyncio
    async def test_user_message_in_store(self, isolated_home, fresh_tasks):
        llm = _ScriptedLLM([_ai(text="ok")])
        eng = _make_engine(llm)
        await eng.query("hello")
        users = [m for m in eng.store.all() if isinstance(m, UserMessage)]
        assert len(users) == 1
        assert users[0].content == "hello"

    @pytest.mark.asyncio
    async def test_consumer_receives_every_message(self, isolated_home, fresh_tasks):
        llm = _ScriptedLLM([_ai(text="hi")])
        eng = _make_engine(llm)
        c = _CollectingConsumer()
        eng.subscribe(c)
        await eng.query("hello")
        # At minimum: UserMessage + StatusMessage + AssistantMessage
        types = {type(m).__name__ for m in c.messages}
        assert "UserMessage" in types
        assert "StatusMessage" in types
        assert "AssistantMessage" in types


class TestDispatch:
    @pytest.mark.asyncio
    async def test_consumer_exception_isolated(
        self, isolated_home, fresh_tasks, capsys
    ):
        llm = _ScriptedLLM([_ai(text="hi")])
        eng = _make_engine(llm)
        bad = _RaisingConsumer()
        good = _CollectingConsumer()
        eng.subscribe(bad)
        eng.subscribe(good)
        await eng.query("hello")
        assert bad.count > 0  # it was called
        assert good.messages  # and the other consumer still got messages
        err = capsys.readouterr().err
        assert "consumer boom" in err

    @pytest.mark.asyncio
    async def test_store_append_matches_dispatched(
        self, isolated_home, fresh_tasks
    ):
        llm = _ScriptedLLM([_ai(text="hi")])
        eng = _make_engine(llm)
        c = _CollectingConsumer()
        eng.subscribe(c)
        await eng.query("hello")
        # Consumer should see every store message, in order.
        assert [m.id for m in c.messages] == [m.id for m in eng.store.all()]


class TestBoot:
    @pytest.mark.asyncio
    async def test_boot_emits_system_prompt(self, isolated_home, fresh_tasks):
        llm = _ScriptedLLM([])
        eng = _make_engine(llm)
        c = _CollectingConsumer()
        eng.subscribe(c)
        await eng.boot()
        from mini_cc.engine.messages import SystemPromptMessage
        assert any(isinstance(m, SystemPromptMessage) for m in c.messages)
