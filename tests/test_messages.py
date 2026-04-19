"""Unit tests for messages.py — types, factory, converters, serialization."""
import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine import messages as M
from mini_cc.engine.messages import (
    AssistantMessage, CompactBoundaryMessage, ContentBlock, StatusMessage,
    SystemPromptMessage, TextBlock, ToolResultMessage, ToolUseBlock, UserMessage,
    LAYER_1_TYPES, LAYER_2_TYPES,
    assistant_messages_from_ai, to_langchain_single,
)


@pytest.fixture(autouse=True)
def stable_session(monkeypatch):
    monkeypatch.setattr(persistence, "SESSION_ID", "test-session")
    monkeypatch.setattr(config, "CWD", "/test/cwd")


# ---------------------------------------------------------------------------
# assistant_messages_from_ai — factory
# ---------------------------------------------------------------------------

class TestAssistantMessagesFromAi:
    def _ai(self, content="", tool_calls=None, response_id=None):
        metadata = {"id": response_id} if response_id else {}
        return AIMessage(
            content=content,
            tool_calls=tool_calls or [],
            response_metadata=metadata,
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )

    def test_text_only_produces_one_message(self):
        ai = self._ai(content="hello", response_id="r1")
        msgs = assistant_messages_from_ai(ai, "model-x", None, "agent")
        assert len(msgs) == 1
        assert isinstance(msgs[0].content, TextBlock)
        assert msgs[0].content.text == "hello"

    def test_turn_id_from_response_metadata(self):
        ai = self._ai(content="hi", response_id="msg_abc")
        msgs = assistant_messages_from_ai(ai, "m", None, "s")
        assert msgs[0].turn_id == "msg_abc"

    def test_turn_id_fallback_when_missing(self):
        ai = self._ai(content="hi")
        msgs = assistant_messages_from_ai(ai, "m", None, "s")
        assert msgs[0].turn_id.startswith("local-")

    def test_i4_empty_text_skipped(self):
        """I4: no TextBlock emitted when ai.content is empty."""
        tc = [{"id": "c1", "name": "ls", "args": {}}]
        ai = self._ai(content="", tool_calls=tc, response_id="r1")
        msgs = assistant_messages_from_ai(ai, "m", None, "s")
        assert all(isinstance(m.content, ToolUseBlock) for m in msgs)

    def test_tool_only_produces_one_tool_use_block(self):
        tc = [{"id": "c1", "name": "run", "args": {"x": 1}}]
        ai = self._ai(content="", tool_calls=tc, response_id="r1")
        msgs = assistant_messages_from_ai(ai, "m", None, "s")
        assert len(msgs) == 1
        assert isinstance(msgs[0].content, ToolUseBlock)
        assert msgs[0].content.call_id == "c1"
        assert msgs[0].content.name == "run"
        assert msgs[0].content.args == {"x": 1}

    def test_mixed_text_and_tools_produces_correct_count(self):
        tc = [{"id": "c1", "name": "ls", "args": {}},
              {"id": "c2", "name": "cat", "args": {"f": "x"}}]
        ai = self._ai(content="I'll run ls and cat", tool_calls=tc, response_id="r1")
        msgs = assistant_messages_from_ai(ai, "m", None, "s")
        assert len(msgs) == 3
        assert isinstance(msgs[0].content, TextBlock)
        assert isinstance(msgs[1].content, ToolUseBlock)
        assert isinstance(msgs[2].content, ToolUseBlock)

    def test_all_messages_share_turn_id(self):
        tc = [{"id": "c1", "name": "ls", "args": {}}]
        ai = self._ai(content="doing it", tool_calls=tc, response_id="r99")
        msgs = assistant_messages_from_ai(ai, "m", None, "s")
        assert all(m.turn_id == "r99" for m in msgs)

    def test_parent_id_propagated(self):
        ai = self._ai(content="hi", response_id="r1")
        msgs = assistant_messages_from_ai(ai, "m", "parent-xyz", "s")
        assert msgs[0].parent_id == "parent-xyz"

    def test_model_and_source_set(self):
        ai = self._ai(content="hi", response_id="r1")
        msgs = assistant_messages_from_ai(ai, "deepseek-reasoner", None, "agent")
        assert msgs[0].model == "deepseek-reasoner"
        assert msgs[0].source == "agent"


# ---------------------------------------------------------------------------
# to_langchain_single — conversions
# ---------------------------------------------------------------------------

class TestToLangchainSingle:
    def test_system_prompt_message(self):
        msg = SystemPromptMessage(content="you are helpful", source="boot")
        lc = to_langchain_single(msg)
        assert isinstance(lc, SystemMessage)
        assert lc.content == "you are helpful"

    def test_user_message(self):
        msg = UserMessage(content="hello")
        lc = to_langchain_single(msg)
        assert isinstance(lc, HumanMessage)
        assert lc.content == "hello"

    def test_tool_result_message(self):
        msg = ToolResultMessage(content="output", tool_call_id="c1")
        lc = to_langchain_single(msg)
        assert isinstance(lc, ToolMessage)
        assert lc.content == "output"
        assert lc.tool_call_id == "c1"

    def test_assistant_message_returns_none(self):
        msg = AssistantMessage(
            turn_id="t1", model="m",
            content=TextBlock(text="hi"),
        )
        assert to_langchain_single(msg) is None

    def test_compact_boundary_returns_none(self):
        msg = CompactBoundaryMessage(pre_count=5, auto=False)
        assert to_langchain_single(msg) is None

    def test_status_message_returns_none(self):
        msg = StatusMessage(event="session_started")
        assert to_langchain_single(msg) is None


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------

class TestLayerClassification:
    def test_layer_1_types(self):
        assert SystemPromptMessage in LAYER_1_TYPES
        assert UserMessage in LAYER_1_TYPES
        assert AssistantMessage in LAYER_1_TYPES
        assert ToolResultMessage in LAYER_1_TYPES

    def test_layer_2_types(self):
        assert CompactBoundaryMessage in LAYER_2_TYPES
        assert StatusMessage in LAYER_2_TYPES

    def test_no_overlap(self):
        assert set(LAYER_1_TYPES).isdisjoint(set(LAYER_2_TYPES))


# ---------------------------------------------------------------------------
# Pydantic serialization round-trips
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    def _roundtrip(self, msg):
        dumped = msg.model_dump(mode="json")
        raw = json.dumps(dumped)
        loaded = json.loads(raw)
        return loaded

    def test_system_prompt_message(self):
        msg = SystemPromptMessage(content="sys", source="boot")
        d = self._roundtrip(msg)
        assert d["type"] == "system_prompt"
        assert d["content"] == "sys"
        assert d["session_id"] == "test-session"

    def test_user_message_synthetic(self):
        msg = UserMessage(content="summary", is_synthetic=True, source="compact")
        d = self._roundtrip(msg)
        assert d["type"] == "user"
        assert d["is_synthetic"] is True

    def test_assistant_message_text_block(self):
        msg = AssistantMessage(
            turn_id="t1", model="deepseek-reasoner",
            content=TextBlock(text="hello"),
        )
        d = self._roundtrip(msg)
        assert d["type"] == "assistant"
        assert d["content"]["type"] == "text"
        assert d["content"]["text"] == "hello"

    def test_assistant_message_tool_use_block(self):
        msg = AssistantMessage(
            turn_id="t1", model="m",
            content=ToolUseBlock(call_id="c1", name="ls", args={"path": "."}),
        )
        d = self._roundtrip(msg)
        assert d["content"]["type"] == "tool_use"
        assert d["content"]["call_id"] == "c1"
        assert d["content"]["name"] == "ls"

    def test_tool_result_message(self):
        msg = ToolResultMessage(content="output", tool_call_id="c1")
        d = self._roundtrip(msg)
        assert d["type"] == "tool_result"
        assert d["tool_call_id"] == "c1"

    def test_compact_boundary_message(self):
        msg = CompactBoundaryMessage(pre_count=42, auto=True, source="compact")
        d = self._roundtrip(msg)
        assert d["type"] == "compact_boundary"
        assert d["pre_count"] == 42
        assert d["auto"] is True

    def test_status_message(self):
        msg = StatusMessage(event="session_started", data={"version": "1"})
        d = self._roundtrip(msg)
        assert d["type"] == "status"
        assert d["event"] == "session_started"
        assert d["data"]["version"] == "1"

    def test_datetime_serializes_to_string(self):
        msg = UserMessage(content="hi")
        d = self._roundtrip(msg)
        assert isinstance(d["created_at"], str)

    def test_parent_id_null_when_none(self):
        msg = UserMessage(content="hi")
        d = self._roundtrip(msg)
        assert d["parent_id"] is None

    def test_parent_id_preserved(self):
        msg = UserMessage(content="hi", parent_id="abc-123")
        d = self._roundtrip(msg)
        assert d["parent_id"] == "abc-123"
