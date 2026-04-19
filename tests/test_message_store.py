"""Unit tests for message_store.py — api_view merge, adjacency, subscribers."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from mini_cc import config
from mini_cc.consumers import persistence
from mini_cc.engine.messages import (
    AssistantMessage, CompactBoundaryMessage, SystemPromptMessage,
    TextBlock, ToolResultMessage, ToolUseBlock, UserMessage,
)
from mini_cc.engine.store import MessageStore


@pytest.fixture(autouse=True)
def stable_session(monkeypatch):
    monkeypatch.setattr(persistence, "SESSION_ID", "test-session")
    monkeypatch.setattr(config, "CWD", "/test/cwd")


@pytest.fixture
def store():
    return MessageStore()


def _asst_text(turn_id, text, parent_id=None):
    return AssistantMessage(
        turn_id=turn_id, model="m",
        content=TextBlock(text=text),
        parent_id=parent_id,
    )


def _asst_tool(turn_id, call_id, name="ls", parent_id=None):
    return AssistantMessage(
        turn_id=turn_id, model="m",
        content=ToolUseBlock(call_id=call_id, name=name, args={}),
        parent_id=parent_id,
    )


# ---------------------------------------------------------------------------
# api_view — merge logic
# ---------------------------------------------------------------------------

class TestApiViewMerge:
    def test_empty_store(self, store):
        assert store.api_view() == []

    def test_system_prompt_converts(self, store):
        store.append(SystemPromptMessage(content="sys", source="boot"))
        out = store.api_view()
        assert len(out) == 1
        assert isinstance(out[0], SystemMessage)
        assert out[0].content == "sys"

    def test_user_message_converts(self, store):
        store.append(UserMessage(content="hello"))
        out = store.api_view()
        assert isinstance(out[0], HumanMessage)
        assert out[0].content == "hello"

    def test_tool_result_converts(self, store):
        store.append(ToolResultMessage(content="output", tool_call_id="c1"))
        out = store.api_view()
        assert isinstance(out[0], ToolMessage)
        assert out[0].content == "output"
        assert out[0].tool_call_id == "c1"

    def test_single_text_assistant_message(self, store):
        store.append(_asst_text("t1", "hi"))
        out = store.api_view()
        assert len(out) == 1
        ai = out[0]
        assert isinstance(ai, AIMessage)
        assert ai.content == "hi"
        assert ai.tool_calls == []

    def test_single_tool_use_assistant_message(self, store):
        store.append(_asst_tool("t1", "c1", "ls"))
        out = store.api_view()
        assert len(out) == 1
        ai = out[0]
        assert isinstance(ai, AIMessage)
        assert ai.content == ""
        assert len(ai.tool_calls) == 1
        assert ai.tool_calls[0]["name"] == "ls"
        assert ai.tool_calls[0]["id"] == "c1"

    def test_mixed_text_and_tool_same_turn(self, store):
        """Text + tool blocks with same turn_id → merged into one AIMessage."""
        store.append(_asst_text("t1", "running ls"))
        store.append(_asst_tool("t1", "c1", "ls"))
        out = store.api_view()
        assert len(out) == 1
        ai = out[0]
        assert ai.content == "running ls"
        assert len(ai.tool_calls) == 1

    def test_two_tool_blocks_same_turn(self, store):
        store.append(_asst_tool("t1", "c1", "ls"))
        store.append(_asst_tool("t1", "c2", "cat"))
        out = store.api_view()
        assert len(out) == 1
        ai = out[0]
        assert len(ai.tool_calls) == 2
        names = {tc["name"] for tc in ai.tool_calls}
        assert names == {"ls", "cat"}

    def test_two_turns_produce_two_ai_messages(self, store):
        store.append(_asst_text("t1", "first"))
        store.append(ToolResultMessage(content="result", tool_call_id="c1"))
        store.append(_asst_text("t2", "second"))
        out = store.api_view()
        ai_msgs = [m for m in out if isinstance(m, AIMessage)]
        assert len(ai_msgs) == 2
        assert ai_msgs[0].content == "first"
        assert ai_msgs[1].content == "second"

    def test_full_turn_sequence(self, store):
        """sys → user → assistant(text+tool) → tool_result → assistant(text)"""
        store.append(SystemPromptMessage(content="sys", source="boot"))
        store.append(UserMessage(content="hello"))
        store.append(_asst_text("t1", "I'll run ls"))
        store.append(_asst_tool("t1", "c1", "ls"))
        store.append(ToolResultMessage(content="file.py", tool_call_id="c1"))
        store.append(_asst_text("t2", "done"))

        out = store.api_view()
        assert len(out) == 5
        assert isinstance(out[0], SystemMessage)
        assert isinstance(out[1], HumanMessage)
        assert isinstance(out[2], AIMessage)
        assert out[2].content == "I'll run ls"
        assert len(out[2].tool_calls) == 1
        assert isinstance(out[3], ToolMessage)
        assert isinstance(out[4], AIMessage)
        assert out[4].content == "done"

    def test_layer_2_messages_excluded(self, store):
        store.append(CompactBoundaryMessage(pre_count=5, auto=False))
        out = store.api_view()
        assert out == []

    def test_sidechain_filtered_from_main(self, store):
        """Messages with parent_id are excluded from main branch api_view."""
        store.append(UserMessage(content="main user"))
        store.append(UserMessage(content="sub user", parent_id="parent-abc"))
        out = store.api_view(parent_id=None)
        assert len(out) == 1
        assert out[0].content == "main user"

    def test_sidechain_api_view(self, store):
        """api_view(parent_id=X) returns only sidechain messages."""
        store.append(UserMessage(content="main user"))
        store.append(SystemPromptMessage(content="sub sys", parent_id="parent-abc"))
        store.append(UserMessage(content="sub user", parent_id="parent-abc"))
        out = store.api_view(parent_id="parent-abc")
        assert len(out) == 2
        assert isinstance(out[0], SystemMessage)
        assert isinstance(out[1], HumanMessage)


# ---------------------------------------------------------------------------
# append — adjacency invariant I1
# ---------------------------------------------------------------------------

class TestAdjacency:
    def test_same_turn_id_contiguous_ok(self, store):
        store.append(_asst_text("t1", "a"))
        store.append(_asst_tool("t1", "c1"))  # same turn_id, contiguous
        # no exception

    def test_different_turn_ids_ok(self, store):
        store.append(_asst_text("t1", "a"))
        store.append(ToolResultMessage(content="r", tool_call_id="c1"))
        store.append(_asst_text("t2", "b"))  # new turn_id
        # no exception

    def test_non_contiguous_same_turn_id_raises(self, store):
        store.append(_asst_text("t1", "a"))
        store.append(ToolResultMessage(content="r", tool_call_id="c1"))
        store.append(_asst_text("t2", "b"))
        with pytest.raises(AssertionError, match="Adjacency violation"):
            store.append(_asst_text("t1", "late"))  # t1 was before t2, non-contiguous

    def test_adjacency_independent_per_branch(self, store):
        """Adjacency is per-branch; different parent_ids don't interfere."""
        store.append(_asst_text("t1", "main", parent_id=None))
        store.append(_asst_text("t1", "sub", parent_id="parent-abc"))
        # t1 in main, t1 in sub — different branches, no violation


# Subscriber dispatch tests removed — notification moved to QueryEngine._dispatch.
# See test_query_engine.py for consumer-isolation tests.


# ---------------------------------------------------------------------------
# clear_layer_1
# ---------------------------------------------------------------------------

class TestClearLayer1:
    def test_removes_layer_1(self, store):
        store.append(SystemPromptMessage(content="sys", source="boot"))
        store.append(UserMessage(content="hi"))
        store.append(_asst_text("t1", "hello"))
        store.append(ToolResultMessage(content="r", tool_call_id="c1"))
        removed = store.clear_layer_1()
        assert removed == 4
        assert store.all() == []

    def test_preserves_layer_2(self, store):
        store.append(CompactBoundaryMessage(pre_count=5, auto=False))
        store.append(UserMessage(content="hi"))
        store.clear_layer_1()
        remaining = store.all()
        assert len(remaining) == 1
        assert isinstance(remaining[0], CompactBoundaryMessage)

    def test_returns_count_removed(self, store):
        store.append(UserMessage(content="a"))
        store.append(UserMessage(content="b"))
        store.append(CompactBoundaryMessage(pre_count=2, auto=False))
        n = store.clear_layer_1()
        assert n == 2

    def test_empty_store_returns_zero(self, store):
        assert store.clear_layer_1() == 0


# ---------------------------------------------------------------------------
# replace_system_prompt
# ---------------------------------------------------------------------------

class TestReplaceSystemPrompt:
    def test_replaces_existing(self, store):
        store.append(SystemPromptMessage(content="old", source="boot"))
        store.replace_system_prompt(SystemPromptMessage(content="new", source="boot"))
        msgs = store.all()
        assert len(msgs) == 1
        assert msgs[0].content == "new"

    def test_appends_if_not_found(self, store):
        new_sys = SystemPromptMessage(content="sys", source="boot")
        store.replace_system_prompt(new_sys)
        assert len(store.all()) == 1

    def test_does_not_replace_sidechain_system_prompt(self, store):
        """Main-branch replacement leaves sidechain SystemPromptMessages alone."""
        store.append(SystemPromptMessage(content="main", source="boot", parent_id=None))
        store.append(SystemPromptMessage(content="sub", source="task", parent_id="p1"))
        store.replace_system_prompt(SystemPromptMessage(content="main-new", source="boot"))
        msgs = store.all()
        assert msgs[0].content == "main-new"
        assert msgs[1].content == "sub"
