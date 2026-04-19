"""Unit tests for engine.predicates — named predicates + combinators."""
from mini_cc.engine.messages import (
    AssistantMessage,
    CompactBoundaryMessage,
    StatusMessage,
    SystemPromptMessage,
    TextBlock,
    ToolResultMessage,
    UserMessage,
)
from mini_cc.engine.predicates import (
    accept_all,
    all_of,
    any_of,
    is_assistant,
    is_compact_boundary,
    is_main_branch,
    is_persisted_layer,
    is_status,
    is_synthetic_user,
    is_tool_result,
    is_user,
    negate,
)


def _user(**kw):
    return UserMessage(content="hi", **kw)


def _asst():
    return AssistantMessage(turn_id="t1", model="m", content=TextBlock(text="x"))


class TestNamedPredicates:
    def test_accept_all(self):
        assert accept_all(_user()) is True
        assert accept_all(_asst()) is True

    def test_is_main_branch(self):
        assert is_main_branch(_user()) is True
        assert is_main_branch(_user(parent_id="abc")) is False

    def test_is_assistant(self):
        assert is_assistant(_asst()) is True
        assert is_assistant(_user()) is False

    def test_is_user(self):
        assert is_user(_user()) is True
        assert is_user(_asst()) is False

    def test_is_synthetic_user(self):
        assert is_synthetic_user(_user(is_synthetic=True)) is True
        assert is_synthetic_user(_user(is_synthetic=False)) is False
        assert is_synthetic_user(_asst()) is False

    def test_is_status(self):
        assert is_status(StatusMessage(event="turn_start")) is True
        assert is_status(_user()) is False

    def test_is_tool_result(self):
        assert is_tool_result(ToolResultMessage(content="out", tool_call_id="c1")) is True
        assert is_tool_result(_asst()) is False

    def test_is_compact_boundary(self):
        assert is_compact_boundary(CompactBoundaryMessage(pre_count=1, auto=False)) is True
        assert is_compact_boundary(_user()) is False

    def test_is_persisted_layer(self):
        assert is_persisted_layer(SystemPromptMessage(content="s")) is True
        assert is_persisted_layer(_user()) is True
        assert is_persisted_layer(_asst()) is True
        assert is_persisted_layer(ToolResultMessage(content="o", tool_call_id="c1")) is True
        assert is_persisted_layer(CompactBoundaryMessage(pre_count=1, auto=False)) is True
        assert is_persisted_layer(StatusMessage(event="x")) is True


class TestCombinators:
    def test_all_of_empty_is_true(self):
        assert all_of()(_user()) is True

    def test_all_of_both_true(self):
        p = all_of(is_user, is_main_branch)
        assert p(_user()) is True

    def test_all_of_one_false(self):
        p = all_of(is_user, is_assistant)
        assert p(_user()) is False

    def test_any_of_empty_is_false(self):
        assert any_of()(_user()) is False

    def test_any_of_one_true(self):
        p = any_of(is_user, is_assistant)
        assert p(_user()) is True
        assert p(_asst()) is True

    def test_any_of_all_false(self):
        p = any_of(is_assistant, is_tool_result)
        assert p(_user()) is False

    def test_negate(self):
        p = negate(is_user)
        assert p(_user()) is False
        assert p(_asst()) is True

    def test_nested_composition(self):
        p = all_of(is_main_branch, negate(is_synthetic_user))
        assert p(_user()) is True
        assert p(_user(is_synthetic=True)) is False
        assert p(_user(parent_id="x")) is False
