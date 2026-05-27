"""方案 C: api_view merges consecutive HumanMessages at the API boundary
so DeepSeek-v4-pro never sees two user turns in a row. ToolMessages
(role=tool) must NOT be swept into the merge."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mini_cc.engine.store import _merge_consecutive_human


def test_merge_two_consecutive_humans():
    out = _merge_consecutive_human([HumanMessage(content="a"), HumanMessage(content="b")])
    assert len(out) == 1
    assert out[0].content == "a\n\nb"


def test_merge_preserves_order_and_breaks_on_ai():
    out = _merge_consecutive_human([
        HumanMessage(content="ctx"),
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
    ])
    assert [type(m).__name__ for m in out] == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert out[0].content == "ctx\n\nq1"
    assert out[2].content == "q2"


def test_tool_message_is_not_merged():
    # ToolMessage sits between an AI tool call and the next human turn; it must
    # stay its own message (role=tool), never merged into a human.
    msgs = [
        AIMessage(content="calling"),
        ToolMessage(content="result", tool_call_id="t1"),
        HumanMessage(content="next"),
    ]
    out = _merge_consecutive_human(msgs)
    assert [type(m).__name__ for m in out] == ["AIMessage", "ToolMessage", "HumanMessage"]


def test_single_and_empty():
    assert _merge_consecutive_human([]) == []
    one = [HumanMessage(content="solo")]
    assert _merge_consecutive_human(one)[0].content == "solo"
