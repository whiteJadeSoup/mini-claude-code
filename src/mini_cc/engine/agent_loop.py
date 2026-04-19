"""AgentLoop — pure async message producer.

Given a bound LLM, a tool registry, and a way to fetch the current
conversation, AgentLoop yields messages (status, assistant, tool_result)
until the LLM stops calling tools. It has no awareness of:
  - the MessageStore (doesn't read or append)
  - consumers (doesn't dispatch)
  - compaction, context budget, or any store-level concerns

The caller is expected to dispatch each yielded message before the
generator is resumed — that's how tool_results from this iteration
become visible to the next api_view() call (the generator pauses at
`yield`; the caller's `await dispatch(msg)` runs; when `__anext__` is
awaited, we loop and call `get_messages()` again, which now sees the
freshly-appended tool_result).

Why split from QueryEngine: the engine's job is "own the store and fan
out messages to consumers". Knowing how to stream an LLM, aggregate
chunks, and execute tools is a different concern — it belongs here so
the engine's code reads as orchestration, not mechanics.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from mini_cc.engine.messages import (
    Message,
    StatusMessage,
    ToolResultMessage,
    ToolUseBlock,
    assistant_messages_from_ai,
)
from mini_cc.engine.store import _triggering_asst_id
from mini_cc.state import usage
from mini_cc.tools.base import ToolOutput, ToolErrorOutput, get_tool


class AgentLoop:
    def __init__(
        self,
        bound_llm: BaseChatModel,
        tools_by_name: dict[str, BaseTool],
        model_name: str,
    ) -> None:
        self._llm = bound_llm
        self._tools = tools_by_name
        self._model_name = model_name

    async def run(
        self,
        get_messages: Callable[[], Awaitable[list[BaseMessage]]],
        parent_id: str | None = None,
        source: str = "agent",
    ) -> AsyncIterator[Message]:
        """Run one multi-step turn; yield every produced message.

        Args:
            get_messages: async callable that returns the fresh LangChain
                message list for the next LLM call. Called once per
                iteration. The caller owns any pre-call bookkeeping
                (context trimming, auto-compact) — we just ask for the
                current view and send it.
            parent_id: sidechain parent id, or None for main branch.
            source: usage-tracking label; updated to "tool: X, Y" after
                each tool-dispatch step so subsequent LLM calls are
                attributed to their trigger.
        """
        while True:
            messages = await get_messages()

            yield StatusMessage(
                event="llm_request",
                data={"parent_id": parent_id, "source": source},
                parent_id=parent_id,
                source=source,
            )

            full_response = None
            async for chunk in self._llm.astream(messages):
                full_response = (
                    chunk if full_response is None else full_response + chunk
                )
                if chunk.content:
                    usage._tracker.count_stream_chunk()

            if full_response is None:
                raise RuntimeError("LLM returned empty response")

            usage._tracker.record(
                source,
                full_response.usage_metadata,
                getattr(full_response, "response_metadata", None),
            )

            asst_msgs = assistant_messages_from_ai(
                full_response,
                model=self._model_name,
                parent_id=parent_id,
                source=source,
            )
            for m in asst_msgs:
                yield m

            if not full_response.tool_calls:
                return

            for tc in full_response.tool_calls:
                name = tc["name"]
                tool_fn = self._tools.get(name)
                # Locate the AssistantMessage that carried this tool_use —
                # its id becomes parent_id for any sidechain the tool spawns.
                trigger = next(
                    (
                        m
                        for m in asst_msgs
                        if isinstance(m.content, ToolUseBlock)
                        and m.content.call_id == tc["id"]
                    ),
                    None,
                )
                if tool_fn is None:
                    content = (
                        f"Error: unknown tool '{name}'. "
                        f"Available: {', '.join(self._tools)}"
                    )
                    output = None
                else:
                    token = _triggering_asst_id.set(
                        trigger.id if trigger else None
                    )
                    try:
                        # MiniTools: execute() never throws; validation errors
                        # return a str via handle_validation_error callback.
                        raw = await tool_fn.ainvoke(tc["args"])
                    finally:
                        _triggering_asst_id.reset(token)

                    mini = get_tool(name)
                    if mini is not None and isinstance(raw, ToolOutput):
                        content = mini.to_api_content(raw)
                        output = raw
                    else:
                        content = str(raw)
                        # validation error returns str; wrap so TUI shows red ●
                        output = ToolErrorOutput(message=content) if mini is not None else None

                yield ToolResultMessage(
                    content=content,
                    output=output,
                    tool_call_id=tc["id"],
                    parent_id=parent_id,
                    source=source,
                )

            names = [tc["name"] for tc in full_response.tool_calls]
            source = "tool: " + ", ".join(names)
