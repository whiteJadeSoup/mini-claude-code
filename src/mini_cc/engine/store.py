"""MessageStore — the single source of truth for all session events.

The engine owns a MessageStore instance and is the sole writer. Consumer
fan-out has moved to QueryEngine._dispatch — the store no longer notifies
subscribers, keeping it a pure data structure.
"""
from __future__ import annotations

from contextvars import ContextVar

from langchain_core.messages import AIMessage, BaseMessage

from mini_cc.engine.messages import (
    AssistantMessage,
    LAYER_1_TYPES,
    Message,
    SystemPromptMessage,
    TextBlock,
    ToolUseBlock,
    to_langchain_single,
)

_triggering_asst_id: ContextVar[str | None] = ContextVar(
    "triggering_asst_id", default=None
)
"""Set by _run_loop before each tool invocation so task()/run_skill() can
read it to determine their sidechain parent_id. Cross-module contract with
tools.py — must stay here (not on the engine) because tools import it
without holding an engine reference."""


class MessageStore:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    # --- Mutation ---

    def append(self, msg: Message) -> None:
        """Validate adjacency (I1) and append.

        Note: consumer notification is done by QueryEngine._dispatch, NOT here.
        The store is a pure data structure — one write path, one dispatch path.
        """
        if isinstance(msg, AssistantMessage) and self._messages:
            same_branch = [
                m for m in self._messages
                if isinstance(m, AssistantMessage) and m.parent_id == msg.parent_id
            ]
            if same_branch:
                last_turn_id = same_branch[-1].turn_id
                if last_turn_id != msg.turn_id:
                    seen = {m.turn_id for m in same_branch}
                    if msg.turn_id in seen:
                        raise AssertionError(
                            f"Adjacency violation: turn_id '{msg.turn_id}' appeared non-contiguously"
                        )
        self._messages.append(msg)

    def clear_layer_1(self) -> int:
        """Remove all Layer-1 messages. Layer 2 history is preserved.

        Returns count removed. Used by compact().
        """
        before = len(self._messages)
        self._messages = [m for m in self._messages if not isinstance(m, LAYER_1_TYPES)]
        return before - len(self._messages)

    def replace_system_prompt(self, msg: SystemPromptMessage) -> None:
        """Replace the main-branch system prompt in-place (for skill rescans)."""
        for i, m in enumerate(self._messages):
            if isinstance(m, SystemPromptMessage) and m.parent_id is None:
                self._messages[i] = msg
                return
        self.append(msg)

    def reset(self) -> None:
        """Clear all messages. For test isolation only."""
        self._messages.clear()

    # --- Views ---

    def all(self) -> list[Message]:
        """Full event stream, immutable copy."""
        return list(self._messages)

    def api_view(self, parent_id: str | None = None) -> list[BaseMessage]:
        """Filter by parent_id + Layer 1, merge AssistantMessages by turn_id.

        Returns LangChain BaseMessages ready for LLM.stream().
        """
        filtered = [
            m for m in self._messages
            if m.parent_id == parent_id and isinstance(m, LAYER_1_TYPES)
        ]

        out: list[BaseMessage] = []
        pending: list[AssistantMessage] = []

        def flush() -> None:
            if not pending:
                return
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            for m in pending:
                if isinstance(m.content, TextBlock):
                    text_parts.append(m.content.text)
                elif isinstance(m.content, ToolUseBlock):
                    b = m.content
                    tool_calls.append({"id": b.call_id, "name": b.name, "args": b.args})
            out.append(AIMessage(
                content="\n\n".join(text_parts),
                tool_calls=tool_calls,
            ))
            pending.clear()

        for m in filtered:
            if isinstance(m, AssistantMessage):
                if pending and pending[-1].turn_id != m.turn_id:
                    flush()
                pending.append(m)
                continue
            flush()
            if (lc := to_langchain_single(m)) is not None:
                out.append(lc)

        flush()
        return out
