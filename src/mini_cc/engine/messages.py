"""Unified message framework — all session event types.

Three layers:
  Layer 1 (API): sent to LLM + persisted — SystemPromptMessage, UserMessage,
                 AssistantMessage, ToolResultMessage
  Layer 2 (UI):  persisted, never sent to API — CompactBoundaryMessage, StatusMessage
  Layer 3 (Control): transient, not stored — future use
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field


def _default_session_id() -> str:
    from mini_cc.consumers import persistence
    return persistence.SESSION_ID


def _default_cwd() -> str:
    from mini_cc import config
    return config.CWD


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=_default_session_id)
    cwd: str = Field(default_factory=_default_cwd)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_id: str | None = None
    source: str = "user"


# --- ContentBlock discriminated union ---

class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    call_id: str
    name: str
    args: dict


ContentBlock = Annotated[
    Union[TextBlock, ToolUseBlock],
    Field(discriminator="type"),
]


# --- Layer 1: API Messages ---

class SystemPromptMessage(Message):
    type: Literal["system_prompt"] = "system_prompt"
    content: str


class UserMessage(Message):
    type: Literal["user"] = "user"
    content: str
    is_synthetic: bool = False


class AssistantMessage(Message):
    type: Literal["assistant"] = "assistant"
    turn_id: str
    model: str
    content: ContentBlock  # exactly ONE block per message


class ToolResultMessage(Message):
    type: Literal["tool_result"] = "tool_result"
    content: str
    tool_call_id: str


# --- Layer 2: UI Messages ---

class CompactBoundaryMessage(Message):
    type: Literal["compact_boundary"] = "compact_boundary"
    pre_count: int
    auto: bool
    # Audit metadata populated by compact(). Defaults keep older JSONL
    # records deserializable after this change.
    dropped_rounds: int = 0
    retained_rounds: int = 0
    marker_used: bool = False
    attempts: int = 1
    original_chars: int = 0
    sent_chars: int = 0


class StatusMessage(Message):
    type: Literal["status"] = "status"
    event: str
    data: dict = Field(default_factory=dict)


# --- Layer classification ---

LAYER_1_TYPES = (SystemPromptMessage, UserMessage, AssistantMessage, ToolResultMessage)
LAYER_2_TYPES = (CompactBoundaryMessage, StatusMessage)


# --- Conversions ---

def to_langchain_single(msg: Message) -> BaseMessage | None:
    """Convert one Layer-1 Message to its LangChain counterpart.

    AssistantMessage returns None — must be merged by turn_id via api_view first.
    Layer 2 messages return None (not API-visible).
    """
    if isinstance(msg, SystemPromptMessage):
        return SystemMessage(content=msg.content)
    if isinstance(msg, UserMessage):
        return HumanMessage(content=msg.content)
    if isinstance(msg, AssistantMessage):
        return None
    if isinstance(msg, ToolResultMessage):
        return ToolMessage(content=msg.content, tool_call_id=msg.tool_call_id)
    return None


def assistant_messages_from_ai(
    ai: AIMessage,
    model: str,
    parent_id: str | None,
    source: str,
) -> list[AssistantMessage]:
    """Split one AIMessage into N AssistantMessages, one ContentBlock each.

    turn_id = response_metadata["id"] when available; falls back to local UUID.
    Empty text content is skipped (I4: tool-only responses produce no TextBlock).
    """
    turn_id = (ai.response_metadata or {}).get("id") or f"local-{uuid.uuid4()}"
    msgs: list[AssistantMessage] = []

    if ai.content:
        msgs.append(AssistantMessage(
            turn_id=turn_id,
            model=model,
            content=TextBlock(text=ai.content),
            parent_id=parent_id,
            source=source,
        ))

    for tc in (ai.tool_calls or []):
        msgs.append(AssistantMessage(
            turn_id=turn_id,
            model=model,
            content=ToolUseBlock(
                call_id=tc["id"],
                name=tc["name"],
                args=tc.get("args", {}),
            ),
            parent_id=parent_id,
            source=source,
        ))

    return msgs
