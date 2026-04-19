"""Named predicates and combinators for declarative subscription filters.

A predicate is any ``Callable[[Message], bool]``. Subscribers use these to
declare *what* messages they want at wire-up time, so engine dispatch
never has to know which consumer cares about which message type.
"""
from __future__ import annotations

from typing import Callable

from mini_cc.engine.messages import (
    LAYER_1_TYPES,
    LAYER_2_TYPES,
    AssistantMessage,
    CompactBoundaryMessage,
    Message,
    StatusMessage,
    ToolResultMessage,
    UserMessage,
)

Predicate = Callable[[Message], bool]


def accept_all(m: Message) -> bool:
    return True


def is_main_branch(m: Message) -> bool:
    return m.parent_id is None


def is_assistant(m: Message) -> bool:
    return isinstance(m, AssistantMessage)


def is_user(m: Message) -> bool:
    return isinstance(m, UserMessage)


def is_synthetic_user(m: Message) -> bool:
    return isinstance(m, UserMessage) and m.is_synthetic


def is_status(m: Message) -> bool:
    return isinstance(m, StatusMessage)


def is_tool_result(m: Message) -> bool:
    return isinstance(m, ToolResultMessage)


def is_compact_boundary(m: Message) -> bool:
    return isinstance(m, CompactBoundaryMessage)


def is_persisted_layer(m: Message) -> bool:
    return isinstance(m, LAYER_1_TYPES + LAYER_2_TYPES)


def all_of(*ps: Predicate) -> Predicate:
    return lambda m: all(p(m) for p in ps)


def any_of(*ps: Predicate) -> Predicate:
    return lambda m: any(p(m) for p in ps)


def negate(p: Predicate) -> Predicate:
    return lambda m: not p(m)
