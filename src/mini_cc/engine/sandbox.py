"""Sandbox — per-call tool authorization for AgentLoop.

A Sandbox decides, for each (tool, args) pair the LLM wants to invoke,
whether the call proceeds (`Allow`, optionally with rewritten args) or is
short-circuited into a synthetic error tool_result (`Deny`).

This is intentionally *just* an authorization gate. No UI prompting, no
permission modes, no rules engine — those belong to a product wrapping
mini-cc, not to the agent loop. The protocol mirrors CC's `CanUseToolFn`
in shape (per-call async callback returning a decision) while dropping
the parts mini-cc has no use for (the `ask` variant, decisionReason union,
ToolUseContext bag).

Typical use: a launch point (TUI, task tool, run_skill tool, future
memory extraction sub-agent) constructs a Sandbox and passes it to
`AgentLoop.run(sandbox=...)`. The LLM still sees the full tool list — the
sandbox gates at execution time. A denied call yields a tool_result whose
content explains the denial, letting the LLM self-correct on the next turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mini_cc.tools.base import MiniTool


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Allow:
    """The call proceeds.

    `updated_args` lets the sandbox rewrite the LLM's args before execution
    (e.g. canonicalize a path, strip a field the policy wants discarded).
    None means "use the LLM's args unchanged".
    """
    updated_args: dict | None = None


@dataclass(frozen=True)
class Deny:
    """The call is rejected. `reason` becomes the tool_result content the
    LLM sees, so it should read naturally as feedback the model can act on
    (e.g. "file_write only allowed under .claude/memory/, got: /etc/passwd")."""
    reason: str


Decision = Allow | Deny


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Sandbox(Protocol):
    """Per-call gate for tool invocations.

    `name` is used for logging and for the `[sandbox:<name>] denied: ...`
    prefix on synthesized error messages — keep it short and stable.

    `check` is async to accommodate sandboxes that need to consult external
    state (file metadata, an LLM classifier, etc.) without a later signature
    break. Pure in-memory checks are still trivial — just `return Allow()`.
    """
    name: str

    async def check(self, tool: MiniTool, args: dict) -> Decision: ...


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------


class AllowAll:
    """Permits every call. Default for the main agent; safe singleton."""
    name = "allow_all"

    async def check(self, tool: MiniTool, args: dict) -> Decision:
        return Allow()


# Singleton for default kwargs — AllowAll is stateless so sharing one
# instance avoids constructing a throwaway per AgentLoop.run() call.
_ALLOW_ALL = AllowAll()


def allow_all() -> Sandbox:
    """Returns the shared AllowAll instance."""
    return _ALLOW_ALL


@dataclass(frozen=True)
class StaticSandbox:
    """Declarative tool-name allow/deny lists.

    Semantics:
        - `allow=None`        : no allow list — anything not in `deny` is allowed.
        - `allow=frozenset()` : explicit empty allow — everything is denied
                                 (unless also covered by a more specific subclass).
        - `deny`              : always wins; a name in both is denied.

    Use this for the common "main agent minus a few tools" pattern, e.g.
    a sub-agent that denies `task` / `run_skill` to prevent recursion.
    """
    name: str
    allow: frozenset[str] | None = None
    deny: frozenset[str] = frozenset()

    async def check(self, tool: MiniTool, args: dict) -> Decision:
        if tool.name in self.deny:
            return Deny(f"tool '{tool.name}' is in the deny list")
        if self.allow is not None and tool.name not in self.allow:
            return Deny(
                f"tool '{tool.name}' is not in the allow list "
                f"(allowed: {', '.join(sorted(self.allow)) or '<none>'})"
            )
        return Allow()


class ReadOnlySandbox:
    """Allows any tool with `is_read_only=True`; denies the rest.

    `extra_allow` lets a caller permit specific non-read-only tools by name
    (e.g. `todo_plan` in a planning sub-agent that has no filesystem access
    but should still be able to manage its todo list).
    """
    name = "read_only"

    def __init__(self, extra_allow: frozenset[str] = frozenset()) -> None:
        self._extra_allow = frozenset(extra_allow)

    async def check(self, tool: MiniTool, args: dict) -> Decision:
        if tool.is_read_only or tool.name in self._extra_allow:
            return Allow()
        return Deny(
            f"tool '{tool.name}' is not read-only "
            f"(this sub-agent is restricted to read-only tools)"
        )


# ---------------------------------------------------------------------------
# Shared presets
# ---------------------------------------------------------------------------


# Default sandbox for sub-agents launched via `task` / `run_skill`. Denies
# the recursion-prone delegation tools so a sub-agent cannot spawn further
# sub-agents. This duplicates the protection currently provided by
# `_sub_loop`'s tool binding (which already excludes these names from the
# LLM's tool list) — having it at the call site makes the "this is a
# sub-agent restriction" intent explicit and survives any future change to
# how loops are constructed.
SUB_AGENT_SANDBOX: Sandbox = StaticSandbox(
    name="sub_agent",
    deny=frozenset({"task", "run_skill"}),
)
