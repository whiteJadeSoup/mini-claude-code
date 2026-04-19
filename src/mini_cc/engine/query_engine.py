"""QueryEngine — async multi-turn conversation manager.

The engine owns the MessageStore and is the single writer of messages.
Consumers (UI, persistence) subscribe and receive each message via push
(`await on_message(msg)`). A turn is exposed as an async iterator
(`async for msg in engine.query(text)`) so tests and scripted callers can
also observe the sequence.

Why async + generator:
- Tool calls are awaitable (`tool_fn.ainvoke`). A sub-agent tool runs
  another turn inside its invocation — making the outer loop async lets
  that recursion happen naturally instead of via nested sync stacks.
- Yielding messages as they are produced (text → yield; each tool_use →
  yield; each tool_result → yield) means the UI consumer can render
  incrementally without being wired into the LLM stream itself.

Why engine owns the store:
- One writer, one code path (`_dispatch`). Easier to reason about
  ordering and persistence guarantees.
- Consumers never touch the store; they only react to messages. This
  keeps the store a pure data structure and keeps consumers swappable.
"""
from __future__ import annotations

import sys
import uuid
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from mini_cc import prompts
from mini_cc.engine.agent_loop import AgentLoop
from mini_cc.engine.messages import (
    AssistantMessage,
    CompactBoundaryMessage,
    Message,
    SystemPromptMessage,
    StatusMessage,
    TextBlock,
    ToolResultMessage,
    ToolUseBlock,
    UserMessage,
)
from mini_cc.engine.predicates import Predicate, accept_all
from mini_cc.engine.store import MessageStore
from mini_cc.engine.subscription import Consumer, Policy, Subscription
from mini_cc.engine.transforms import Transform, identity
from mini_cc.state import tasks, usage
from mini_cc.tools.skills import _skill_manager


# ---------------------------------------------------------------------------
# Module singleton (tools resolve engine lazily)
# ---------------------------------------------------------------------------


_engine: "QueryEngine | None" = None


def get_engine() -> "QueryEngine":
    """Tools call this to reach the engine without an import cycle.

    Tools live in tools.py; the engine imports tools (indirectly via
    the main_tools list). If tools imported QueryEngine at module load,
    we'd have a cycle. The singleton + lazy lookup sidesteps it — the
    same pattern we use for todos._todos / tasks._tasks.
    """
    if _engine is None:
        raise RuntimeError("QueryEngine not initialized; call set_engine first")
    return _engine


def set_engine(engine: "QueryEngine") -> None:
    global _engine
    _engine = engine


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------


class QueryEngine:
    def __init__(
        self,
        llm_base: BaseChatModel,
        main_tools: list[BaseTool],
        sub_tools: list[BaseTool],
        model_name: str,
        system_prompt_builder: Callable[[], str],
    ) -> None:
        self._llm_base = llm_base
        self._model_name = model_name
        self._build_system_prompt = system_prompt_builder
        self.store = MessageStore()
        self._subscriptions: list[Subscription] = []
        # Pre-built AgentLoop instances for main branch and sidechains.
        # Construction is cheap — split purely so each loop owns its own
        # bound_llm + tool registry. Engine keeps no loop state.
        self._main_loop = AgentLoop(
            bound_llm=llm_base.bind_tools(main_tools),
            tools_by_name={t.name: t for t in main_tools},
            model_name=model_name,
        )
        self._sub_loop = AgentLoop(
            bound_llm=llm_base.bind_tools(sub_tools),
            tools_by_name={t.name: t for t in sub_tools},
            model_name=model_name,
        )

    # -- subscribers --------------------------------------------------------

    def subscribe(
        self,
        consumer: Consumer,
        *,
        name: str | None = None,
        filter: Predicate | None = None,
        transform: Transform | None = None,
        policy: Policy = "sync",
        drop_oldest_maxsize: int = 256,
    ) -> Subscription:
        """Register a consumer. Kwargs declare *what* it gets and *how*.

        The default ``policy="sync"`` preserves the old direct-await
        dispatch semantics (tests observe messages the moment
        ``engine.query`` returns). Consumers that need their own drain
        task (persistence, UI) opt in with ``policy="async"``.
        """
        sub = Subscription(
            consumer=consumer,
            name=name or type(consumer).__name__,
            filter=filter if filter is not None else accept_all,
            transform=transform if transform is not None else identity,
            policy=policy,
            drop_oldest_maxsize=drop_oldest_maxsize,
        )
        self._subscriptions.append(sub)
        return sub

    @property
    def subscriptions(self) -> tuple[Subscription, ...]:
        return tuple(self._subscriptions)

    async def _dispatch(self, msg: Message) -> None:
        """Append to store, then route to each subscription whose filter matches.

        Filter exceptions are isolated per-subscription: a misbehaving
        predicate drops the match for that subscription (recorded in
        ``stats``) but does not stop the rest of the fan-out. Consumer
        exceptions are isolated inside ``Subscription.deliver``.
        """
        self.store.append(msg)
        for sub in self._subscriptions:
            try:
                if not sub.filter(msg):
                    continue
            except Exception as e:  # noqa: BLE001
                sub.record_error(f"filter: {e!r}")
                continue
            await sub.deliver(msg)

    # -- boot ---------------------------------------------------------------

    async def shutdown(self) -> None:
        """Drain and stop every subscription. Call once before process exit."""
        for sub in self._subscriptions:
            try:
                await sub.stop()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[engine: sub {sub.name!r} stop failed: {e}]",
                    file=sys.stderr,
                    flush=True,
                )

    async def boot(self) -> None:
        """Seed the session with a system prompt and any pending task state.

        Call this exactly once, after subscribing consumers. The initial
        system prompt + task-state injection must be visible to
        persistence and UI, so they go through _dispatch.
        """
        for sub in self._subscriptions:
            await sub.start()
        await self._dispatch(
            SystemPromptMessage(content=self._build_system_prompt(), source="boot")
        )
        state = tasks._tasks.state_summary()
        if state:
            # Kept out of the system prompt so DeepSeek's prefix cache stays
            # stable turn-to-turn. Same rationale as the previous sync code.
            await self._dispatch(
                UserMessage(content=state, is_synthetic=True, source="task_state")
            )
            await self._dispatch(
                AssistantMessage(
                    turn_id=f"local-{uuid.uuid4()}",
                    model=self._model_name,
                    content=TextBlock(
                        text="Acknowledged. I have the current task plan."
                    ),
                    source="task_state",
                )
            )

    # -- turn entries -------------------------------------------------------

    async def query(self, user_text: str) -> None:
        """Run one main-branch turn.

        Consumers observe the message stream via `on_message`; this method
        returns None. Callers that want to await turn completion just
        `await engine.query(text)` — no need to drain an iterator.

        If the first LLM call fails with a context-length error, one auto-
        compact is attempted and the turn retries from the compact summary.
        """
        await self._dispatch(UserMessage(content=user_text, source="user"))
        await self._dispatch(StatusMessage(event="turn_start", source="agent"))
        try:
            for attempt in range(2):
                try:
                    async for msg in self._main_loop.run(
                        get_messages=lambda: self._prepare_messages(parent_id=None),
                        parent_id=None,
                        source="agent",
                    ):
                        await self._dispatch(msg)
                    break  # success
                except Exception as e:
                    if attempt == 0 and _is_context_exceeded(e):
                        print("[context exceeded — auto-compacting]",
                              file=sys.stderr, flush=True)
                        try:
                            await self.compact(auto=True)
                            # Re-add the user's original message so the model
                            # still knows what was asked after compact clears it.
                            self.store._messages.append(
                                UserMessage(content=user_text, source="user")
                            )
                        except Exception as ce:
                            print(f"[auto-compact failed: {ce}]",
                                  file=sys.stderr, flush=True)
                            raise e  # give up; surface original error
                        continue
                    raise
        finally:
            await self._dispatch(StatusMessage(event="turn_end", source="agent"))

    async def run_sidechain(
        self,
        parent_id: str,
        system_prompt: str,
        user_prompt: str,
        label: str,
    ) -> str:
        """Run a sub-agent on a sidechain branch; return final text.

        Sidechain messages are dispatched to consumers but NOT yielded
        back to the caller — this method is invoked from inside a tool's
        `ainvoke`, and you cannot yield through an awaited call. The
        tool's return value (the last TextBlock's text) becomes the main
        branch's ToolResultMessage content.
        """
        # Lazy import: tools.py imports this module (via get_engine),
        # and we need _sub_agent_scope from tools without a cycle at load.
        from mini_cc.tools.builtins import _sub_agent_scope

        last_text: str | None = None
        with _sub_agent_scope(label):
            await self._dispatch(
                SystemPromptMessage(
                    content=system_prompt, parent_id=parent_id, source=label
                )
            )
            await self._dispatch(
                UserMessage(
                    content=user_prompt,
                    is_synthetic=True,
                    parent_id=parent_id,
                    source=label,
                )
            )
            async for msg in self._sub_loop.run(
                get_messages=lambda: self._prepare_messages(parent_id=parent_id),
                parent_id=parent_id,
                source=label,
            ):
                await self._dispatch(msg)
                # Track the most recent TextBlock so we can return it as
                # the tool result. The loop returns after a no-tool-call
                # response, so the last TextBlock is the final answer.
                if isinstance(msg, AssistantMessage) and isinstance(
                    msg.content, TextBlock
                ):
                    last_text = msg.content.text
        return last_text or "(completed, no summary)"

    # -- compact ------------------------------------------------------------

    async def compact(self, custom_instructions: str = "", auto: bool = False) -> int:
        """Summarize, clear Layer 1, and re-seed with system + boundary + summary.

        Does NOT yield. Compact is a boundary event, not a conversational
        turn — its messages reach consumers via _dispatch directly. This
        matches the semantics of `/compact` (callers want a return value,
        not an iterator).
        """
        from mini_cc.tools.builtins import _sub_agent_scope

        # Snapshot the system prompt so we can re-insert it after
        # clear_layer_1. We insert it via direct list manipulation
        # (store._messages.insert(0, ...)) to avoid re-dispatching a
        # message consumers have already seen.
        system_msg = next(
            (
                m
                for m in self.store.all()
                if isinstance(m, SystemPromptMessage) and m.parent_id is None
            ),
            None,
        )
        pre_count = len(self.store.all())

        langchain_msgs = self.store.api_view(parent_id=None)
        # Skip the SystemMessage (index 0) — compact prompt has its own.
        formatted = _format_history_for_summary(langchain_msgs[1:])
        if custom_instructions:
            formatted += f"\n\n## Compact Instructions\n{custom_instructions}"

        usage._tracker.reset()
        with _sub_agent_scope("compact"):
            response = await self._llm_base.ainvoke(
                [
                    SystemMessage(content=prompts.COMPACT_PROMPT),
                    HumanMessage(content=formatted),
                ]
            )
            usage._tracker.record(
                "compact",
                response.usage_metadata,
                getattr(response, "response_metadata", None),
            )

        summary = _extract_summary(response.content)
        body = _build_compact_body(summary, auto=auto)
        if task_state := tasks._tasks.state_summary():
            body += f"\n\n---\n\nActive tasks at time of compaction:\n{task_state}"

        self.store.clear_layer_1()
        if system_msg is not None:
            # Direct insert: consumers already saw this message at boot.
            # Re-dispatching would duplicate JSONL + re-render in UI.
            self.store._messages.insert(0, system_msg)

        await self._dispatch(
            CompactBoundaryMessage(pre_count=pre_count, auto=auto, source="compact")
        )
        await self._dispatch(
            UserMessage(content=body, is_synthetic=True, source="compact")
        )

        return max(0, pre_count - len(self.store.all()))

    # -- skills -------------------------------------------------------------

    async def refresh_skills(self) -> bool:
        """Rescan skills; update system prompt + emit StatusMessage on change.

        Returns True iff anything changed. Called by main.py before each
        user input so newly dropped-in skills are picked up without a
        restart.
        """
        from mini_cc.commands import sync_skill_commands

        old_names = set(_skill_manager.names())
        _skill_manager.rescan()
        new_names = set(_skill_manager.names())
        changed = sync_skill_commands(_skill_manager)
        if not changed:
            return False

        self.store.replace_system_prompt(
            SystemPromptMessage(content=self._build_system_prompt(), source="boot")
        )
        await self._dispatch(
            StatusMessage(
                event="skills_changed",
                data={
                    "added": sorted(new_names - old_names),
                    "removed": sorted(old_names - new_names),
                },
                source="skills",
            )
        )
        return True

    # -- pre-call hook ------------------------------------------------------

    async def _prepare_messages(
        self, parent_id: str | None
    ) -> list[BaseMessage]:
        """Called by AgentLoop before each LLM request.

        Two-pass compact check:
        1. Token-based: uses the last LLM call's recorded input token count.
        2. Size-based: estimates the next call's actual payload size from
           character count. Catches the case where a large tool result pushed
           context over the limit between calls — the token-based check is
           blind to tokens added after the last API response.
        """
        self._clear_old_tool_results(parent_id)
        messages = self.store.api_view(parent_id=parent_id)

        needs_compact = self._should_auto_compact() or self._payload_too_large(messages)
        if needs_compact:
            try:
                await self.compact(auto=True)
                messages = self.store.api_view(parent_id=parent_id)
            except Exception as e:  # noqa: BLE001
                print(f"[auto-compact failed: {e}]", file=sys.stderr, flush=True)

        return messages

    def _clear_old_tool_results(self, parent_id: str | None) -> None:
        """Replace old ToolResultMessage content with '[Cleared]' in-place.

        Keeps the most recent tool-result group intact (the LLM needs
        those to reason). Earlier tool outputs are large and no longer
        useful; collapsing them reclaims context budget.
        """
        branch = [
            m
            for m in self.store._messages
            if m.parent_id == parent_id
            and isinstance(m, (AssistantMessage, ToolResultMessage))
        ]
        last_tool_asst_idx = -1
        for i, m in enumerate(branch):
            if isinstance(m, AssistantMessage) and isinstance(m.content, ToolUseBlock):
                last_tool_asst_idx = i
        if last_tool_asst_idx <= 0:
            return
        for m in branch[:last_tool_asst_idx]:
            if isinstance(m, ToolResultMessage) and not m.content.startswith(
                "[Cleared]"
            ):
                m.content = "[Cleared]"

    # Headroom constants kept out of the condition so they're easy to tune.
    # Large headroom is intentional: compact() itself sends the full history
    # to the LLM, so we must leave room for that call to succeed.
    _COMPACT_SUMMARY_RESERVE = 20_000   # typical compact summary length
    _API_ROUND_RESERVE = 13_000         # one turn of thinking + response
    _COMPACT_CALL_RESERVE = 20_000      # buffer for compact's own LLM call

    def _should_auto_compact(self) -> bool:
        """Compact when projected next input would leave less than ~53k headroom."""
        headroom = self._COMPACT_SUMMARY_RESERVE + self._API_ROUND_RESERVE + self._COMPACT_CALL_RESERVE
        return (
            usage._tracker.projected_next_input() + headroom
            > usage._tracker.context_limit
        )

    def _payload_too_large(self, messages: list[BaseMessage]) -> bool:
        """Same condition, but using character-based token estimate.

        Catches the gap where a large tool result was added after the last
        LLM call — projected_next_input() won't see those tokens yet.
        Uses // 2 (not // 3) because code, JSON, and Chinese text run closer
        to 1-2 chars/token; underestimating here causes missed compacts.
        """
        estimated = sum(len(str(getattr(m, "content", ""))) for m in messages) // 2
        headroom = self._COMPACT_SUMMARY_RESERVE + self._API_ROUND_RESERVE + self._COMPACT_CALL_RESERVE
        return estimated + headroom > usage._tracker.context_limit


# ---------------------------------------------------------------------------
# Private helpers (copied from agent.py — they had no dependencies)
# ---------------------------------------------------------------------------


def _is_context_exceeded(e: Exception) -> bool:
    msg = str(e)
    return "maximum context length" in msg or "context_length_exceeded" in msg


def _format_history_for_summary(messages: list) -> str:
    import json as _json

    from langchain_core.messages import AIMessage, HumanMessage

    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                parts.append(f"Assistant: {msg.content}")
            for tc in (msg.tool_calls or []):
                parts.append(
                    f"  → {tc['name']}({_json.dumps(tc.get('args', {}), ensure_ascii=False)})"
                )
        else:
            content = getattr(msg, "content", "")
            if content:
                parts.append(f"Tool result: {content}")
    return "\n\n".join(parts)


def _extract_summary(content: str) -> str:
    import re as _re

    m = _re.search(r"<summary>(.*?)</summary>", content, _re.DOTALL)
    return m.group(1).strip() if m else content


def _build_compact_body(summary: str, auto: bool) -> str:
    from mini_cc.consumers import persistence

    path = persistence.transcript_path()
    if auto:
        intro = (
            "This session is being continued from a previous conversation that ran "
            "out of context. The summary below covers the earlier portion of the "
            "conversation."
        )
        tail = (
            "\n\nContinue the conversation from where it left off without asking the "
            "user any further questions. Resume directly — do not acknowledge the "
            "summary, do not recap what was happening, do not preface with 'I'll "
            "continue' or similar. Pick up the last task as if the break never happened."
        )
    else:
        intro = (
            "The conversation has been compacted at the user's request. The summary "
            "below covers the earlier portion of the conversation; use it as context "
            "for the next message."
        )
        tail = ""
    return (
        f"{intro}\n\n"
        f"Summary:\n{summary}\n\n"
        "If you need specific details from before compaction (exact code snippets, "
        "error messages, tool output, or content you generated), read the full "
        f"transcript at: {path}"
        f"{tail}"
    )
