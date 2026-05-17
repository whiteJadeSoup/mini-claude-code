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
from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from mini_cc.engine.agent_loop import AgentLoop
from mini_cc.engine.compact import (
    _is_context_exceeded,
    _parse_context_gap_tokens,
    run_compact,
)
from mini_cc.engine.sandbox import Sandbox, allow_all
from mini_cc.engine.messages import (
    AssistantMessage,
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
from mini_cc.skills import _skill_manager


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

    async def run_loop(
        self,
        *,
        parent_id: str | None,
        source: str,
        sandbox: Sandbox = allow_all(),
        is_sub_agent: bool = False,
    ) -> str | None:
        """Run agent loop iterations, dispatching each yielded message.

        The shared core of `query()` and `run_sidechain()`. Callers own
        any wrapping work — seed messages (User/SystemPrompt), retry on
        context overflow, `_sub_agent_scope` setup. This method just
        drives `AgentLoop.run` and routes messages to consumers.

        Args:
            parent_id: None for main branch; an AssistantMessage id for
                sidechains.
            source: usage-tracking label propagated into every emitted
                message.
            sandbox: per-call tool authorization gate. Default
                `allow_all()` matches the historical unrestricted
                behavior; sub-agent launch points pass a narrower gate.
            is_sub_agent: selects the pre-built AgentLoop — False uses
                `_main_loop` (full MAIN_TOOLS, includes task/run_skill);
                True uses `_sub_loop` (SUB_TOOLS, excludes task/run_skill
                to prevent recursion at the schema level — sandbox can
                still further restrict).

        Returns the last TextBlock text emitted by an AssistantMessage,
        or None if the loop ended without producing text. Used by
        sub-agents to surface their final answer as a tool_result.
        """
        loop = self._sub_loop if is_sub_agent else self._main_loop
        last_text: str | None = None
        async for msg in loop.run(
            get_messages=lambda: self._prepare_messages(parent_id=parent_id),
            parent_id=parent_id,
            source=source,
            sandbox=sandbox,
        ):
            await self._dispatch(msg)
            # Track the most recent TextBlock so sub-agents can return it
            # as the tool result. The loop returns after a no-tool-call
            # response, so the last TextBlock is the final answer.
            if isinstance(msg, AssistantMessage) and isinstance(
                msg.content, TextBlock
            ):
                last_text = msg.content.text
        return last_text

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
                    await self.run_loop(
                        parent_id=None,
                        source="agent",
                        is_sub_agent=False,
                    )
                    break  # success
                except Exception as e:
                    if attempt == 0 and _is_context_exceeded(e):
                        print("[context exceeded — auto-compacting]",
                              file=sys.stderr, flush=True)
                        try:
                            from mini_cc.consumers import persistence
                            from mini_cc.engine._diagnostics import log_event, tracker_snapshot
                            gap = _parse_context_gap_tokens(str(e))
                            log_event(
                                "astream_context_exceeded",
                                session_id=persistence.SESSION_ID,
                                source="agent",
                                tracker_before=tracker_snapshot(),
                                store_estimate_chars=sum(
                                    len(m.content) if hasattr(m, "content") and isinstance(m.content, str) else 0
                                    for m in self.store.api_view(parent_id=None)
                                ),
                                error=str(e),
                                gap_tokens=gap,
                                caught_by="query_retry",
                            )
                            await self.compact(auto=True, trigger="query_retry")
                            # Re-add the user's original message so the model
                            # still knows what was asked after compact clears it.
                            await self._dispatch(UserMessage(content=user_text, source="user"))
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
        *,
        sandbox: Sandbox = allow_all(),
    ) -> str:
        """Run a sub-agent on a sidechain branch; return final text.

        Sidechain messages are dispatched to consumers but NOT yielded
        back to the caller — this method is invoked from inside a tool's
        `ainvoke`, and you cannot yield through an awaited call. The
        tool's return value (the last TextBlock's text) becomes the main
        branch's ToolResultMessage content.

        `sandbox` is the per-call tool authorization gate for this
        sub-agent. Defaults to `allow_all()` (preserving the prior
        unrestricted behavior); a launch point with restricted intent
        (e.g. memory extraction) should pass a narrower sandbox.
        """
        # Lazy import: tools.py imports this module (via get_engine),
        # and we need _sub_agent_scope from tools without a cycle at load.
        from mini_cc.tools._utils import _sub_agent_scope

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
            last_text = await self.run_loop(
                parent_id=parent_id,
                source=label,
                sandbox=sandbox,
                is_sub_agent=True,
            )
        return last_text or "(completed, no summary)"

    # -- compact ------------------------------------------------------------

    _MAX_COMPACT_ATTEMPTS = 3  # K in the plan — retry budget for overflow recovery

    async def compact(
        self,
        custom_instructions: str = "",
        auto: bool = False,
        trigger: str = "manual",
    ) -> int:
        """Thin wrapper over engine.compact.run_compact.

        Body lives in compact.py — it owns the API-round trimming + retry
        state machine + LLM call. Only the public method shape stays here
        because /compact and the auto-trigger path call `engine.compact(...)`.
        """
        return await run_compact(
            self,
            custom_instructions=custom_instructions,
            auto=auto,
            trigger=trigger,
            max_attempts=self._MAX_COMPACT_ATTEMPTS,
        )

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

        Runs _clear_old_tool_results to free space, then asks the tracker
        whether we have enough headroom for one more round + compact. The
        tracker's answer is API-baseline + a char estimate of messages
        added since the last API response (mostly tool_results from the
        current turn) — see UsageTracker.context_tokens_used.
        """
        self._clear_old_tool_results(parent_id)
        messages = self.store.api_view(parent_id=parent_id)

        if self._should_auto_compact(messages):
            try:
                await self.compact(auto=True, trigger="pre_call")
                messages = self.store.api_view(parent_id=parent_id)
            except Exception as e:  # noqa: BLE001
                from mini_cc.consumers import persistence
                from mini_cc.engine._diagnostics import log_event, tracker_snapshot
                log_event(
                    "astream_context_exceeded",
                    session_id=persistence.SESSION_ID,
                    source=f"_prepare_messages(parent_id={parent_id})",
                    tracker_before=tracker_snapshot(),
                    store_estimate_chars=sum(
                        len(m.content) if hasattr(m, "content") and isinstance(m.content, str) else 0
                        for m in messages
                    ),
                    error=str(e),
                    gap_tokens=_parse_context_gap_tokens(str(e)),
                    caught_by="_prepare_messages",
                )
                print(f"[auto-compact failed: {e}]", file=sys.stderr, flush=True)

        return messages

    # Time-based opportunistic clearing of old tool_result content.
    # Modeled after CC's microCompact (services/compact/microCompact.ts +
    # services/compact/timeBasedMCConfig.ts default `gapThresholdMinutes: 60`,
    # `keepRecent: 5`).
    #
    # The original mini-cc design fired this hook on every prepare_messages
    # call, keeping only the most recent tool batch. That made file_read
    # dedup stubs reference deleted content (the stub said "refer to the
    # earlier tool_result" but that result had already been wiped to
    # "[Cleared]") — the LLM observed the broken contract and fell back to
    # execute_command(cat). Verified via diagnostic logging on 2026-05-10:
    # 129 file_read calls → 183 tool_results cleared by session end.
    #
    # Aligning with CC: clear only on idle gap, keep last N. Active sessions
    # never trip this; budget overflow within an active session is handled
    # by the existing auto-compact path (_should_auto_compact).
    _TIME_BASED_CLEAR_GAP_SEC = 60 * 60   # 60 min idle, matches CC default
    _TIME_BASED_CLEAR_KEEP = 5            # last 5 tool_results spared

    def _clear_old_tool_results(self, parent_id: str | None) -> None:
        """Wipe old ToolResultMessage content to '[Cleared]' when idle.

        MUTATION CONTRACT: mutates ToolResultMessage .content/.output on
        live store objects so api_view() returns "[Cleared]" for those
        slots. Consumers must NOT cache .content/.output after dispatch —
        TUI has already rendered, JSONL has already persisted the original.
        Future replay reads JSONL, not mutated objects.
        """
        branch = [
            m for m in self.store._messages
            if m.parent_id == parent_id
        ]

        # Idle-gap trigger: most-recent UserMessage (real or synthetic) on
        # this branch vs now. Synthetic users (e.g. compact body, task_state)
        # count — they reset the clock just like a real turn would, since
        # they represent a fresh conversational anchor the LLM should not
        # immediately discard.
        last_user_at = None
        for m in reversed(branch):
            if isinstance(m, UserMessage):
                last_user_at = m.created_at
                break
        if last_user_at is None:
            return
        gap_sec = (datetime.now(timezone.utc) - last_user_at).total_seconds()
        if gap_sec < self._TIME_BASED_CLEAR_GAP_SEC:
            return

        tool_results = [m for m in branch if isinstance(m, ToolResultMessage)]
        if len(tool_results) <= self._TIME_BASED_CLEAR_KEEP:
            return

        cleared_marker = "[Cleared]"
        for m in tool_results[:-self._TIME_BASED_CLEAR_KEEP]:
            if not m.content.startswith(cleared_marker):
                m.content = cleared_marker
                m.output = None

    # Buffer for compact's own LLM call. The other two reserve constants
    # (COMPACT_SUMMARY_RESERVE, API_ROUND_RESERVE) live in compact.py since
    # they only matter inside run_compact's budget calculation; this one is
    # specific to the auto-trigger headroom check below.
    _COMPACT_CALL_RESERVE = 20_000

    def _should_auto_compact(self, messages: list[BaseMessage]) -> bool:
        """Compact when less than ~53k tokens of headroom remain.

        The reserve covers: room for compact's summary output, one API
        round's overhead, and a safety margin so compact's own LLM call
        doesn't fail on first attempt due to token-estimation drift.
        """
        from mini_cc.engine.compact import COMPACT_SUMMARY_RESERVE, API_ROUND_RESERVE
        reserve = COMPACT_SUMMARY_RESERVE + API_ROUND_RESERVE + self._COMPACT_CALL_RESERVE
        return usage._tracker.headroom_left(messages) < reserve

    def current_context_tokens(self, parent_id: str | None = None) -> int:
        """Convenience wrapper for UI callers that don't already hold api_view.

        `/context` and the TUI status bar read this — they don't want to
        assemble a LangChain message list themselves. Tracker logic stays
        pure (no store reference) by funnelling through here.
        """
        return usage._tracker.context_tokens_used(
            self.store.api_view(parent_id=parent_id)
        )


# Compact-related helpers and the `run_compact` driver live in
# mini_cc.engine.compact — only the auto-trigger plumbing and the
# `is_context_exceeded` predicate are imported above for query()'s retry
# loop and _should_auto_compact's headroom check.
