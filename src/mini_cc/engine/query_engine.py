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

import re
import sys
import uuid
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

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

    _MAX_COMPACT_ATTEMPTS = 3  # K in the plan — retry budget for overflow recovery

    async def compact(self, custom_instructions: str = "", auto: bool = False) -> int:
        """Summarize, clear Layer 1, and re-seed with system + boundary + summary.

        Does NOT yield. Compact is a boundary event, not a conversational
        turn — its messages reach consumers via _dispatch directly. This
        matches the semantics of `/compact` (callers want a return value,
        not an iterator).

        Overflow handling: the compact call itself can exceed the context
        window when the history is large. We split the history into
        ApiRounds, drop the oldest until it fits, and retry up to
        _MAX_COMPACT_ATTEMPTS times, using the provider's error message
        to compute the next drop size (DeepSeek gap / 20% fallback).
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

        all_msgs = self.store.api_view(parent_id=None)
        # Skip the SystemMessage (index 0) — compact prompt has its own.
        history = all_msgs[1:]
        groups = _group_by_api_round(history)
        original_chars = sum(g.size_chars for g in groups)

        # Budget for the compact call's HumanMessage payload.
        # context_limit includes completion reserve, so subtract reserves
        # for the summary output and one round of API overhead; subtract
        # COMPACT_PROMPT which rides along as the system message.
        budget_chars = (
            usage._tracker.context_limit
            - self._COMPACT_SUMMARY_RESERVE
            - self._API_ROUND_RESERVE
            - len(prompts.COMPACT_PROMPT)
        ) * 2

        response = None
        attempt = 0
        extra_chars = 0
        dropped = 0
        marker_needed = False
        kept: list[ApiRound] = []
        formatted = ""
        sent_chars = 0

        with _sub_agent_scope("compact"):
            while True:
                kept, dropped, marker_needed = _plan_kept_groups(
                    groups, budget_chars, extra_chars_to_shed=extra_chars
                )

                messages_to_send: list[BaseMessage] = []
                if marker_needed:
                    messages_to_send.append(HumanMessage(content=MARKER_TEXT))
                for g in kept:
                    messages_to_send.extend(g.messages)

                formatted = _format_history_for_summary(messages_to_send)
                if custom_instructions:
                    formatted += f"\n\n## Compact Instructions\n{custom_instructions}"
                sent_chars = len(formatted)

                try:
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
                    break
                except Exception as e:  # noqa: BLE001
                    if not _is_context_exceeded(e):
                        raise
                    attempt += 1
                    if attempt >= self._MAX_COMPACT_ATTEMPTS:
                        raise
                    # Single-round overflow — more drops can't help (H2).
                    if len(kept) <= 1:
                        raise
                    extra_chars += _chars_to_shed_on_retry(
                        str(e), usage._tracker.context_limit
                    )

        summary = _extract_summary(response.content)
        body = _build_compact_body(summary, auto=auto)
        if task_state := tasks._tasks.state_summary():
            body += f"\n\n---\n\nActive tasks at time of compaction:\n{task_state}"

        # Reset tracker AFTER the compact call succeeds. If it had raised,
        # the tracker keeps its last good record so the UI doesn't flip
        # to "0 / 128k — no LLM calls yet" mid-session.
        usage._tracker.reset()

        self.store.clear_layer_1()
        if system_msg is not None:
            # Direct insert: consumers already saw this message at boot.
            # Re-dispatching would duplicate JSONL + re-render in UI.
            self.store._messages.insert(0, system_msg)

        await self._dispatch(
            CompactBoundaryMessage(
                pre_count=pre_count,
                auto=auto,
                dropped_rounds=dropped,
                retained_rounds=len(kept),
                marker_used=marker_needed,
                attempts=attempt + 1,
                original_chars=original_chars,
                sent_chars=sent_chars,
                source="compact",
            )
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
        Uses the shared _estimate_message_chars so tool_call args (which
        don't live in `content`) are counted; missing them lets `Edit`-style
        calls with large `new_string` args slip past this check.
        """
        estimated = sum(_estimate_message_chars(m) for m in messages) // 2
        headroom = self._COMPACT_SUMMARY_RESERVE + self._API_ROUND_RESERVE + self._COMPACT_CALL_RESERVE
        return estimated + headroom > usage._tracker.context_limit

    def current_context_tokens(self, parent_id: str | None = None) -> int:
        """Tokens currently held in history — what the next LLM call will send.

        Two signals, whichever is larger:
          1. `last_input + last_output` — API-accurate baseline right after a
             call completes. Stale as soon as any new content lands in the
             store (tool_result, next user message, compact summary).
          2. char-based estimate of the current api_view — catches content
             added since the last LLM response, which signal 1 is blind to.

        Context occupancy should never under-report. Taking the max keeps the
        footer honest when the real value lies between the two.
        """
        messages = self.store.api_view(parent_id=parent_id)
        char_est = sum(_estimate_message_chars(m) for m in messages) // 2
        if usage._tracker._records:
            r = usage._tracker._records[-1]
            return max(r.input + r.output, char_est)
        return char_est


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


# ---------------------------------------------------------------------------
# API-round trimming for compact input
# ---------------------------------------------------------------------------
#
# compact() summarises the conversation by sending a large HumanMessage to
# the LLM. When history is big enough to need compacting, the full formatted
# string can itself exceed the context window. These helpers split the
# history into "API rounds" (an AIMessage + its tool_results + trailing
# user follow-ups), trim from the head until the input fits the budget, and
# prefix a MARKER when the original first user turn has been dropped.
#
# Invariants enforced by _plan_kept_groups:
#   H1 — drop at least 1 round when len(groups) >= 2.
#   H2 — keep at least 1 round always.


MARKER_TEXT = (
    "[System-inserted MARKER — not from the real user. "
    "Earlier parts of this conversation were truncated because the history "
    "exceeded the compact window. The first real user/assistant turns below "
    "are not the original start of the conversation; treat them as the new "
    "starting point for your summary.]"
)


def _estimate_message_chars(m: BaseMessage) -> int:
    """Char-count estimate matching what _format_history_for_summary renders.

    Includes tool_call args (which don't live in `content`) because in
    agentic workflows they're often the single largest payload — missing
    them makes budget checks optimistic by orders of magnitude.
    """
    import json as _json

    total = len(str(getattr(m, "content", "")))
    for tc in (getattr(m, "tool_calls", None) or []):
        total += len(tc.get("name", ""))
        total += len(_json.dumps(tc.get("args", {}), ensure_ascii=False))
    # Per-message formatting prefix ("Human: ", "Assistant: ", "  → ")
    # — 16 chars is a generous upper bound covering all cases.
    total += 16
    return total


class ApiRound(BaseModel):
    """One logical turn in the conversation.

    An AIMessage plus the ToolMessages and any HumanMessages that follow
    it, up to (but not including) the next AIMessage. The first group —
    everything before any AIMessage — is the "preamble" and holds the
    original opening user message(s).

    Why this shape: compact drops whole rounds rather than individual
    messages, so orphan ToolMessages (without the AIMessage whose
    tool_calls they answer) can never appear after trimming.
    """

    messages: list[BaseMessage]
    is_preamble: bool

    model_config = {"arbitrary_types_allowed": True}

    @property
    def size_chars(self) -> int:
        return sum(_estimate_message_chars(m) for m in self.messages)


def _group_by_api_round(messages: list[BaseMessage]) -> list[ApiRound]:
    """Split a LangChain message list into API rounds.

    api_view() already merges consecutive AssistantMessages by turn_id,
    so each AIMessage in the input represents exactly one assistant
    round. That makes the boundary trivial: any AIMessage after a
    non-empty `current` buffer flushes the buffer as a round.
    """
    groups: list[ApiRound] = []
    current: list[BaseMessage] = []

    def _flush() -> None:
        if not current:
            return
        is_preamble = not any(isinstance(x, AIMessage) for x in current)
        groups.append(ApiRound(messages=list(current), is_preamble=is_preamble))
        current.clear()

    for m in messages:
        if isinstance(m, AIMessage) and current:
            _flush()
        current.append(m)
    _flush()
    return groups


def _plan_kept_groups(
    groups: list[ApiRound],
    budget_chars: int,
    extra_chars_to_shed: int = 0,
) -> tuple[list[ApiRound], int, bool]:
    """Decide which API rounds survive into the compact input.

    Hard invariants (product requirements):
      H1 — drop at least 1 round when len(groups) >= 2. compact is a
           reduction operation; dropping 0 defeats the purpose.
      H2 — keep at least 1 round always. With 0 rounds there is nothing
           to summarise; the session would silently lose all history.

    On retry, the caller passes `extra_chars_to_shed` > 0 to force more
    drops. The value is treated as a tightening of the effective budget.

    Returns:
        (kept, dropped_count, marker_needed)
        `marker_needed` is True iff at least one round was dropped AND
        the original groups[0] (the preamble) is no longer in `kept`.
    """
    if not groups:
        return ([], 0, False)

    marker_budget = len(MARKER_TEXT) + 32
    effective_budget = budget_chars - extra_chars_to_shed - marker_budget

    kept = list(groups)
    dropped = 0

    # H1: drop at least 1 when we have room (single-group case falls to H2).
    if len(kept) > 1:
        kept.pop(0)
        dropped += 1

    # Keep dropping from the head until we fit, or only 1 round remains (H2).
    while len(kept) > 1 and sum(g.size_chars for g in kept) > effective_budget:
        kept.pop(0)
        dropped += 1

    marker_needed = dropped > 0 and (not kept or kept[0] is not groups[0])
    return (kept, dropped, marker_needed)


_GAP_PATTERN = re.compile(
    r"maximum context length is (\d+) tokens.*?requested (\d+) tokens",
    re.DOTALL,
)


def _parse_context_gap_tokens(error_msg: str) -> int | None:
    """Parse a DeepSeek/OpenAI-compat context-exceeded error for the gap.

    Returns (requested - limit) in tokens when the message matches the
    expected pattern, else None. Anthropic and other providers use a
    different format — the caller should fall back to a percentage-based
    heuristic for those.
    """
    m = _GAP_PATTERN.search(error_msg)
    if not m:
        return None
    limit, requested = int(m.group(1)), int(m.group(2))
    return max(0, requested - limit)


# Tokens of safety margin added on top of the parsed gap on each retry.
# Large enough to cover estimator drift; small enough to not overshoot.
_RETRY_SAFETY_MARGIN_TOKENS = 5_000
# When the error format is unknown, shed this fraction of the context
# window per retry. Two retries shed ~40% combined — enough for any
# model even when we can't read its error message.
_RETRY_FALLBACK_FRACTION = 0.20


def _chars_to_shed_on_retry(error_msg: str, context_limit_tokens: int) -> int:
    """How many additional characters to drop before the next compact try.

    If the error message carries a parseable gap (DeepSeek path), target
    `gap + safety_margin` tokens. Otherwise fall back to a fixed fraction
    of the context window. Result is returned in chars (x2 the tokens)
    so it plugs directly into the char-based budget.
    """
    gap = _parse_context_gap_tokens(error_msg)
    if gap is not None:
        target_tokens = gap + _RETRY_SAFETY_MARGIN_TOKENS
    else:
        target_tokens = int(context_limit_tokens * _RETRY_FALLBACK_FRACTION)
    return target_tokens * 2
