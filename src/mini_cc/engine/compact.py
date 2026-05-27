"""Compact subsystem — summarise history, drop oldest rounds, retry on overflow.

Extracted from query_engine.py because compact carries enough complexity
(state machine + budget planning + diagnostic logging) to deserve its own
module. The split keeps query_engine.py focused on the per-turn dispatch
loop; compact.py owns:

  - API-round grouping and trimming (ApiRound, _group_by_api_round,
    _plan_kept_groups, MARKER_TEXT)
  - Provider-specific overflow parsing (_parse_context_gap_tokens,
    _chars_to_shed_on_retry)
  - Summary formatting (_format_history_for_summary, _extract_summary,
    _build_compact_body)
  - The `run_compact` driver that owns the LLM call + retry loop

`QueryEngine.compact` is now a thin wrapper around `run_compact(self, ...)`.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel

from mini_cc import prompts
from mini_cc.engine.messages import (
    CompactBoundaryMessage,
    SystemPromptMessage,
    UserMessage,
)
from mini_cc.state import tasks, usage

if TYPE_CHECKING:
    from mini_cc.engine.query_engine import QueryEngine


# ---------------------------------------------------------------------------
# Reserves and retry budget
# ---------------------------------------------------------------------------
#
# Headroom kept out of the budget so compact() itself can succeed.
# Tweak in this file rather than the QueryEngine class — keeps related
# tuning knobs together.

COMPACT_SUMMARY_RESERVE = 20_000   # typical compact summary length
API_ROUND_RESERVE = 13_000         # one turn of thinking + response


# ---------------------------------------------------------------------------
# Marker for truncated preambles
# ---------------------------------------------------------------------------

MARKER_TEXT = (
    "[System-inserted MARKER — not from the real user. "
    "Earlier parts of this conversation were truncated because the history "
    "exceeded the compact window. The first real user/assistant turns below "
    "are not the original start of the conversation; treat them as the new "
    "starting point for your summary.]"
)


# ---------------------------------------------------------------------------
# API rounds — a logical grouping unit for trimming
# ---------------------------------------------------------------------------

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
        return sum(usage.estimate_chars(m) for m in self.messages)


def _group_by_api_round(messages: list[BaseMessage]) -> list[ApiRound]:
    """Split a LangChain message list into API rounds.

    api_view() already merges consecutive AssistantMessages by turn_id AND
    consecutive HumanMessages (方案 C boundary merge), so each AIMessage in
    the input represents exactly one assistant round, and the preamble holds
    a single merged Human turn (never multiple). That makes the boundary
    trivial: any AIMessage after a non-empty `current` buffer flushes it.
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


# ---------------------------------------------------------------------------
# Provider-specific overflow recovery
# ---------------------------------------------------------------------------

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


def _is_context_exceeded(e: Exception) -> bool:
    msg = str(e)
    return "maximum context length" in msg or "context_length_exceeded" in msg


# ---------------------------------------------------------------------------
# Summary formatting
# ---------------------------------------------------------------------------

def _format_history_for_summary(messages: list) -> str:
    import json as _json

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
    m = re.search(r"<summary>(.*?)</summary>", content, re.DOTALL)
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
# The driver
# ---------------------------------------------------------------------------

async def run_compact(
    engine: "QueryEngine",
    custom_instructions: str = "",
    auto: bool = False,
    trigger: str = "manual",
    max_attempts: int = 3,
) -> int:
    """Summarise, clear Layer 1, and re-seed with system + boundary + summary.

    Does NOT yield. Compact is a boundary event, not a conversational
    turn — its messages reach consumers via engine._dispatch directly.
    This matches the semantics of `/compact` (callers want a return
    value, not an iterator).

    Overflow handling: the compact call itself can exceed the context
    window when the history is large. We split the history into ApiRounds,
    drop the oldest until it fits, and retry up to `max_attempts` times,
    using the provider's error message to compute the next drop size
    (DeepSeek gap / fallback fraction).

    `trigger` labels the call in the diagnostic log: "pre_call" from
    _prepare_messages, "query_retry" from the query()-level retry,
    "manual" from the /compact slash command.

    Returns: number of store messages removed (pre_count - post_count).
    """
    from mini_cc.tools._utils import _sub_agent_scope
    from mini_cc.consumers import persistence
    from mini_cc.engine._diagnostics import log_event, tracker_snapshot

    # Snapshot BEFORE _sub_agent_scope swaps the tracker for the sub-agent's fresh one.
    tracker_before = tracker_snapshot()

    # Snapshot the system prompt so we can re-insert it after clear_layer_1.
    system_msg = next(
        (
            m
            for m in engine.store.all()
            if isinstance(m, SystemPromptMessage) and m.parent_id is None
        ),
        None,
    )
    pre_count = len(engine.store.all())

    all_msgs = engine.store.api_view(parent_id=None)
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
        - COMPACT_SUMMARY_RESERVE
        - API_ROUND_RESERVE
        - len(prompts.COMPACT_PROMPT)
    ) * 2

    response = None
    attempt = 0
    extra_chars = 0
    dropped = 0
    marker_needed = False
    kept: list[ApiRound] = []
    sent_chars = 0

    # Diagnostic accumulators — written to JSONL on any exit path.
    diag_attempts: list[dict] = []
    final_outcome = "success"
    final_error: str | None = None

    try:
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

                attempt_record: dict = {
                    "attempt": attempt,
                    "extra_chars": extra_chars,
                    "dropped_rounds": dropped,
                    "retained_rounds": len(kept),
                    "marker_used": marker_needed,
                    "sent_chars": sent_chars,
                }

                try:
                    response = await engine._llm_base.ainvoke(
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
                    attempt_record["outcome"] = "success"
                    um = response.usage_metadata or {}
                    attempt_record["response_input_tokens"] = um.get("input_tokens")
                    attempt_record["response_output_tokens"] = um.get("output_tokens")
                    diag_attempts.append(attempt_record)
                    break
                except Exception as e:  # noqa: BLE001
                    if not _is_context_exceeded(e):
                        attempt_record["outcome"] = "non_context_error"
                        attempt_record["error"] = str(e)
                        diag_attempts.append(attempt_record)
                        final_outcome = "non_context_error"
                        final_error = str(e)
                        raise
                    attempt += 1
                    if attempt >= max_attempts:
                        attempt_record["outcome"] = "context_exceeded"
                        attempt_record["error"] = str(e)
                        diag_attempts.append(attempt_record)
                        final_outcome = "context_exhausted"
                        final_error = str(e)
                        raise
                    # Single-round overflow — more drops can't help (H2).
                    if len(kept) <= 1:
                        diag_attempts.append(attempt_record)
                        final_outcome = "single_round_overflow"
                        final_error = str(e)
                        raise
                    gap = _parse_context_gap_tokens(str(e))
                    attempt_record["outcome"] = "context_exceeded"
                    attempt_record["error"] = str(e)
                    if gap is not None:
                        attempt_record["gap_tokens"] = gap
                    next_extra = _chars_to_shed_on_retry(
                        str(e), usage._tracker.context_limit
                    )
                    attempt_record["next_extra_chars"] = next_extra
                    diag_attempts.append(attempt_record)
                    extra_chars += next_extra

        raw_content = response.content
        if isinstance(raw_content, list):
            # Some models return a list of content blocks (e.g. [{type: text, text: ...}])
            # instead of a plain string. Concatenate all text blocks.
            raw_content = "".join(
                b.get("text", "") for b in raw_content if isinstance(b, dict)
            )
        summary = _extract_summary(raw_content)
        body = _build_compact_body(summary, auto=auto)
        if task_state := tasks._tasks.state_summary():
            body += f"\n\n---\n\nActive tasks at time of compaction:\n{task_state}"

        # Reset tracker AFTER the compact call succeeds. If it had raised,
        # the tracker keeps its last good record so the UI doesn't flip
        # to "0 / 128k — no LLM calls yet" mid-session.
        usage._tracker.reset()

        engine.store.clear_layer_1()

        # Compact erases the original file_read tool_results — neither
        # full content nor any dedup stub reference survives in the new
        # conversation prefix. The cached file_read_state must be reset
        # to match: post-compact file_read should return fresh content
        # (`unchanged=False`), not pretend the LLM still has access to
        # something that has been replaced by the compact summary.
        from mini_cc.state import file_read_state
        file_read_state._state.clear()

        if system_msg is not None:
            await engine._dispatch(system_msg)

        await engine._dispatch(
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
        await engine._dispatch(
            UserMessage(content=body, is_synthetic=True, source="compact")
        )
    finally:
        log_event(
            "compact",
            session_id=persistence.SESSION_ID,
            trigger=trigger,
            auto=auto,
            tracker_before=tracker_before,
            n_groups=len(groups),
            original_chars=original_chars,
            budget_chars=budget_chars,
            attempts=diag_attempts,
            final_outcome=final_outcome,
            final_error=final_error,
        )

    return max(0, pre_count - len(engine.store.all()))
