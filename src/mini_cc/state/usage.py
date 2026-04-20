"""Token accounting for LLM calls and projected next-call input.

UsageTracker records what the API bills us for on every response. The
forward-looking question — "how many tokens will the NEXT call send?" —
is answered on demand by context_tokens_used(messages): the last
API-reported baseline (input + output) plus a char-based local estimate
of any Layer-1 messages dispatched after that response and not yet
rebaselined.

This mirrors the industry pattern (OpenCode / Kilo Code / Roo-Code /
Gemini CLI): trust the API for what it has seen; estimate only the
pending delta. No hooks threaded through _dispatch, no running state to
keep in sync with the store.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel
from rich.table import Table
from rich.console import Console


_console = Console()


class CallRecord(BaseModel):
    """One row in /context usage table.

    Each LLM call produces one record. Sub-agent records are collapsed into
    a single row (calls > 1) so the table stays readable regardless of how
    many internal round-trips a sub-agent made.
    """
    source: str
    input: int
    output: int
    cache_read: int = 0       # DeepSeek prefix-caching — tracks cost savings
    reasoning: int = 0        # reasoning model (deepseek-reasoner) thinking tokens
    calls: int = 1            # >1 when merge_sub collapses a sub-agent's records


def estimate_chars(msg) -> int:
    """Char-count estimate of a message's contribution to API input.

    Includes tool_use args (which don't live in `content`) — in agentic
    workflows they're often the largest single payload, and missing them
    makes budget checks optimistic by orders of magnitude.

    Accepts either our internal Message types (SystemPromptMessage / User /
    Assistant / ToolResult) or LangChain BaseMessage subclasses. Both expose
    a `.content` attr; only the langchain path carries tool_calls as a list
    of dicts, so we read that defensively.
    """
    content = getattr(msg, "content", "")
    # AssistantMessage wraps content in a ContentBlock (TextBlock or ToolUseBlock)
    # rather than a plain string. Peek at .text / serialize the args so we count
    # the bytes the API will actually see.
    if hasattr(content, "text"):
        total = len(content.text)
    elif hasattr(content, "args"):
        total = len(content.name) + len(json.dumps(content.args, ensure_ascii=False))
    else:
        total = len(str(content))

    for tc in (getattr(msg, "tool_calls", None) or []):
        total += len(tc.get("name", ""))
        total += len(json.dumps(tc.get("args", {}), ensure_ascii=False))

    # Per-message formatting overhead (role markers, JSON wrapping). 16 chars
    # is a generous upper bound covering "Human: ", "Assistant: ", "  → " etc.
    total += 16
    return total


class UsageTracker:
    """Tracks token usage across all LLM calls in a session."""

    MODEL_LIMITS = {
        "deepseek-chat": 131072,      # 128K
        "deepseek-reasoner": 131072,  # 128K
    }

    def __init__(self):
        self._records: list[CallRecord] = []
        self._model: str = ""
        self._context_limit: int = 131072
        self._total_in: int = 0
        self._total_out: int = 0
        self._total_cache: int = 0
        self._total_reasoning: int = 0
        self._streaming_out: int = 0  # chunk proxy during active streaming

    def count_stream_chunk(self) -> None:
        """Increment the live output-token proxy by one streaming chunk."""
        self._streaming_out += 1

    def record(self, source: str, usage_metadata: dict | None,
               response_metadata: dict | None = None):
        """Record one LLM call's token usage.

        response_metadata is optional because it's only needed to detect
        the model name (for auto-setting context limits).
        """
        if not usage_metadata:
            return
        rec = CallRecord(
            source=source,
            input=usage_metadata.get("input_tokens", 0),
            output=usage_metadata.get("output_tokens", 0),
            cache_read=(usage_metadata.get("input_token_details") or {}).get("cache_read", 0),
            reasoning=(usage_metadata.get("output_token_details") or {}).get("reasoning", 0),
        )
        self._records.append(rec)
        self._total_in += rec.input
        self._total_out += rec.output
        self._streaming_out = 0  # exact value now in _total_out
        self._total_cache += rec.cache_read
        self._total_reasoning += rec.reasoning
        if response_metadata and (model := response_metadata.get("model_name")):
            self._model = model
            if model in self.MODEL_LIMITS:
                self._context_limit = self.MODEL_LIMITS[model]

    def merge_sub(self, description: str, sub: "UsageTracker"):
        """Collapse a sub-agent's records into one summary row."""
        if not sub._records:
            return
        rec = CallRecord(
            source=f"sub-agent: {description}",
            input=sub._total_in, output=sub._total_out,
            cache_read=sub._total_cache, reasoning=sub._total_reasoning,
            calls=len(sub._records),
        )
        self._records.append(rec)
        self._total_in += rec.input
        self._total_out += rec.output
        self._total_cache += rec.cache_read
        self._total_reasoning += rec.reasoning

    @property
    def context_limit(self) -> int:
        return self._context_limit

    def context_tokens_used(self, messages: list[BaseMessage]) -> int:
        """Tokens currently in context = what the next API call will send.

        Pre-first-record (boot, post-compact-reset): full-history char
        estimate, since no API baseline exists yet.

        Post-first-record: last.input + last.output (API-billed baseline)
        plus char-estimate of messages appended AFTER the last AI response
        — these are the tool_results / new user turn that the most recent
        record() hasn't counted yet. They'll be folded into the next
        response's input_tokens; until then, we estimate locally.
        """
        if not self._records:
            return sum(estimate_chars(m) for m in messages) // 2
        last = self._records[-1]
        # Walk from the end backwards; stop at the most recent AIMessage.
        # That AI (and everything before it) is covered by last.input +
        # last.output.
        new_chars = 0
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                break
            new_chars += estimate_chars(m)
        return last.input + last.output + new_chars // 2

    def headroom_left(self, messages: list[BaseMessage]) -> int:
        return self._context_limit - self.context_tokens_used(messages)

    def input_tokens_used(self) -> int:
        return self._total_in

    def output_tokens_used(self) -> int:
        return self._total_out + self._streaming_out

    def reset(self):
        """Clear per-call records after a compact. Preserves session totals."""
        self._records.clear()

    def set_limit(self, limit: int):
        self._context_limit = limit

    def summary(
        self,
        history_len: int,
        console: Console | None = None,
        current_tokens: int | None = None,
    ) -> None:
        """Render /context output using rich.

        `current_tokens` is the occupancy number from
        `context_tokens_used(messages)`. Caller computes it (e.g. via
        engine.current_context_tokens()) and passes it in — this class
        doesn't hold a MessageStore reference, so it can't derive the
        number itself. If omitted, falls back to the last API input count
        so tests exercising summary() without an engine still render.
        """
        c = console or _console
        ctx_used = current_tokens if current_tokens is not None else (
            self._records[-1].input if self._records else 0
        )
        pct = ctx_used / self._context_limit * 100 if self._context_limit else 0
        color = "green" if pct < 50 else "yellow" if pct < 80 else "red"

        c.print(f"[bold]Model:[/]    {self._model or 'unknown'}")
        c.print(f"[bold]Context:[/]  [{color}]{ctx_used:,} / {self._context_limit:,} ({pct:.1f}%)[/]")
        c.print(f"[bold]History:[/]  {history_len} messages")

        if not self._records:
            c.print("(no LLM calls yet)")
            return

        table = Table(show_header=True, show_footer=True, show_edge=False, pad_edge=False)
        table.add_column("#", style="dim", footer_style="dim")
        table.add_column("Source", footer="Total")
        table.add_column("Input", justify="right", footer="-")
        table.add_column("Output", justify="right", footer=f"{self._total_out:,}")
        table.add_column("Cache", justify="right",
                         footer=f"{self._total_cache:,}" if self._total_cache else "-")
        table.add_column("Reasoning", justify="right",
                         footer=f"{self._total_reasoning:,}" if self._total_reasoning else "-")

        for i, r in enumerate(self._records, 1):
            suffix = f" ({r.calls} calls)" if r.calls > 1 else ""
            table.add_row(
                str(i), r.source + suffix,
                f"{r.input:,}", f"{r.output:,}",
                f"{r.cache_read:,}" if r.cache_read else "-",
                f"{r.reasoning:,}" if r.reasoning else "-",
            )
        c.print(table)


# NOTE: _tracker is swapped by task tool for sub-agent isolation.
# Always access as usage._tracker.
_tracker = UsageTracker()
