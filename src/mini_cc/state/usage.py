"""Token accounting for LLM calls and projected next-call input.

UsageTracker is the single source of truth for three questions:
  1. What did the API bill us for? (per-call records + session totals)
  2. How much is in context right now? (i.e. what will the next API call
     send — includes already-billed history + anything appended since)
  3. Is there enough headroom for one more round + compact?

The "right now" number is maintained incrementally via note_message /
note_content_cleared, called by engine._dispatch and _clear_old_tool_results
respectively. That keeps context_tokens_used() O(1) and lets callers
(`/context`, TUI status bar, _should_auto_compact) read a single authority
without passing messages around.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel
from rich.table import Table
from rich.console import Console

if TYPE_CHECKING:
    from mini_cc.engine.messages import Message


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
        # Running delta: chars appended to api_view since the last
        # AssistantMessage (i.e. since the last API response). Updated by
        # note_message on every dispatch. Used by context_tokens_used() to
        # answer in O(1) without walking the store.
        self._chars_since_last_ai: int = 0
        # Chars removed in-place by _clear_old_tool_results since the last
        # record(). Those chars are still counted in last.input (API saw
        # them), but won't be in the next call's input, so we subtract them
        # from the baseline.
        self._cleared_chars_since_record: int = 0

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
        # A fresh record means last.input + last.output is the new baseline;
        # anything the delta was tracking has been folded into last.output
        # (the LLM's response) by definition.
        self._chars_since_last_ai = 0
        self._cleared_chars_since_record = 0
        if response_metadata and (model := response_metadata.get("model_name")):
            self._model = model
            if model in self.MODEL_LIMITS:
                self._context_limit = self.MODEL_LIMITS[model]

    def note_message(self, msg: "Message") -> None:
        """Hook called by engine._dispatch for every stored message.

        Maintains _chars_since_last_ai so context_tokens_used() stays
        accurate between record() calls. Four cases:

        - AssistantMessage → reset delta to 0. The response that this
          represents is the API-billed last.output; subsequent messages
          accumulate fresh.
        - SystemPromptMessage / UserMessage / ToolResultMessage → append
          to delta. These are Layer-1, part of the next API input.
        - Layer-2 (StatusMessage / CompactBoundaryMessage) or unknown →
          ignore; they are not sent to the API.
        """
        # Lazy import: mini_cc.engine.messages pulls langchain + pydantic
        # and creates a cycle at module load time.
        from mini_cc.engine.messages import (
            AssistantMessage,
            SystemPromptMessage,
            ToolResultMessage,
            UserMessage,
        )
        if isinstance(msg, AssistantMessage):
            self._chars_since_last_ai = 0
        elif isinstance(msg, (SystemPromptMessage, UserMessage, ToolResultMessage)):
            self._chars_since_last_ai += estimate_chars(msg)

    def note_content_cleared(self, old_chars: int, new_chars: int) -> None:
        """Hook for _clear_old_tool_results' in-place content mutation.

        Without this, old tool_result chars remain counted in last.input
        even though the next API call won't include them — causing
        context_tokens_used() to over-report (and compact to fire
        unnecessarily) right after a clear.
        """
        dropped = max(0, old_chars - new_chars)
        self._cleared_chars_since_record += dropped

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

    def context_tokens_used(self) -> int:
        """Tokens currently in context = what the next API call will send.

        Pre-first-record (boot, post-compact-reset): delta alone, since
        nothing has been billed yet. Post-first-record: billed baseline
        (last.input + last.output) minus any cleared chars, plus the
        running delta. Clamped at 0 in case clearing outpaces baseline
        due to estimate drift.
        """
        if not self._records:
            return self._chars_since_last_ai // 2
        last = self._records[-1]
        baseline = last.input + last.output - self._cleared_chars_since_record // 2
        return max(0, baseline) + self._chars_since_last_ai // 2

    def headroom_left(self) -> int:
        return self._context_limit - self.context_tokens_used()

    def input_tokens_used(self) -> int:
        return self._total_in

    def output_tokens_used(self) -> int:
        return self._total_out + self._streaming_out

    def reset(self):
        """Clear per-call records after a compact. Preserves session totals."""
        self._records.clear()
        self._chars_since_last_ai = 0
        self._cleared_chars_since_record = 0

    def set_limit(self, limit: int):
        self._context_limit = limit

    def summary(
        self,
        history_len: int,
        console: Console | None = None,
    ) -> None:
        """Render /context output using rich.

        Context % reflects CURRENT occupancy (what the next call will send)
        via context_tokens_used() — same number the TUI status bar reads.
        """
        c = console or _console
        ctx_used = self.context_tokens_used()
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
