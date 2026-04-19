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

    def context_tokens_used(self) -> int:
        return self._records[-1].input if self._records else 0

    def projected_next_input(self) -> int:
        """Lower-bound estimate of the next API call's input token count.

        Next input >= last_input + last_output, because the model's response
        is appended to history before the next call. Tool results on top of
        that are handled separately by the character-based estimate.
        DeepSeek's input_tokens already includes cache_read (cache is a subset
        of input, not additive).
        """
        if not self._records:
            return 0
        r = self._records[-1]
        return r.input + r.output

    def input_tokens_used(self) -> int:
        return self._total_in

    def output_tokens_used(self) -> int:
        return self._total_out + self._streaming_out

    def reset(self):
        """Clear per-call records after a compact. Preserves session totals."""
        self._records.clear()

    def set_limit(self, limit: int):
        self._context_limit = limit

    def summary(self, history_len: int, console: Console | None = None) -> None:
        """Render /context output using rich.

        Context % is based on the last call's input tokens — that's the
        actual context window consumption, since every call sends full history.
        Pass a custom console to capture output (e.g. StringIO-backed Console).
        """
        c = console or _console
        last_input = self.context_tokens_used()
        pct = last_input / self._context_limit * 100 if self._context_limit else 0
        color = "green" if pct < 50 else "yellow" if pct < 80 else "red"

        c.print(f"[bold]Model:[/]    {self._model or 'unknown'}")
        c.print(f"[bold]Context:[/]  [{color}]{last_input:,} / {self._context_limit:,} ({pct:.1f}%)[/]")
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
