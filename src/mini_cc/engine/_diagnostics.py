"""Best-effort diagnostic logging for the context-exceeded failure paths.

When a DeepSeek-reasoner session hits `400 maximum context length`, the
engine's first job is to recover (auto-compact and retry). But from a
debugger's perspective, the interesting question is *why* — which call
overflowed, how close to the limit we were, how many rounds compact
dropped, and how the retries unfolded. The regular transcript answers
none of that.

This module emits structured JSONL events for compact-level and
astream-level context-exceeded incidents. Writes are append-only and
best-effort: any exception is silently swallowed, because diagnostics
must never break the agent flow. Inspect the file with `jq` / `cat`
when a failure happens.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def diag_path() -> Path:
    """Path of the current session's diagnostics JSONL file.

    Lives next to the transcript so debug sessions can correlate events
    with the conversation state at the same wall-clock moment.
    """
    from mini_cc.consumers import persistence

    t = persistence.transcript_path()
    return t.parent / f"{persistence.SESSION_ID}-diag.jsonl"


def log_event(event: str, **payload: Any) -> None:
    """Append one JSON line with the given event type and payload.

    All exceptions are swallowed. This function is on the error-handling
    path of the error-handling path — a failure here (no disk, bad
    permissions, JSON-encoding edge case on some exotic payload) should
    absolutely not mask the original problem.
    """
    try:
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        path = diag_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def tracker_snapshot() -> dict[str, Any]:
    """Pydantic-free snapshot of the global UsageTracker.

    Returns only the fields useful for diagnosing context overflow:
    the model name and context limit, the most recent call's token
    breakdown, and session totals. All zero when no LLM call has
    completed yet in this session.
    """
    from mini_cc.state import usage

    t = usage._tracker
    last = t._records[-1] if t._records else None
    return {
        "model": t._model,
        "context_limit": t._context_limit,
        "last_input": last.input if last else 0,
        "last_output": last.output if last else 0,
        "last_cache_read": last.cache_read if last else 0,
        "last_reasoning": last.reasoning if last else 0,
        "total_in": t._total_in,
        "total_out": t._total_out,
    }
