"""Memory freshness as human-readable age (port of CC memoryAge.ts + memoryHeader).

Models reason poorly about ISO timestamps but well about "47 days ago", which
triggers staleness reasoning. memory_header() is computed once at surfacing time
and frozen into SurfacedMemory.header so the rendered bytes stay stable across
turns (prompt-cache friendly).
"""
from __future__ import annotations

import time

_DAY_MS = 86_400_000


def _now_ms() -> int:
    return int(time.time() * 1000)


def memory_age_days(mtime_ms: int) -> int:
    return max(0, (_now_ms() - mtime_ms) // _DAY_MS)


def memory_age(mtime_ms: int) -> str:
    d = memory_age_days(mtime_ms)
    if d == 0:
        return "today"
    if d == 1:
        return "yesterday"
    return f"{d} days ago"


def memory_freshness_text(mtime_ms: int) -> str:
    """Staleness caveat for memories > 1 day old; '' for fresh ones (<=1 day —
    a warning there is just noise)."""
    d = memory_age_days(mtime_ms)
    if d <= 1:
        return ""
    return (
        f"This memory is {d} days old. Memories are point-in-time observations, "
        "not live state — claims about code behavior or file:line citations may "
        "be outdated. Verify against current code before asserting as fact."
    )


def memory_header(filename: str, mtime_ms: int) -> str:
    """Per-memory header line: stale → caveat + 'Memory: <name>:'; fresh →
    'Memory (saved today): <name>:'. Frozen into SurfacedMemory at creation."""
    staleness = memory_freshness_text(mtime_ms)
    if staleness:
        return f"{staleness}\n\nMemory: {filename}:"
    return f"Memory (saved {memory_age(mtime_ms)}): {filename}:"


def memory_freshness_note(mtime_ms: int) -> str:
    """Self-wrapped staleness note for callers that don't add their own
    <system-reminder> (e.g. file_read output). '' for memories <=1 day old.
    Mirrors CC memoryAge.ts:49 (memoryFreshnessNote)."""
    text = memory_freshness_text(mtime_ms)
    return f"<system-reminder>{text}</system-reminder>\n" if text else ""
