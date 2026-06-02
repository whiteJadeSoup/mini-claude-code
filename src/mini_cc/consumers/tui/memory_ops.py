"""Classify + summarize agent-initiated memory tool operations (surface ②).

See docs/superpowers/specs/2026-06-02-memory-op-display-design.md. Pure
functions + a pure accumulator (no widget state) so they unit-test directly,
mirroring CC's getSearchReadSummaryText and renderers._recalled_markup.
"""
from __future__ import annotations

from mini_cc import config
from mini_cc.memdir import is_memory_path

_READ_TOOLS = frozenset({"file_read"})
_WRITE_TOOLS = frozenset({"file_edit", "file_write"})
_SEARCH_TOOLS = frozenset({"grep", "glob"})


def classify_memory_op(name: str, args: dict) -> str | None:
    """Return "read" | "write" | "search" if this tool call targets a file
    inside the memory directory, else None.

    The path is resolved with config.safe_path — the same resolution the tools
    themselves apply — so classification matches what actually gets read/written.
    Paths outside cwd+memdir raise in safe_path and yield None. grep/glob with no
    explicit `path` search the project, not memory.
    """
    if name in _READ_TOOLS:
        kind = "read"
    elif name in _WRITE_TOOLS:
        kind = "write"
    elif name in _SEARCH_TOOLS:
        kind = "search"
    else:
        return None

    path = args.get("path")
    if not path:
        return None
    try:
        resolved = config.safe_path(path)
    except ValueError:
        return None
    return kind if is_memory_path(resolved) else None


def memory_run_summary(read: int, write: int, search: int, *, active: bool) -> str:
    """Collapsed line text for a memory run. Verb order read · search · write;
    the first part's verb is capitalized, the rest lowercased (CC parity). Read
    and write carry a count (singular/plural); search does not. Returns "" when
    all counts are zero.
    """
    parts: list[str] = []

    def verb(present: str, past: str) -> str:
        word = present if active else past
        # Callers pass title-case literals; lowering all but the first part
        # yields the CC rule "first verb capitalized, the rest lowercase".
        return word if not parts else word.lower()

    if read:
        noun = "memory" if read == 1 else "memories"
        parts.append(f"{verb('Recalling', 'Recalled')} {read} {noun}")
    if search:
        parts.append(f"{verb('Searching', 'Searched')} memories")
    if write:
        noun = "memory" if write == 1 else "memories"
        parts.append(f"{verb('Writing', 'Wrote')} {write} {noun}")

    return " · ".join(parts)


class MemoryRun:
    """Open memory-run accumulator (surface ②). Pure state machine — no widget —
    so the consecutive-grouping logic unit-tests without importing the TUI.
    ToolStatus owns one instance and renders its summaries as markup rows.

    Membership is decided at absorb()-time (emit order), NOT at completion time,
    so counts/grouping stay correct once tool execution becomes parallel; only
    flushed-row ordering would then depend on completion (a pre-existing TUI
    property shared by all tools).
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] | None = None

    def absorb(self, name: str, args: dict) -> bool:
        """Fold a memory op into the open run (opening one if needed) and return
        True. Return False for a non-memory op — the caller should flush() the
        run and handle the tool normally."""
        kind = classify_memory_op(name, args)
        if kind is None:
            return False
        if self._counts is None:
            self._counts = {"read": 0, "write": 0, "search": 0}
        self._counts[kind] += 1
        return True

    @property
    def is_open(self) -> bool:
        return self._counts is not None

    def live_summary(self) -> str | None:
        """Present-tense summary for the live row, or None if no run is open."""
        if self._counts is None:
            return None
        c = self._counts
        return memory_run_summary(c["read"], c["write"], c["search"], active=True)

    def flush(self) -> str | None:
        """Close the run and return the past-tense summary to persist, or None
        if no run was open."""
        if self._counts is None:
            return None
        c = self._counts
        self._counts = None
        return memory_run_summary(c["read"], c["write"], c["search"], active=False)
