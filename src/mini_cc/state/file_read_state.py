"""File read state cache shared by file_read / file_write / file_edit.

Records evidence that the LLM has seen a file's content as of a specific mtime,
so subsequent file_edit / file_write calls can gate against blind overwrites
and file_read can dedup repeat reads.

Access the singleton as `file_read_state._state` (module attribute), never via
`from file_read_state import _state` — it is reassigned by `_sub_agent_scope`
for sub-agent isolation.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Entry(BaseModel):
    """Snapshot of a file the LLM has seen, used as read-gate evidence."""

    content: str
    """CRLF-normalized content the LLM saw at read time. Stored even for partial
    reads so dedup can compare exact slices and Edit/Write can fall back to
    content equality when mtime advances spuriously (Windows cloud sync, etc.)."""

    mtime_ms: int
    """int(os.stat(path).st_mtime * 1000), aligned with CC's `readFileState.timestamp`."""

    offset: Optional[int]
    """1-based start line if the entry came from a file_read; None when the
    entry was self-recorded by file_edit / file_write after writing — the
    sentinel `offset is None` means "entry covers full current disk content."
    """

    limit: Optional[int]
    """Line count limit at read time; None when entry came from Edit/Write
    (full content is implied) or when the read had no limit set."""


class FileReadState:
    """Per-agent file read evidence store. In-memory only, not persisted."""

    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}

    def get(self, resolved_path: str) -> Entry | None:
        return self._entries.get(resolved_path)

    def record(
        self,
        resolved_path: str,
        content: str,
        mtime_ms: int,
        offset: int | None,
        limit: int | None,
    ) -> None:
        self._entries[resolved_path] = Entry(
            content=content, mtime_ms=mtime_ms, offset=offset, limit=limit,
        )

    def clear(self) -> None:
        self._entries.clear()

    def is_consistent(
        self,
        entry: Entry,
        current_mtime_ms: int,
        current_content: str,
    ) -> bool:
        """Whether `entry` still reflects current disk state.

        Strategy:
          1. mtime equal → consistent (cheap path, common case).
          2. mtime differ + entry covers full current content + content equal
             → consistent (Windows mtime jitter recovery).
          3. else → not consistent (caller should error with re-read instruction).

        On entry "covers full content": Edit/Write self-records use
        `offset=None, limit=None` as the sentinel (`_covers_full_content`
        treats this as full). Reads with `offset=1` and a limit ≥ total lines
        also qualify. Partial reads (offset>1 or limit<total) cannot use the
        content fallback because their `entry.content` only contains the slice.
        """
        if current_mtime_ms == entry.mtime_ms:
            return True
        if not _covers_full_content(entry, current_content):
            return False
        return current_content == entry.content


def _covers_full_content(entry: Entry, current_content: str) -> bool:
    # Edit/Write self-record sentinel
    if entry.offset is None and entry.limit is None:
        return True
    # Read at offset=1 with no limit, or limit covering all current lines
    if entry.offset != 1:
        return False
    if entry.limit is None:
        return True
    # Use splitlines so trailing-\n files don't get counted as N+1 lines
    # (and empty files count as 0 lines, not 1).
    return entry.limit >= len(current_content.splitlines())


# NOTE: _state is reassigned (not just mutated) by `_sub_agent_scope`
# for sub-agent isolation. `from file_read_state import _state` would capture
# a stale reference — always access as `file_read_state._state` (module lookup).
_state = FileReadState()
