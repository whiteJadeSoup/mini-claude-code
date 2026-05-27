"""Truncate MEMORY.md index content to line + byte caps.

Why two caps (mirrors CC memdir.ts:truncateEntrypointContent):
  - line cap (200) bounds a normal index — one entry per line.
  - byte cap (25KB) catches the pathological case the line cap misses:
    a few very long lines. CC observed a 197KB index that was still under
    200 lines. Without the byte cap that whole 197KB would hit every
    request as cache-prefix tokens.
Line-truncate first (a natural boundary), THEN byte-truncate at the last
newline before the cap, so a line is never cut mid-way — EXCEPT when the
content has no newline within the byte window (a single huge line), where
it is cut at the byte boundary as a best effort.

The byte check uses the ORIGINAL byte count, not the line-truncated size.
A 250-line × 120-char index reports "250 lines and 30249 bytes" even though
line-truncation alone already brought the loaded content under 25KB. This is
intentional and CC-faithful (memdir.ts:64): the warning describes the SOURCE
file's size so the user understands why it was truncated and that the index
needs trimming — it is not a claim about the size of what was loaded.
"""
from __future__ import annotations

MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000


def truncate_entrypoint(raw: str) -> str:
    trimmed = raw.strip()
    lines = trimmed.split("\n")
    line_count = len(lines)
    # Measure the ORIGINAL byte count: long lines are exactly the failure
    # mode the byte cap targets, so measuring after line-truncation would
    # understate the size and suppress the warning.
    byte_count = len(trimmed.encode("utf-8"))

    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES

    if not was_line_truncated and not was_byte_truncated:
        return trimmed

    truncated = (
        "\n".join(lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else trimmed
    )

    if len(truncated.encode("utf-8")) > MAX_ENTRYPOINT_BYTES:
        # Slice on bytes, then back up to the last newline so we don't split
        # a UTF-8 line; decode with errors="ignore" to drop any partial
        # multibyte char left at the cut.
        head = truncated.encode("utf-8")[:MAX_ENTRYPOINT_BYTES]
        cut = head.rfind(b"\n")
        # `cut > 0` (not `!= -1`): no newline in the window (single huge line)
        # → cut is -1 → keep the whole head (best-effort byte cut). cut == 0
        # can't occur (trimmed never starts with \n), and the > 0 guard avoids a
        # head[:0]="" wipe if that invariant ever breaks.
        head = head[:cut] if cut > 0 else head
        truncated = head.decode("utf-8", errors="ignore")

    if was_byte_truncated and not was_line_truncated:
        reason = f"{byte_count} bytes (limit: {MAX_ENTRYPOINT_BYTES}) — index entries are too long"
    elif was_line_truncated and not was_byte_truncated:
        reason = f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})"
    else:
        reason = f"{line_count} lines and {byte_count} bytes"

    return (
        truncated
        # Blank line (\n\n) before the warning — mirrors CC memdir.ts:97 and keeps
        # the warning a standalone block. Tests partition on the full "\n\n> WARNING"
        # so the separator isn't miscounted as a content line.
        + f"\n\n> WARNING: MEMORY.md is {reason}. Only part of it was loaded. "
        "Keep index entries to one line under ~200 chars; move detail into topic files."
    )
