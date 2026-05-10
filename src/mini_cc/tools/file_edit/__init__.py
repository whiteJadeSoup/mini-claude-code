"""file_edit tool: targeted string replacement with read-gate, uniqueness, and atomic write.

Implements §D3 algorithm B from the design plan:
- Static rejects (no IO): old_string == new_string
- Read-gate: file must have been read first (entry in file_read_state)
- Sync critical section: stat → CRLF-normalize read → staleness check →
  uniqueness count → write → record
- Multi-match error includes the first 3 line numbers (OQ2)
- Self-record post-edit so chained edits don't see "modified" their own writes
"""
import os

from mini_cc import config
from mini_cc.state import file_read_state
from mini_cc.tools.base import (
    FileEditOutput,
    MiniTool,
    ToolErrorOutput,
    ToolOutput,
    register,
)

from .prompt import PROMPT
from .render import render_complete, render_received


def _find_match_linenos(content: str, needle: str, max_count: int = 3) -> list[int]:
    """Return 1-based line numbers of the first `max_count` non-overlapping matches.

    Matches what `content.count(needle)` counts (non-overlapping). Used in
    multi-match error to give the LLM concrete anchors for adding context.
    """
    if not needle:
        return []
    line_nums: list[int] = []
    pos = 0
    while len(line_nums) < max_count:
        idx = content.find(needle, pos)
        if idx == -1:
            break
        line_nums.append(content[:idx].count("\n") + 1)
        pos = idx + len(needle)
    return line_nums


class FileEditTool(MiniTool):
    name = "file_edit"
    description = "Make a targeted string replacement in an existing file"
    prompt = PROMPT

    async def _run(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolOutput:
        # 1. Static rejects (no IO)
        if old_string == new_string:
            return ToolErrorOutput(message=(
                "old_string and new_string are identical, so no edit would be applied. "
                "If you intended to change something, double-check both strings — "
                "likely the new_string is missing the change you wanted to make."
            ))
        # Empty old_string is dangerous: str.replace("", X) inserts X between
        # every character. Catch here before any IO so a buggy/confused caller
        # can't corrupt the file via the replace_all=True path.
        if not old_string:
            return ToolErrorOutput(message=(
                "old_string cannot be empty. file_edit needs a non-empty anchor "
                "string to locate where in the file the change should apply — "
                "an empty anchor would either insert content between every "
                "character or fail with no useful match. To insert content, "
                "pass a unique anchor + your insertion in new_string. To create "
                "a new file or fully overwrite content, use file_write."
            ))

        try:
            p = config.safe_path(path)
        except ValueError:
            return ToolErrorOutput(message=(
                f"Path '{path}' is outside the working directory ({config.CWD}). "
                f"file_edit can only modify files inside this project — editing "
                f"outside would break sandbox isolation."
            ))

        # 2. Read-gate first stage: must have an entry from a prior file_read
        entry = file_read_state._state.get(p)
        if entry is None:
            return ToolErrorOutput(message=(
                f"This file at '{path}' has not been read yet. "
                f"Editing without reading first risks overwriting recent changes "
                f"you haven't seen. Please call file_read('{path}') first, "
                f"then retry file_edit."
            ))

        # === sync critical section: stat → check → write → record ===

        # 3. Stat (file may have been deleted after read)
        try:
            st = os.stat(p)
        except FileNotFoundError:
            return ToolErrorOutput(message=(
                f"File not found at '{path}'. The file may have been deleted "
                f"after your last read; there is nothing to edit. Call "
                f"file_read('{path}') again to confirm — if it's truly gone, "
                f"use file_write to create it instead of file_edit."
            ))

        # 4. Read disk content (CRLF-normalize for consistent comparison)
        try:
            with open(p, "r", encoding="utf-8") as f:
                disk_content = f.read().replace("\r\n", "\n")
        except UnicodeDecodeError:
            return ToolErrorOutput(message=(
                f"File at '{path}' is not UTF-8 text and cannot be edited by file_edit. "
                f"Binary content has no meaningful textual representation here. "
                f"For binary files, use execute_command instead."
            ))

        current_mtime_ms = int(st.st_mtime * 1000)

        # 5. Staleness check (mtime + content fallback for full reads)
        if not file_read_state._state.is_consistent(
            entry, current_mtime_ms, disk_content
        ):
            return ToolErrorOutput(message=(
                f"This file at '{path}' has been modified since you last read it "
                f"(the disk content differs from your cached view). Editing now "
                f"would overwrite those external changes. Please call "
                f"file_read('{path}') again to see the current content, then "
                f"retry file_edit with the updated context."
            ))

        # 6. Uniqueness count
        n = disk_content.count(old_string)
        if n == 0:
            return ToolErrorOutput(message=(
                f"old_string was not found anywhere in '{path}'. Your edit "
                f"will not be applied. Likely causes: "
                f"(1) the string differs in whitespace or indentation; "
                f"(2) the file was edited since your last file_read. "
                f"Call file_read('{path}') again to verify the exact bytes, "
                f"then retry with the corrected old_string."
            ))
        if n > 1 and not replace_all:
            line_nums = _find_match_linenos(disk_content, old_string, max_count=3)
            line_hint = (
                f" First {len(line_nums)} matches are at lines {line_nums}."
                if line_nums else ""
            )
            return ToolErrorOutput(message=(
                f"Found {n} occurrences of old_string in '{path}', but replace_all "
                f"is false. Pick one of two paths:\n"
                f"  (1) set replace_all=true to replace all {n} occurrences at once;\n"
                f"  (2) extend old_string with surrounding context (lines above/below) "
                f"so it matches exactly one location.{line_hint}"
            ))

        # 7. Apply edit
        if replace_all:
            new_content = disk_content.replace(old_string, new_string)
            count = n
        else:
            new_content = disk_content.replace(old_string, new_string, 1)
            count = 1

        # 8. Write
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)

        # 9. Self-record (use new content & mtime; offset/limit None marks self-record)
        new_mtime_ms = int(os.stat(p).st_mtime * 1000)
        file_read_state._state.record(
            p, new_content, new_mtime_ms, offset=None, limit=None,
        )

        # === end critical section ===

        return FileEditOutput(
            path=path,
            replaced=True,
            replace_count=count,
            old_string=old_string,
            new_string=new_string,
            original_content=disk_content,
        )

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(FileEditTool())
