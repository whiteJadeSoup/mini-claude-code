"""file_read tool: cat -n style read with offset/limit, sandboxed, dedup'd.

Implements §D3 algorithm A from the design plan: stat predicate → dedup hit
check → read+CRLF normalize → token-budget gate → slice+truncate → format
with line numbers → record state for read-gate.
"""
import os

from mini_cc import config
from mini_cc.state import file_read_state
from mini_cc.tools.base import (
    FileReadOutput,
    MiniTool,
    ToolErrorOutput,
    ToolOutput,
    register,
)

from .prompt import PROMPT
from .render import render_complete, render_received

# Limits (G1 + G5)
DEFAULT_LIMIT = 2000
MAX_LINE_CHARS = 2000        # single-line truncation
MAX_FILE_BYTES = 256 * 1024  # G5 layer 1: stat-time pre-filter
MAX_FILE_CHARS = 100_000     # G5 layer 2: char-count proxy for ~25k tokens
LINE_TRUNCATED_SUFFIX = "...[line truncated]"


class FileReadTool(MiniTool):
    name = "file_read"
    description = "Read a file from the filesystem with cat -n line numbers"
    prompt = PROMPT

    async def _run(
        self,
        path: str,
        offset: int = 1,
        limit: int = DEFAULT_LIMIT,
    ) -> ToolOutput:
        # 1. Sandbox check
        try:
            p = config.safe_path(path)
        except ValueError:
            return ToolErrorOutput(message=(
                f"Path '{path}' is outside the working directory ({config.CWD}). "
                f"file_read can only access files inside this project. "
                f"To read system files, use execute_command "
                f"(e.g., execute_command('cat {path}')) instead."
            ))

        # 2. Stat
        try:
            st = os.stat(p)
        except FileNotFoundError:
            return ToolErrorOutput(message=(
                f"File not found at '{path}'. "
                f"Verify the path is correct relative to the working directory ({config.CWD}). "
                f"Use execute_command('ls <dir>') to list available files in the directory."
            ))

        # 3. Size pre-filter (G5 layer 1)
        if st.st_size > MAX_FILE_BYTES:
            return ToolErrorOutput(message=(
                f"File at '{path}' is {st.st_size}B (limit: {MAX_FILE_BYTES // 1024} KB). "
                f"Reading the whole file would exceed the token budget. "
                f"Use offset and limit to read in chunks — "
                f"e.g., file_read(path='{path}', offset=1, limit=500). "
                f"Run execute_command('wc -l {path}') first to see the total line count."
            ))

        # 4. Dedup check (G6) — only against entries that came from a prior Read.
        mtime_ms = int(st.st_mtime * 1000)
        entry = file_read_state._state.get(p)
        if (
            entry is not None
            and entry.offset is not None        # exclude Edit/Write self-records
            and entry.offset == offset
            and entry.limit == limit
            and entry.mtime_ms == mtime_ms
        ):
            return FileReadOutput(path=path, unchanged=True)

        # 5. Read with UTF-8 + CRLF normalize (OQ1) — non-UTF-8 → error.
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError:
            return ToolErrorOutput(message=(
                f"File at '{path}' is not UTF-8 text and cannot be read by file_read. "
                f"For binary files, use execute_command — "
                f"e.g., execute_command('file {path}') to identify the type, "
                f"or execute_command('xxd {path} | head') to view bytes."
            ))
        raw = raw.replace("\r\n", "\n")

        # 6. Char budget (G5 layer 2)
        if len(raw) > MAX_FILE_CHARS:
            return ToolErrorOutput(message=(
                f"File content at '{path}' is {len(raw)} characters "
                f"(limit: {MAX_FILE_CHARS} ≈ 25k tokens). "
                f"Reading the whole file would exceed the token budget. "
                f"Use offset and limit to read in chunks — "
                f"e.g., file_read(path='{path}', offset=1, limit=500). "
                f"Run execute_command('wc -l {path}') first to see the total line count."
            ))

        # 7. Slice
        lines = raw.splitlines()
        total = len(lines)
        start = max(0, offset - 1)
        end = min(total, start + limit)
        sliced = lines[start:end]

        # 8. Single-line truncation
        sliced = [
            (line[:MAX_LINE_CHARS] + LINE_TRUNCATED_SUFFIX) if len(line) > MAX_LINE_CHARS else line
            for line in sliced
        ]

        # 9. Format with right-aligned 6-wide line numbers + tab
        formatted = "\n".join(
            f"{idx + offset:>6}\t{line}" for idx, line in enumerate(sliced)
        )

        # 10. Record evidence for read-gate; raw (full file) is what we cache.
        file_read_state._state.record(p, raw, mtime_ms, offset, limit)

        # 11. Return
        return FileReadOutput(
            path=path,
            content=formatted,
            total_lines=total,
            start_line=offset,
            returned_lines=len(sliced),
            truncated_by_limit=(end < total),
        )

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(FileReadTool())
