"""file_write tool: create new file or overwrite an existing one with a read-gate.

Implements §D3 algorithm C from the design plan:
- Sandbox path resolution
- Read-gate: existing files must have been read first (entry in file_read_state)
- Sync critical section: stat → staleness check → write → record
- Self-record post-write so subsequent edit/write don't see "modified" their own writes
- New files: skip read-gate, just mkdir + write + record
"""
import os

from mini_cc import config
from mini_cc.state import file_read_state
from mini_cc.tools.base import (
    FileWriteOutput,
    MiniTool,
    ToolErrorOutput,
    ToolOutput,
    register,
)

from .prompt import PROMPT
from .render import render_complete, render_received


class FileWriteTool(MiniTool):
    name = "file_write"
    description = "Create a new file or fully overwrite an existing one"
    prompt = PROMPT

    async def _run(self, path: str, content: str) -> ToolOutput:
        # 1. Sandbox check
        try:
            p = config.safe_path(path)
        except ValueError:
            return ToolErrorOutput(message=(
                f"Path '{path}' is outside the working directory ({config.CWD}). "
                f"file_write can only create or overwrite files inside this project — "
                f"writing outside would break sandbox isolation. To create a file "
                f"elsewhere on the system, use execute_command instead."
            ))

        file_exists = os.path.exists(p)

        # 2. Read-gate (existing files only)
        entry = None
        if file_exists:
            entry = file_read_state._state.get(p)
            if entry is None:
                return ToolErrorOutput(message=(
                    f"This file already exists at '{path}' but has not been read yet. "
                    f"Overwriting without reading first risks losing the current content "
                    f"you haven't seen. Please call file_read('{path}') first to inspect "
                    f"the current content, then retry file_write."
                ))

        # === sync critical section: stat → check → write → record ===

        # 3. Staleness check (existing only). A delete-then-stat race is handled
        #    by demoting the path to "create".
        old_content: str | None = None
        if file_exists:
            try:
                st = os.stat(p)
            except FileNotFoundError:
                file_exists = False  # race: deleted between exists() and stat()
            else:
                with open(p, "r", encoding="utf-8") as f:
                    old_content = f.read().replace("\r\n", "\n")
                current_mtime_ms = int(st.st_mtime * 1000)
                if not file_read_state._state.is_consistent(
                    entry, current_mtime_ms, old_content
                ):
                    return ToolErrorOutput(message=(
                        f"This file at '{path}' has been modified since you last read it "
                        f"(the disk content differs from your cached view). Overwriting "
                        f"now would lose those external changes. Please call "
                        f"file_read('{path}') again to see the current content, then "
                        f"decide whether to overwrite."
                    ))

        # 4. mkdir parent (idempotent; harmless when parent already exists)
        parent = os.path.dirname(p) or "."
        os.makedirs(parent, exist_ok=True)

        # 5. Write
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(content)

        # 6. Self-record. Use CRLF-normalized content so future content-fallback
        #    comparisons match what file_read would observe (which CRLF-normalizes).
        normalized_content = content.replace("\r\n", "\n")
        new_mtime_ms = int(os.stat(p).st_mtime * 1000)
        file_read_state._state.record(
            p, normalized_content, new_mtime_ms, offset=None, limit=None,
        )

        # === end critical section ===

        return FileWriteOutput(
            path=path,
            operation="update" if file_exists else "create",
            bytes_written=len(content.encode("utf-8")),
            content=content,
            original_content=old_content,
        )

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(FileWriteTool())
