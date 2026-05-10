"""Concrete ToolOutput subclasses + output_from_dict.

Extracted from base.py because every new tool grows the discriminated
union, but a tool author rarely wants to scan the whole MiniTool ABC at
the same time. base.py now keeps just the abstract types (ToolOutput
base, MiniTool, registry); concrete output shapes live here. base.py
re-exports everything in this module so existing
`from mini_cc.tools.base import FileReadOutput` imports keep working.
"""
from __future__ import annotations

from typing import Literal

from pydantic import model_validator

# ToolOutput base lives in base.py to avoid an import cycle (base.py → MiniTool
# uses ToolOutput, this file's subclasses use ToolOutput; concrete subclasses
# don't form the cycle, the base does).
from mini_cc.tools._output_base import ToolOutput


class ToolErrorOutput(ToolOutput):
    type: Literal["error"] = "error"
    is_error: bool = True
    message: str

    def to_api_str(self) -> str:
        return self.message


class CommandOutput(ToolOutput):
    type: Literal["command"] = "command"
    stdout: str
    returncode: int

    @model_validator(mode="after")
    def _derive_is_error(self) -> "CommandOutput":
        self.is_error = self.returncode != 0
        return self

    def to_api_str(self) -> str:
        return self.stdout if self.stdout else "(no output)"


class FileWriteOutput(ToolOutput):
    type: Literal["file_write"] = "file_write"
    path: str
    # Default keeps older JSONL records (pre-PR shape: {type, path, bytes_written})
    # deserializable after this change — same idiom as CompactBoundaryMessage's
    # audit fields in engine/messages.py:101-107. New code always sets it explicitly.
    operation: Literal["create", "update"] = "update"
    bytes_written: int
    # Rich data for UI v2+ — NOT exposed via to_api_str
    content: str = ""
    original_content: str | None = None       # None for create

    def to_api_str(self) -> str:
        # Aligned with CC FileWriteTool.ts:418-432
        if self.operation == "create":
            return f"File created successfully at: {self.path}"
        return f"The file {self.path} has been updated successfully."


class FileEditOutput(ToolOutput):
    type: Literal["file_edit"] = "file_edit"
    path: str
    replaced: bool
    replace_count: int = 0
    # Rich data for UI v2+ — NOT exposed via to_api_str
    old_string: str = ""
    new_string: str = ""
    original_content: str = ""

    @model_validator(mode="after")
    def _derive_is_error(self) -> "FileEditOutput":
        self.is_error = not self.replaced
        return self

    def to_api_str(self) -> str:
        # Aligned with CC FileEditTool.ts:581-593
        if not self.replaced:
            return f"Error: edit not applied to {self.path}"
        if self.replace_count > 1:
            return (f"The file {self.path} has been updated. "
                    f"All occurrences were successfully replaced.")
        return f"The file {self.path} has been updated successfully."


class FileReadOutput(ToolOutput):
    type: Literal["file_read"] = "file_read"
    path: str
    content: str = ""                # cat -n line-numbered text; empty when unchanged=True
    total_lines: int = 0
    start_line: int = 1              # 1-based starting line of this slice
    returned_lines: int = 0
    truncated_by_limit: bool = False
    unchanged: bool = False          # G6 dedup hit (path/offset/limit/mtime all unchanged)

    def to_api_str(self) -> str:
        # Plain text only — no <system-reminder> tag (CC uses it because its
        # system prompt teaches the LLM how to read the tag; mini-cc's prompts.py
        # has no such convention, so wrapping in the tag would invent a protocol.)
        #
        # The `unchanged` flag is now a UI-only annotation — the API path
        # always returns content. CC's `FILE_UNCHANGED_STUB` ("refer to the
        # earlier tool_result") relies on that earlier result still being
        # in context, but mini-cc's engine clears old tool_results on every
        # turn (see _clear_old_tool_results); the stub would point at
        # `[Cleared]`. Re-emitting content keeps the dedup tool-IO-cheap
        # (no disk re-read) without breaking the LLM's view.
        if self.total_lines == 0:
            return f"File is empty: {self.path}"
        if self.returned_lines == 0:
            return (f"File has {self.total_lines} lines; "
                    f"offset {self.start_line} is beyond the end.")
        return self.content


class GrepOutput(ToolOutput):
    type: Literal["grep"] = "grep"
    mode: Literal["content", "files_with_matches", "count"]
    num_files: int
    filenames: list[str] = []                # Empty in content/count modes
    content: str = ""                        # Set in content/count modes
    num_matches: int = 0                     # Set in count mode (sum across files)
    applied_limit: int | None = None         # Filled only when truncation kicked in
    applied_offset: int = 0

    def to_api_str(self) -> str:
        # Aligned with CC GrepTool.ts:254-308 mapToolResultToToolResultBlockParam
        if self.mode == "content":
            body = self.content or "No matches found"
            paging = self._paging_hint()
            return f"{body}\n\n{paging}" if paging else body
        if self.mode == "count":
            paging = self._paging_hint()
            files_word = "file" if self.num_files == 1 else "files"
            occ_word = "occurrence" if self.num_matches == 1 else "occurrences"
            summary = (
                f"\n\nFound {self.num_matches} total {occ_word} "
                f"across {self.num_files} {files_word}."
            )
            if paging:
                summary += f" with pagination = {paging}"
            return (self.content or "No matches found") + summary
        # files_with_matches
        if self.num_files == 0:
            return "No files found"
        paging = self._paging_hint()
        files_word = "file" if self.num_files == 1 else "files"
        header = f"Found {self.num_files} {files_word}"
        if paging:
            header += f" {paging}"
        return f"{header}\n" + "\n".join(self.filenames)

    def _paging_hint(self) -> str:
        parts = []
        if self.applied_limit is not None:
            parts.append(f"limit: {self.applied_limit}")
        if self.applied_offset:
            parts.append(f"offset: {self.applied_offset}")
        return ", ".join(parts)


class GlobOutput(ToolOutput):
    type: Literal["glob"] = "glob"
    filenames: list[str] = []        # CWD-relative paths
    num_files: int
    truncated: bool = False          # True when matches > GLOB_CAP
    duration_ms: int

    def to_api_str(self) -> str:
        # Aligned with CC GlobTool.ts:177-197
        if not self.filenames:
            return "No files found"
        body = "\n".join(self.filenames)
        if self.truncated:
            body += "\n(Results are truncated. Consider using a more specific path or pattern.)"
        return body


class TodoPlanOutput(ToolOutput):
    type: Literal["todo_plan"] = "todo_plan"
    count: int
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class TodoUpdateOutput(ToolOutput):
    type: Literal["todo_update"] = "todo_update"
    item: str
    status: str
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class TaskPlanOutput(ToolOutput):
    type: Literal["task_plan"] = "task_plan"
    count: int
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class TaskUpdateOutput(ToolOutput):
    type: Literal["task_update"] = "task_update"
    task_id: str
    status: str
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class RunSkillOutput(ToolOutput):
    type: Literal["run_skill"] = "run_skill"
    skill_name: str
    result: str

    def to_api_str(self) -> str:
        return self.result


class SubTaskOutput(ToolOutput):
    type: Literal["sub_task"] = "sub_task"
    result: str

    def to_api_str(self) -> str:
        return self.result


def output_from_dict(d: dict) -> ToolOutput:
    """Reconstruct the correct ToolOutput subclass from a serialized dict.

    Subclasses self-register via __init_subclass__ when the module is imported.
    Unknown or missing type keys fall back to the ToolOutput base class.
    """
    cls = ToolOutput._registry.get(d.get("type", ""), ToolOutput)
    return cls.model_validate(d)
