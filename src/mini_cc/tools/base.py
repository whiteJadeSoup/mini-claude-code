"""MiniTool base framework: structured Output types, render protocol, registry."""
import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ValidationError, model_validator
from langchain_core.tools import StructuredTool


# ---------------------------------------------------------------------------
# Output base + concrete types
# ---------------------------------------------------------------------------

class ToolOutput(BaseModel):
    # "base" is this class's own sentinel; __init_subclass__ skips it so the
    # base class is never auto-registered — output_from_dict falls back to it
    # for unknown/missing type keys, which is the correct behavior.
    type: str = "base"
    is_error: bool = False
    _registry: ClassVar[dict[str, Any]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kw: Any) -> None:
        # __pydantic_init_subclass__ fires after Pydantic's metaclass has fully
        # processed the subclass — cls.model_fields already reflects the override.
        # Plain __init_subclass__ fires too early: it sees the parent's type field.
        super().__pydantic_init_subclass__(**kw)
        field = cls.model_fields.get("type")
        if field is not None and field.default and field.default != "base":
            ToolOutput._registry[field.default] = cls

    def to_api_str(self) -> str:
        return self.model_dump_json()


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
        if self.unchanged:
            # Aligned with CC `FILE_UNCHANGED_STUB` (FileReadTool/prompt.ts:7-8)
            return ("File unchanged since last read. The content from the earlier "
                    "file_read tool_result in this conversation is still current — "
                    "refer to that instead of re-reading.")
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


# ---------------------------------------------------------------------------
# Oversized tool-result offload
# ---------------------------------------------------------------------------
#
# A single Read on a large file, or a Bash command with runaway output, can
# produce a tool result bigger than the headroom we reserve for a whole
# compact round. Sending it verbatim to the API wastes window on content
# the model rarely needs in full. Instead:
#
#   - Full content → disk (`persistence.tool_result_path(tool_call_id)`)
#     so the user can inspect it offline and the model can Read() it if
#     the preview turns out to be insufficient.
#   - API `content` → a short marker explaining what happened + the on-disk
#     path + the first PREVIEW_CHARS of the original, enough for the model
#     to recognise the output.
#   - UI `output` (the ToolOutput instance) is NOT touched — the TUI still
#     renders the full result for the user; this split is purely about what
#     crosses the API boundary.

TOOL_CONTENT_MAX_CHARS: int = 30_000     # trigger threshold (~15k tokens)
TOOL_CONTENT_PREVIEW_CHARS: int = 1_024  # kept in the API-facing content


def _sanitize_preview(text: str) -> str:
    """Strip control characters that can cause API 400 errors.

    Keeps printable ASCII/Unicode plus whitespace (tab, newline, carriage
    return). Replaces other C0/C1 control chars with the replacement char.
    These can appear in raw command output (ANSI escapes, terminal codes).
    """
    import re
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "\ufffd", text)


def truncate_tool_content(content: str, tool_call_id: str) -> str:
    """Offload oversized tool output; return API-safe preview."""
    if len(content) <= TOOL_CONTENT_MAX_CHARS:
        return content

    # Lazy import: base.py sits beneath consumers in the dep graph, and
    # persistence pulls mini_cc.engine.messages transitively. Importing at
    # call time keeps this module cheap to load.
    from mini_cc.consumers import persistence

    path = persistence.tool_result_path(tool_call_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        # If the disk write fails we'd rather return the first preview of
        # the original than hand the API a broken path. Caller sees a
        # preview with an explanatory note.
        preview = _sanitize_preview(content[:TOOL_CONTENT_PREVIEW_CHARS])
        return (
            f"[Tool result was {len(content):,} chars — failed to spill to "
            f"disk ({type(e).__name__}: {e}). Showing first "
            f"{TOOL_CONTENT_PREVIEW_CHARS:,} chars only.]\n\n{preview}"
        )

    preview = _sanitize_preview(content[:TOOL_CONTENT_PREVIEW_CHARS])
    # Spill path lives under ~/.minicc/projects/, OUTSIDE the project sandbox.
    # file_read would reject it via safe_path() — direct LLM at execute_command
    # which has no sandbox restriction.
    return (
        f"[Tool result too large ({len(content):,} chars) — truncated. "
        f"Full content saved to {path}. "
        f"Run execute_command('cat \"{path}\"') if you need more than the preview below.]\n\n"
        f"--- First {TOOL_CONTENT_PREVIEW_CHARS:,} chars ---\n"
        f"{preview}"
    )


# ---------------------------------------------------------------------------
# Internal helper — only for MiniTool default render impls, not for UI layer
# ---------------------------------------------------------------------------

def _fmt_args(args: dict, max_len: int = 60) -> str:
    if not args:
        return ""
    val = str(next(iter(args.values())))
    flat = " ".join(val.split())
    if len(args) > 1:
        flat = f"{flat}, …"
    return (flat[: max_len - 1] + "…") if len(flat) > max_len else flat


# ---------------------------------------------------------------------------
# MiniTool ABC
# ---------------------------------------------------------------------------

class MiniTool(ABC):
    name: str
    description: str    # human-facing label, shown in UI
    prompt: str         # model-facing: use-when / don't-use-for / examples
    concurrency: bool = False   # placeholder; always False

    @abstractmethod
    async def _run(self, **kwargs) -> ToolOutput:
        """Actual implementation; may raise exceptions."""
        ...

    # -- public entry point: never throws --

    async def execute(self, **kwargs) -> ToolOutput:
        try:
            return await self._run(**kwargs)
        except Exception as e:
            return self.handle_error(kwargs, e)

    def handle_error(self, args: dict, error: Exception) -> ToolOutput:
        return ToolErrorOutput(message=f"{type(error).__name__}: {error}")

    # -- render methods --

    def render_received(self, args: dict) -> str:
        return _fmt_args(args)

    def render_executing(self, args: dict) -> str:
        return self.render_received(args)

    def render_error(self, args: dict, output: ToolOutput) -> str:
        base = self.render_received(args)
        msg = output.message if isinstance(output, ToolErrorOutput) else "error"
        short = msg[:40] + "…" if len(msg) > 40 else msg
        return f"{base} · {short}" if base else short

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return self.render_received(args)

    # -- API bridge --

    def to_api_content(self, output: ToolOutput) -> str:
        return output.to_api_str()

    # -- LangChain bridge --

    def as_langchain_tool(self) -> StructuredTool:
        sig = inspect.signature(self._run)   # bound method; 'self' excluded
        instance = self

        async def _invoke(**kwargs: Any) -> ToolOutput:
            return await instance.execute(**kwargs)

        # StructuredTool.from_function reads __signature__ to build args_schema,
        # but pydantic's get_type_hints() reads __annotations__ — must set both.
        _invoke.__signature__ = sig
        _invoke.__name__ = self.name
        _invoke.__annotations__ = {
            name: p.annotation
            for name, p in sig.parameters.items()
            if p.annotation is not inspect.Parameter.empty
        }
        _invoke.__annotations__["return"] = ToolOutput

        return StructuredTool.from_function(
            coroutine=_invoke,
            name=self.name,
            description=self.prompt,
            handle_validation_error=self._fmt_validation_error,
        )

    def _fmt_validation_error(self, e: ValidationError) -> str:
        # Uses inspect.signature directly rather than building a new StructuredTool
        # instance, which would be wasteful on every validation-error call.
        sig_str = self.args_schema_description()
        fields = "; ".join(
            f"{err['loc'][0]}: {err['msg']}" for err in e.errors()
        )
        return f"Tool call error: {sig_str}\nValidation: {fields}"

    def args_schema_description(self) -> str:
        sig = inspect.signature(self._run)
        parts = []
        for pname, p in sig.parameters.items():
            ann = p.annotation
            type_str = getattr(ann, "__name__", None) or str(ann)
            if p.default is inspect.Parameter.empty:
                parts.append(f"{pname}: {type_str}")
            else:
                parts.append(f"{pname}: {type_str} = {p.default!r}")
        return f"{self.name}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, MiniTool] = {}


def register(tool: MiniTool) -> MiniTool:
    _REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> MiniTool | None:
    return _REGISTRY.get(name)
