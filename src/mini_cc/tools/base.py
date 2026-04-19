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
    bytes_written: int

    def to_api_str(self) -> str:
        return f"Written {self.bytes_written} bytes to {self.path}"


class FileEditOutput(ToolOutput):
    type: Literal["file_edit"] = "file_edit"
    path: str
    replaced: bool

    @model_validator(mode="after")
    def _derive_is_error(self) -> "FileEditOutput":
        self.is_error = not self.replaced
        return self

    def to_api_str(self) -> str:
        if self.replaced:
            return f"Edited {self.path}"
        return f"Error: old_string not found in {self.path}"


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
