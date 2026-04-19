"""MiniTool base framework: structured Output types, render protocol, registry."""
import inspect
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError
from langchain_core.tools import StructuredTool


# ---------------------------------------------------------------------------
# Output base + concrete types
# ---------------------------------------------------------------------------

class ToolOutput(BaseModel):
    is_error: bool = False

    def to_api_str(self) -> str:
        return self.model_dump_json()


class ToolErrorOutput(ToolOutput):
    is_error: bool = True
    message: str

    def to_api_str(self) -> str:
        return self.message


class CommandOutput(ToolOutput):
    stdout: str
    returncode: int

    def to_api_str(self) -> str:
        return self.stdout if self.stdout else "(no output)"


class FileWriteOutput(ToolOutput):
    path: str
    bytes_written: int

    def to_api_str(self) -> str:
        return f"Written {self.bytes_written} bytes to {self.path}"


class FileEditOutput(ToolOutput):
    path: str
    replaced: bool

    def to_api_str(self) -> str:
        if self.replaced:
            return f"Edited {self.path}"
        return f"Error: old_string not found in {self.path}"


class TodoPlanOutput(ToolOutput):
    count: int
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class TodoUpdateOutput(ToolOutput):
    item: str
    status: str
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class TaskPlanOutput(ToolOutput):
    count: int
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class TaskUpdateOutput(ToolOutput):
    task_id: str
    status: str
    rendered: str

    def to_api_str(self) -> str:
        return self.rendered


class RunSkillOutput(ToolOutput):
    skill_name: str
    result: str

    def to_api_str(self) -> str:
        return self.result


class SubTaskOutput(ToolOutput):
    result: str

    def to_api_str(self) -> str:
        return self.result


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
        return ToolErrorOutput(message=f"Error: {error}")

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
