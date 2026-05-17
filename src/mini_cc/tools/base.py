"""MiniTool framework — abstract base + registry + render protocol.

Concrete output types live in `output_types.py`; oversized-result spilling
lives in `truncate.py`. This file re-exports both for backward compat:
existing `from mini_cc.tools.base import FileReadOutput` calls keep working.
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from mini_cc.tools._output_base import ToolOutput
# Re-export concrete output types so external imports continue to work.
# noqa: F401 below is the contract: these names are re-exported, not unused.
from mini_cc.tools.output_types import (  # noqa: F401
    CommandOutput,
    FileEditOutput,
    FileReadOutput,
    FileWriteOutput,
    GlobOutput,
    GrepOutput,
    RunSkillOutput,
    SubTaskOutput,
    TaskPlanOutput,
    TaskUpdateOutput,
    TodoPlanOutput,
    TodoUpdateOutput,
    ToolErrorOutput,
    output_from_dict,
)
from mini_cc.tools.truncate import (  # noqa: F401
    TOOL_CONTENT_MAX_CHARS,
    TOOL_CONTENT_PREVIEW_CHARS,
    truncate_tool_content,
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
    # Whether this tool is purely read-only (no filesystem mutation, no subprocess
    # spawn, no external side effects). Consumed by sandbox policies that want to
    # express "read-only context" without hard-coding tool names — e.g. a memory
    # extraction sub-agent that allows arbitrary read tools but gates writes.
    is_read_only: bool = False

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
        """Convert pydantic ValidationError into actionable LLM-facing text.

        Goal: replace pydantic's "Field required" / "Input should be a valid
        integer" messages — which read like internal jargon — with the
        three-segment recovery pattern (state + impact + next-step) the rest
        of mini-cc's tool errors follow.

        Single-error path is given a tailored message per error type
        (`missing` / `*_type` / `extra_forbidden`); multi-error or unknown
        types fall back to a structured list. Every branch ends with the
        full signature + a concrete retry example so the LLM can self-correct
        on the next turn instead of falling back to execute_command.
        """
        errs = e.errors()
        sig_str = self.args_schema_description()
        retry_example = self._retry_example()

        if len(errs) == 1:
            err = errs[0]
            loc = err["loc"][0] if err.get("loc") else "<unknown>"
            etype = err.get("type", "")
            if etype == "missing":
                ftype = self._field_type_label(loc)
                return (
                    f"{self.name} missing required argument: `{loc}`"
                    f"{f' ({ftype})' if ftype else ''}. "
                    f"Retry with the argument included, e.g. {retry_example}.\n\n"
                    f"Full signature: {sig_str}"
                )
            if etype.endswith("_type") or etype.endswith("_parsing"):
                expected = self._field_type_label(loc) or "a different type"
                got = self._describe_input(err.get("input"))
                return (
                    f"{self.name} got wrong type for `{loc}`: expected {expected}"
                    f"{f', got {got}' if got else ''}. "
                    f"Retry with the correct type, e.g. {retry_example}.\n\n"
                    f"Full signature: {sig_str}"
                )
            if etype == "extra_forbidden":
                hint = self._did_you_mean(str(loc))
                return (
                    f"{self.name} received unknown argument: `{loc}`"
                    f"{hint}. "
                    f"Drop arguments not in the signature and retry, "
                    f"e.g. {retry_example}.\n\n"
                    f"Full signature: {sig_str}"
                )

        # Fallback: multi-error or unrecognised single-error type.
        fields = "\n".join(
            f"  · {err['loc'][0] if err.get('loc') else '<unknown>'}: {err['msg']}"
            for err in errs
        )
        return (
            f"{self.name} argument validation failed:\n{fields}\n\n"
            f"Full signature: {sig_str}\n"
            f"Retry the call with all required arguments and correct types, "
            f"e.g. {retry_example}."
        )

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

    def _field_type_label(self, name: str) -> str:
        """Human-readable type for a parameter (e.g. 'str', 'int')."""
        sig = inspect.signature(self._run)
        p = sig.parameters.get(str(name))
        if p is None or p.annotation is inspect.Parameter.empty:
            return ""
        ann = p.annotation
        return getattr(ann, "__name__", None) or str(ann)

    def _retry_example(self) -> str:
        """Build a short keyword-style retry example using sample values
        derived from each REQUIRED parameter's type annotation."""
        sig = inspect.signature(self._run)
        kv = []
        for pname, p in sig.parameters.items():
            if p.default is not inspect.Parameter.empty:
                continue   # optional — skip in example
            sample = self._sample_for_annotation(p.annotation)
            kv.append(f'{pname}={sample}')
        return f"{self.name}({', '.join(kv)})" if kv else f"{self.name}()"

    @staticmethod
    def _sample_for_annotation(ann: Any) -> str:
        """Inline a literal of the right shape so the LLM has a concrete
        template to copy. Defaults to '<value>' for unknown annotations."""
        type_name = getattr(ann, "__name__", None) or str(ann)
        return {
            "str": '"..."',
            "int": "1",
            "bool": "True",
            "float": "1.0",
            "list": "[]",
            "dict": "{}",
        }.get(type_name, "<value>")

    @staticmethod
    def _describe_input(v: Any) -> str:
        """Compact description of bad input for the error message."""
        if v is None:
            return ""
        type_name = type(v).__name__
        if isinstance(v, str):
            preview = v if len(v) <= 40 else v[:40] + "…"
            return f'{type_name} ({preview!r})'
        return type_name

    def _did_you_mean(self, name: str) -> str:
        """Suggest the closest valid arg name via shared-prefix matching.

        Same approach as grep's path typo helper — cheap, no dependencies,
        good enough for the common typo case ('paht' → 'path').
        """
        valid = [
            p for p in inspect.signature(self._run).parameters
            if p != "self"
        ]
        # 'shared prefix length' is the same primitive used in grep/__init__.py;
        # not imported to avoid coupling tools/base.py to grep/.
        def _shared_prefix(a: str, b: str) -> int:
            n = min(len(a), len(b))
            for i in range(n):
                if a[i].lower() != b[i].lower():
                    return i
            return n

        candidates = [
            (cand, _shared_prefix(name, cand))
            for cand in valid
            if _shared_prefix(name, cand) >= max(2, len(name) // 2)
        ]
        if not candidates:
            return ""
        candidates.sort(key=lambda x: -x[1])
        return f' (did you mean `{candidates[0][0]}`?)'


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, MiniTool] = {}


def register(tool: MiniTool) -> MiniTool:
    _REGISTRY[tool.name] = tool
    return tool


def get_tool(name: str) -> MiniTool | None:
    return _REGISTRY.get(name)
