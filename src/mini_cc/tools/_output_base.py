"""ToolOutput base class — extracted to break the import cycle.

Lives in its own module so:
  - output_types.py can `from _output_base import ToolOutput` without
    needing base.py (which depends on MiniTool, which depends on
    ToolOutput — a partial cycle).
  - base.py re-exports ToolOutput so existing
    `from mini_cc.tools.base import ToolOutput` keeps working.

Underscore prefix marks this as internal: callers should always import
ToolOutput from `mini_cc.tools.base` (or `mini_cc.tools.output_types`).
"""
from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel


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
