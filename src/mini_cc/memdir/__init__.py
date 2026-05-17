"""mini-cc memory directory module.

Public API surface for the "可存" (storable) layer:
  - :class:`MemoryType`, :class:`MemoryHeader`, :func:`parse_memory_type`
  - :func:`get_auto_mem_path`, :func:`get_auto_mem_entrypoint`,
    :func:`get_auto_mem_daily_log_path`, :func:`validate_memory_path`
  - :func:`scan_memory_files`, :func:`format_memory_manifest`

P1-1C wired everything into existence but no caller invokes these yet.
P2 (可取) will inject scan results into the system prompt; P3 (可写)
will add user-driven and forked-agent writes; P4 (可久) will add age,
freshness, and autoDream.
"""
from mini_cc.memdir.paths import (
    get_auto_mem_daily_log_path,
    get_auto_mem_entrypoint,
    get_auto_mem_path,
    validate_memory_path,
)
from mini_cc.memdir.scan import (
    FRONTMATTER_MAX_LINES,
    MAX_MEMORY_FILES,
    format_memory_manifest,
    scan_memory_files,
)
from mini_cc.memdir.types import (
    MemoryHeader,
    MemoryType,
    parse_memory_type,
)

__all__ = [
    "FRONTMATTER_MAX_LINES",
    "MAX_MEMORY_FILES",
    "MemoryHeader",
    "MemoryType",
    "format_memory_manifest",
    "get_auto_mem_daily_log_path",
    "get_auto_mem_entrypoint",
    "get_auto_mem_path",
    "parse_memory_type",
    "scan_memory_files",
    "validate_memory_path",
]
