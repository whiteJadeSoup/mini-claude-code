"""Oversized tool-result offload.

A single Read on a large file, or a Bash command with runaway output, can
produce a tool result bigger than the headroom we reserve for a whole
compact round. Sending it verbatim to the API wastes window on content
the model rarely needs in full. Instead:

  - Full content → disk (`persistence.tool_result_path(tool_call_id)`)
    so the user can inspect it offline and the model can Read() it if
    the preview turns out to be insufficient.
  - API `content` → a short marker explaining what happened + the on-disk
    path + the first PREVIEW_CHARS of the original, enough for the model
    to recognise the output.
  - UI `output` (the ToolOutput instance) is NOT touched — the TUI still
    renders the full result for the user; this split is purely about what
    crosses the API boundary.
"""
from __future__ import annotations

import re

TOOL_CONTENT_MAX_CHARS: int = 30_000     # trigger threshold (~15k tokens)
TOOL_CONTENT_PREVIEW_CHARS: int = 1_024  # kept in the API-facing content


def _sanitize_preview(text: str) -> str:
    """Strip control characters that can cause API 400 errors.

    Keeps printable ASCII/Unicode plus whitespace (tab, newline, carriage
    return). Replaces other C0/C1 control chars with the replacement char.
    These can appear in raw command output (ANSI escapes, terminal codes).
    """
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "�", text)


def truncate_tool_content(content: str, tool_call_id: str) -> str:
    """Offload oversized tool output; return API-safe preview."""
    if len(content) <= TOOL_CONTENT_MAX_CHARS:
        return content

    # Lazy import: this module sits beneath consumers in the dep graph,
    # and persistence pulls mini_cc.engine.messages transitively. Importing
    # at call time keeps this module cheap to load.
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
