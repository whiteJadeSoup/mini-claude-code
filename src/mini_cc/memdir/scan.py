"""Memory directory scanning + manifest formatting.

Reads ``*.md`` files under a memdir (recursively, so KAIROS-mode
``logs/YYYY/MM/`` daily logs are picked up automatically) and returns a
list of :class:`MemoryHeader` sorted newest-first, capped at
:data:`MAX_MEMORY_FILES`. Excludes the index file (``MEMORY.md``)
because it is structural metadata, not a memory.

Strict-write / lenient-read contract (严进宽出)
------------------------------------------------
The write side enforces frontmatter schema (P3 ``/remember`` validates;
the LLM is told via system-prompt instructions). The read side here is
**radically tolerant**:

  - File has no frontmatter at all → included, description=None, type=None
  - Frontmatter YAML is syntactically broken → included, both fields None
  - ``type:`` value is not a string or not in the closed set → type=None
  - ``description:`` value is missing, blank, or non-string → None
  - Only real IO errors (file deleted mid-scan, permission denied, fatal
    encoding failure) cause the file to be **skipped** (with a warning).

Why so tolerant: a memdir is built up by a mix of automated writes and
user hand-edits over months. A single corrupt frontmatter MUST NOT make
other memories disappear from the manifest. And dropping a file silently
("type=unknown so I'll skip it") is strictly worse than including it
with degraded fields — the selector can still see the filename, and the
user can debug a present-but-empty entry far more easily than absence.

Manifest format (CC parity, ``memoryScan.ts:84-94``)
-----------------------------------------------------
Each entry renders as a single Markdown list bullet:

  - [type] filename (ISO-timestamp): description   ← with type + description
  - [type] filename (ISO-timestamp)                 ← with type, no description
  -        filename (ISO-timestamp): description    ← no type (tag empty)
  -        filename (ISO-timestamp)                 ← neither

The ISO timestamp (not relative age like "3 days ago") is used because
the manifest's consumer is a **Sonnet selector** at query time — it
needs a precise time signal it can compare numerically. Relative-age
phrasing ("N days ago") lands in a different layer (P4's read-time
freshness hint, where the final answering LLM gets the prompt
nudging it to question old memories).

Limits
------
- :data:`MAX_MEMORY_FILES`: 200. This caps the manifest size handed to
  the Sonnet selector. Older files get dropped (mtime-tail truncation),
  which works as an implicit LRU because writes touch mtime.
- :data:`FRONTMATTER_MAX_LINES`: 30. Only the first 30 lines of each
  file are read; the rest of the body is ignored at scan time. This
  assumes ``---`` frontmatter is always at the very top — a write-side
  contract.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from mini_cc.memdir.types import MemoryHeader, parse_memory_type


MAX_MEMORY_FILES = 200
"""Cap on returned headers. Matches CC's ``MAX_MEMORY_FILES``."""

FRONTMATTER_MAX_LINES = 30
"""How many leading lines to read for frontmatter parsing. Matches CC's
``FRONTMATTER_MAX_LINES``. Tolerates verbose multi-line descriptions
without forcing a full-file read."""

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

# YAML special characters that, if present unquoted in a value, make the
# parser choke. ``: `` (colon + space) is the most common offender for
# memory descriptions ("Why: prevents X"). CC pattern, see
# ``frontmatterParser.ts:79``.
_YAML_SPECIAL_CHARS = re.compile(r"[{}\[\]*&#!|>%@`]|: ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_memory_files(memdir: Path) -> list[MemoryHeader]:
    """Scan ``memdir`` (recursively) and return memory headers
    newest-first, capped at :data:`MAX_MEMORY_FILES`.

    See module docstring for the strict-write / lenient-read contract.
    Always returns a list (never raises) — including ``[]`` when the
    directory does not exist or contains nothing usable.
    """
    if not memdir.exists():
        print(f"[memdir] scan: directory does not exist: {memdir}",
              file=sys.stderr, flush=True)
        return []

    candidates = [
        p for p in memdir.rglob("*.md")
        if p.name != "MEMORY.md"
    ]
    print(f"[memdir] scan: found {len(candidates)} candidate .md files in {memdir}",
          file=sys.stderr, flush=True)

    headers: list[MemoryHeader] = []
    for f in candidates:
        try:
            stat = f.stat()
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            text = "\n".join(lines[:FRONTMATTER_MAX_LINES])
            meta = _parse_yaml_frontmatter(text)

            headers.append(MemoryHeader(
                filename=str(f.relative_to(memdir)).replace("\\", "/"),
                file_path=f,
                mtime_ms=stat.st_mtime * 1000,
                description=_coerce_description(meta.get("description")),
                type=parse_memory_type(meta.get("type")),
            ))
        except (OSError, UnicodeDecodeError) as e:
            # Real IO error only. Frontmatter / type errors are absorbed
            # inside _parse_yaml_frontmatter and parse_memory_type as
            # field degradation, never raised.
            print(f"[memdir] scan: skip {f.name}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue

    headers.sort(key=lambda h: h.mtime_ms, reverse=True)
    result = headers[:MAX_MEMORY_FILES]
    print(f"[memdir] scan: returning {len(result)} headers "
          f"(parsed {len(headers)}, max {MAX_MEMORY_FILES})",
          file=sys.stderr, flush=True)
    return result


def format_memory_manifest(headers: Iterable[MemoryHeader]) -> str:
    """Render headers as a Markdown bullet list — the format the Sonnet
    selector consumes (P2). One line per header; see module docstring
    for the four possible row shapes.
    """
    lines: list[str] = []
    for h in headers:
        tag = f"[{h.type.value}] " if h.type is not None else ""
        ts = datetime.fromtimestamp(h.mtime_ms / 1000, tz=timezone.utc).isoformat()
        if h.description:
            lines.append(f"- {tag}{h.filename} ({ts}): {h.description}")
        else:
            lines.append(f"- {tag}{h.filename} ({ts})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_yaml_frontmatter(text: str) -> dict[str, Any]:
    """Tolerant frontmatter parser: never raises, never returns non-dict.

    Two-stage parse (CC parity, ``frontmatterParser.ts:148-169``):
      1. Try ``yaml.safe_load`` on the raw frontmatter block.
      2. If that raises, retry after quoting values that contain YAML
         special characters (``: `` is the common one for human-written
         descriptions like ``Why: prevent X``).
      3. If retry still fails, return ``{}`` — caller treats fields as
         absent and the file is still included in scan with degraded
         metadata.

    Returns ``{}`` (never raises) for any of:
      - No ``---`` frontmatter block detected
      - YAML parsed to something other than a mapping (list, scalar)
      - YAML parser raised even after the quote-fixup retry
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}
    block = match.group(1)

    try:
        parsed = yaml.safe_load(block)
    except yaml.YAMLError:
        # First-pass failure: try quoting problematic values and retry.
        try:
            parsed = yaml.safe_load(_quote_problematic_values(block))
        except yaml.YAMLError as e:
            print(f"[memdir] frontmatter parse failed even after quote-fixup: {e}",
                  file=sys.stderr, flush=True)
            return {}

    if not isinstance(parsed, dict):
        return {}
    return parsed


def _quote_problematic_values(frontmatter_text: str) -> str:
    """Wrap values containing YAML special chars in double quotes.

    Operates per-line on ``key: value`` patterns. Skips already-quoted
    values. Escapes embedded backslashes and double-quotes inside the
    value before wrapping. Anything not matching ``key: value`` (list
    items, block scalars, indented continuations) is passed through
    unchanged — best-effort, not a full YAML rewriter.

    Why we need this for memory frontmatter specifically: memory
    descriptions are human-readable sentences and often contain ``: ``
    (e.g. ``Why: prevent silent data loss``). Without this fixup, the
    affected file's description is silently dropped to ``None`` — the
    file still appears, but its single most useful selector signal is
    gone. CC parity, see ``frontmatterParser.ts:85-121``.
    """
    out: list[str] = []
    for line in frontmatter_text.split("\n"):
        m = re.match(r"^([a-zA-Z_-]+):\s+(.+)$", line)
        if not m:
            out.append(line)
            continue
        key, value = m.group(1), m.group(2)
        # Already-quoted values are left alone.
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            out.append(line)
            continue
        if _YAML_SPECIAL_CHARS.search(value):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'{key}: "{escaped}"')
        else:
            out.append(line)
    return "\n".join(out)


def _coerce_description(raw: Any) -> str | None:
    """Coerce a raw frontmatter ``description`` value into ``str | None``.

    String → trimmed (None if empty after trim)
    Number / bool → ``str(raw)`` (LLMs occasionally produce numeric-looking
      descriptions; accept and stringify rather than drop)
    Anything else (list, dict, None) → None
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        trimmed = raw.strip()
        return trimmed or None
    if isinstance(raw, (int, float, bool)):
        return str(raw)
    return None
