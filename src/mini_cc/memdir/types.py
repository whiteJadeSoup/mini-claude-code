"""Memory type taxonomy and header dataclass.

Pure definition layer: no IO, no logging, no side effects. The four memory
types form a closed enum — they are the **contract** between the LLM and
the system, not free-form tags. See P1-1A teaching notes for why a closed
set is load-bearing:

  1. LLM needs to know what kinds of memory exist (so it can decide what
     to write).
  2. Closed-set anchors prevent LLM-driven drift (synonym creep,
     case-variation, accidental new tag invention).
  3. System code branches on `type` — freshness rules, prompt sections,
     and selector heuristics all need a finite type space to enumerate.

CC parity: mirrors ``memdir/memoryTypes.ts:14-31`` (MEMORY_TYPES const +
parseMemoryType). The tolerant parse (unknown → None, never raise) is
deliberate — see ``parse_memory_type`` docstring.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class MemoryType(StrEnum):
    """The four canonical memory types. Adding a fifth requires a
    deliberate decision: closed-set discipline is the whole point.

    - USER: facts about the user themselves (role, background, tech stack)
    - FEEDBACK: corrections/preferences for the assistant ("use uv, not pip",
      "don't summarize diffs"). The most actionable type for a butler.
    - PROJECT: project-level facts NOT derivable from code/git
      (deadlines, incident causes, decision rationale).
    - REFERENCE: pointers to external resources (Linear projects,
      dashboards, Slack channels).
    """

    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


@dataclass(frozen=True)
class MemoryHeader:
    """Index-layer metadata for one memory file. Body is NOT stored here —
    callers that need the body do ``file_path.read_text()`` themselves.

    Invariant: ``file_path == memdir / filename``. Callers guarantee this;
    we do not re-validate (the dataclass is frozen so the invariant cannot
    be broken after construction).

    Why two path-ish fields (``filename`` + ``file_path``) instead of one:
    - ``filename`` is what the LLM sees in the manifest (short, relative).
    - ``file_path`` is what the program uses for fs.read (absolute).
    Different consumers, different needs — not redundant.
    """

    filename: str
    """Relative path from memdir, e.g. ``user_role.md`` or
    ``logs/2026/05/2026-05-17.md`` (KAIROS-mode daily logs)."""

    file_path: Path
    """Absolute path on disk. Equals ``memdir / filename``."""

    mtime_ms: float
    """POSIX timestamp in milliseconds. Stored as ms (not seconds) to
    match CC's manifest format and to keep precision for tight sorting."""

    description: str | None
    """One-line summary from frontmatter. ``None`` when the file has no
    frontmatter, no ``description`` field, or the field value is not a
    non-empty string."""

    type: MemoryType | None
    """Closed-set type. ``None`` when the file has no ``type:`` field,
    the value is not a string, or the value is not in the closed set."""


def parse_memory_type(raw: Any) -> MemoryType | None:
    """Tolerant parse: never raises.

    - Non-string input (number, list, None, dict) → None
    - String not in the closed set → None
    - String in the closed set → corresponding MemoryType

    Why tolerant (CC ``memoryTypes.ts:25-26`` rationale):
    legacy/handwritten files without a ``type:`` field must keep working,
    and a single bad file (e.g. user typo'd ``feedbck``) must not blow up
    scanning. If this function raised instead, the outer try/except in
    scan would silently DROP that file — strictly worse UX than
    surfacing it with ``type=None`` and letting the selector decide.
    """
    if not isinstance(raw, str):
        return None
    try:
        return MemoryType(raw)
    except ValueError:
        return None
