"""Channel B-1: inject the MEMORY.md index into messages[0] as a
<system-reminder> context message.

Two layers, mirroring CC's split:
  build_memory_context  ↔  getClaudeMds (claudemd.ts:1185) — inner value
  render_user_context   ↔  prependUserContext (api.ts:461) — outer wrapper

Keeping them separate is what makes the wrapper data-driven: the caller
(boot) assembles a {key: value} dict, and render_user_context iterates it.
Adding gitStatus/userEmail later = one more dict key, zero changes here.
"""
from __future__ import annotations

from pathlib import Path

from mini_cc.memdir.truncate import truncate_entrypoint

# Fixed description string, like CC's per-type `description` (claudemd.ts:1176).
# It is content-about-content, not data — so it lives here, not in the dict.
_MEMORY_DESC = "your persistent memory, persists across conversations"


def _neutralize_reminder_tags(text: str) -> str:
    """Stop index content that literally contains a system-reminder tag from
    prematurely closing the injected <system-reminder> block. Escape the
    angle brackets so the model reads it as literal text, not structure.
    Not hypothetical: this project's own memory is ABOUT memory internals,
    so '</system-reminder>' really can appear in the index.

    Case-sensitive by design: only lowercase tags are neutralized, matching
    how the tag is emitted/read. A mixed-case '</System-Reminder>' in the
    index is left as ordinary prose (not treated as a closing tag).

    Replacement order is irrelevant — the two patterns share no substring
    overlap ('</...>' has a '/' where '<...>' does not)."""
    return (
        text.replace("</system-reminder>", "&lt;/system-reminder&gt;")
            .replace("<system-reminder>", "&lt;system-reminder&gt;")
    )


def build_memory_context(memory_dir: Path) -> str | None:
    """Inner layer: produce the `memory` value, or None if there's nothing
    to inject. None (not "") so the caller's `if value:` drops the key
    entirely — an empty index must not emit a bare `# memory` heading."""
    entrypoint = memory_dir / "MEMORY.md"
    try:
        raw = entrypoint.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return None
    # Neutralize AFTER truncate so we escape only what actually gets injected.
    content = _neutralize_reminder_tags(truncate_entrypoint(raw))
    if not content.strip():
        return None
    return f"Contents of {entrypoint} ({_MEMORY_DESC}):\n\n{content}"


def render_user_context(context: dict[str, str]) -> str:
    """Outer layer: wrap a {key: value} dict into one <system-reminder>
    block, iterating keys in insertion order. Empty dict → "" so the caller
    can skip dispatching an empty message.

    Precondition: values are non-None strings. The caller filters absent
    context BEFORE calling (e.g. `if mem := build_memory_context(...): ctx["memory"] = mem`).
    A None value would render the literal "None"; by design we do NOT guard
    here, so a None leak surfaces as a visible bug rather than vanishing."""
    if not context:
        return ""
    body = "\n".join(f"# {key}\n{value}" for key, value in context.items())
    return (
        "<system-reminder>\n"
        "As you answer the user's questions, you can use the following context:\n"
        f"{body}\n\n"
        "IMPORTANT: this context may or may not be relevant to your tasks. You should "
        "not respond to this context unless it is highly relevant to your task.\n"
        "</system-reminder>\n"
    )


def render_relevant_memories(memories) -> str:
    """Render auto-surfaced memories: one <system-reminder> block PER memory.

    Mirrors CC (messages.ts:3708 — wrapMessagesInSystemReminder maps over each
    memory, so 1 memory = 1 block). Each block = the frozen freshness header
    (m.header, computed once at surfacing time) + the file body. Embedded
    system-reminder tags in the body are neutralized so they can't prematurely
    close the block. `memories` is a list[SurfacedMemory] (annotated loosely to
    avoid a runtime import cycle — only .header/.content are read)."""
    blocks = [
        "<system-reminder>\n"
        f"{m.header}\n\n"
        f"{_neutralize_reminder_tags(m.content)}\n"
        "</system-reminder>"
        for m in memories
    ]
    return "\n".join(blocks) + "\n"
