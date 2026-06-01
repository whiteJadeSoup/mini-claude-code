"""Engine-message → TUI rendering, as an Open/Closed dispatch registry.

Adding a new message type = add a `@renders(NewType)` function here; the
`on_engine_msg` dispatcher in app.py never changes (open for extension,
closed for modification).

Layering: renderers live in the UI layer and *read* domain message objects.
The dependency only ever flows UI → domain — `engine/messages.py` knows
nothing about the TUI. Renderers reach widgets through the `MiniCCApp`
accessor properties (app.chat_log / app.tool_status / ...), so they never
import widget classes either.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from mini_cc.engine.messages import (
    AssistantMessage,
    CompactBoundaryMessage,
    Message,
    RelevantMemoryMessage,
    StatusMessage,
    TextBlock,
    ToolResultMessage,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from mini_cc.consumers.tui.app import MiniCCApp

Renderer = Callable[["MiniCCApp", Message], None]

_RENDERERS: dict[type, Renderer] = {}


def renders(msg_type: type):
    """Register a renderer for an engine message type."""
    def deco(fn: Renderer) -> Renderer:
        _RENDERERS[msg_type] = fn
        return fn
    return deco


def dispatch(app: "MiniCCApp", payload: Message) -> None:
    """Route an engine message to its renderer. Unregistered types are a
    no-op — same behavior as the old if-elif chain falling through."""
    renderer = _RENDERERS.get(type(payload))
    if renderer is not None:
        renderer(app, payload)


def _recalled_markup(filenames: list[str]) -> str:
    """One-line feedback for auto-surfaced memories, mirroring CC's collapsed
    "recalled N memories". Pure (no widget) so it is unit-testable."""
    n = len(filenames)
    noun = "memory" if n == 1 else "memories"
    return f"[dim]※ recalled {n} {noun}: {', '.join(filenames)}[/dim]"


@renders(StatusMessage)
def _render_status(app: "MiniCCApp", m: StatusMessage) -> None:
    if m.event == "turn_start":
        app.tool_status.start_turn()
        app.turn_footer.start_turn()
        app.chat_log.scroll_end(animate=False)
    elif m.event == "turn_end":
        app.tool_status.end_turn()
        summary = app.turn_footer.stop_turn()
        if summary:
            app.chat_log.append_markup(summary, classes="turn-summary")
        inp = app.input_bar
        inp.disabled = False
        inp.placeholder = "Message…"
        app.set_focus(inp)
        app.status_bar.refresh_status()
    elif m.event == "skills_changed":
        data = m.data or {}
        for name in data.get("added") or []:
            app.chat_log.append_markup(f"[green dim]＋ skill: /{name}[/green dim]")
        for name in data.get("removed") or []:
            app.chat_log.append_markup(f"[red dim]－ skill: /{name}[/red dim]")


@renders(AssistantMessage)
def _render_assistant(app: "MiniCCApp", m: AssistantMessage) -> None:
    if isinstance(m.content, TextBlock) and m.content.text.strip():
        app.chat_log.append_assistant(m.content.text)
    elif isinstance(m.content, ToolUseBlock):
        block = m.content
        app.tool_status.add_tool(
            call_id=block.call_id,
            name=block.name,
            args=block.args or {},
            prefix="  " if m.parent_id else "",
            asst_id=m.id,
            parent_id=m.parent_id,
        )
    app.status_bar.refresh_status()


@renders(ToolResultMessage)
def _render_tool_result(app: "MiniCCApp", m: ToolResultMessage) -> None:
    app.tool_status.complete_tool(m.tool_call_id, output=m.output)


@renders(CompactBoundaryMessage)
def _render_compact(app: "MiniCCApp", m: CompactBoundaryMessage) -> None:
    label = "auto-compact" if m.auto else "compact"
    core = f" ※ {label} · removed {m.pre_count} msgs "
    width = max(20, (app.chat_log.size.width or 80) - 4)
    pad = max(0, width - len(core))
    left = pad // 2
    right = pad - left
    app.chat_log.append_markup(f"[dim]{'─' * left}{core}{'─' * right}[/dim]")


@renders(RelevantMemoryMessage)
def _render_relevant_memory(app: "MiniCCApp", m: RelevantMemoryMessage) -> None:
    # Auto-surfaced memories were injected into the conversation; give the
    # user a one-line "recalled N" trace (CC parity) instead of silence.
    app.chat_log.append_markup(_recalled_markup([x.filename for x in m.memories]))
