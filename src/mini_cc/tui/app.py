"""Textual TUI for mini-cc.

Layout (ChatLog children, top → bottom):
  [Logo]      — boot banner; scrolls off with history
  [history]   — user / assistant / tool rows (mounted via `mount(before=ToolStatus)`)
  ToolStatus  — fixed live tool rows during a turn (hidden between turns)
  TurnFooter  — spinner during turn / ※ <ci phrase> · Ns after turn ends
  Input       — user input bar (flows with scroll, not docked to screen bottom)
  StatusBar   — session status (model · ctx% · cwd · ↓↑), Input's next sibling
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import signal
import time

# SGR mouse-tracking sequences that Windows Terminal leaks into the Input widget.
# Pattern covers: full ANSI escape, bare numeric tail, and lone M-prefixed tail.
_MOUSE_LEAK_RE = re.compile(
    r"\x1b\[[^a-zA-Z]*[a-zA-Z]"  # full ANSI escape sequence
    r"|[M\x1b][<]?[\d;]+[Mm]"    # SGR mouse tail with or without escape prefix
    r"|[\x00-\x08\x0b-\x1c\x7f]" # stray control characters
)

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message as TxtMessage
from textual.widgets import Input, Markdown, Static

from mini_cc.engine.predicates import is_persisted_layer
from mini_cc.engine.messages import (
    AssistantMessage,
    CompactBoundaryMessage,
    Message,
    StatusMessage,
    SystemPromptMessage,
    TextBlock,
    ToolResultMessage,
    ToolUseBlock,
)
from mini_cc.state import usage

# Braille spinner for TurnFooter
_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Blink-dot for ToolStatus: grey ● on 500 ms / off 500 ms (10 fps = 5 ticks each)
_BLINK = ["●", "●", "●", "●", "●", " ", " ", " ", " ", " "]

# Song ci phrase pairs: (short — shown during turn, full — revealed at turn end).
_CI_PHRASES: list[tuple[str, str]] = [
    ("寻寻觅觅",         "寻寻觅觅冷冷清清凄凄惨惨戚戚"),
    ("知否知否",         "知否知否应是绿肥红瘦"),
    ("昨夜雨疏风骤",     "昨夜雨疏风骤浓睡不消残酒"),
    ("独上高楼",         "独上高楼望断天涯路"),
    ("众里寻他千百度",   "众里寻他千百度蓦然回首那人却在灯火阑珊处"),
    ("蓦然回首",         "蓦然回首那人却在灯火阑珊处"),
    ("无可奈何花落去",   "无可奈何花落去似曾相识燕归来"),
    ("此事古难全",       "此事古难全但愿人长久千里共婵娟"),
    ("明月几时有",       "明月几时有把酒问青天"),
    ("醉里挑灯看剑",     "醉里挑灯看剑梦回吹角连营"),
    ("衣带渐宽终不悔",   "衣带渐宽终不悔为伊消得人憔悴"),
    ("欲说还休",         "欲说还休却道天凉好个秋"),
    ("杨柳岸晓风残月",   "今宵酒醒何处杨柳岸晓风残月"),
    ("把吴钩看了",       "把吴钩看了栏杆拍遍无人会登临意"),
    ("一蓑烟雨任平生",   "竹杖芒鞋轻胜马一蓑烟雨任平生"),
    ("也无风雨也无晴",   "回首向来萧瑟处也无风雨也无晴"),
    ("大江东去",         "大江东去浪淘尽千古风流人物"),
    ("才下眉头却上心头", "此情无计可消除才下眉头却上心头"),
    ("物是人非事事休",   "物是人非事事休欲语泪先流"),
    ("似曾相识燕归来",   "无可奈何花落去似曾相识燕归来"),
    ("昨夜西风凋碧树",   "昨夜西风凋碧树独上高楼望断天涯路"),
    ("庭院深深深几许",   "庭院深深深几许杨柳堆烟帘幕无重数"),
    ("金风玉露一相逢",   "金风玉露一相逢便胜却人间无数"),
]

_TOOL_ARGS_MAX = 60

_LOGO = (
    "[grey50]  / \\__[/grey50]       [bold cyan]█▀▄▀█ █ █▄░█ █[/bold cyan] [dim]·[/dim] [bold cyan]█▀▀ █▀▀[/bold cyan]\n"
    "[grey50] (    @\\___[/grey50]   [bold cyan]█░▀░█ █ █░▀█ █▄▄  █▄▄ █▄▄[/bold cyan]\n"
    "[grey50] /         O[/grey50]\n"
    "[grey50]/   (_____/[/grey50]   [dim]your coding agent[/dim]\n"
    "[grey50]\\_____/   U[/grey50]"
)


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _shorten_cwd(path: str) -> str:
    home = os.path.expanduser("~")
    p = path.replace("\\", "/")
    if p.startswith(home.replace("\\", "/")):
        return "~" + p[len(home):]
    return p


def _fmt_args(args: dict) -> str:
    if not args:
        return ""
    first_key = next(iter(args))
    flat = " ".join(str(args[first_key]).split())
    if len(args) > 1:
        flat = f"{flat}, …"
    return flat[:_TOOL_ARGS_MAX - 1] + "…" if len(flat) > _TOOL_ARGS_MAX else flat


# ---------------------------------------------------------------------------
# Textual messages (cross-widget communication)
# ---------------------------------------------------------------------------


class EngineMsg(TxtMessage):
    """Wraps an engine Message for Textual's message bus."""

    def __init__(self, payload: Message) -> None:
        super().__init__()
        self.payload = payload


class ToolFlushed(TxtMessage):
    """Posted by ToolStatus when a completed tool should appear in ChatLog."""

    def __init__(self, markup: str) -> None:
        super().__init__()
        self.markup = markup


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class ChatLog(VerticalScroll):
    """Scrollable chat history. Completed tool rows (green ●) mount here."""

    DEFAULT_CSS = """
    ChatLog {
        height: 1fr;
        padding: 0 1;
    }
    ChatLog > Static {
        margin-bottom: 1;
    }
    ChatLog > Markdown {
        margin-bottom: 1;
    }
    ChatLog > .agent-msg {
        border-left: heavy $accent;
        padding: 0 0 0 1;
    }
"""

    def compose(self) -> ComposeResult:
        yield Static(_LOGO)   # logo at top of scroll history
        yield ToolStatus()    # live tool rows (hidden between turns)
        yield TurnFooter()    # spinner / summary just above Input
        yield Input(placeholder="Message…", id="input-bar")
        yield StatusBar()     # session status below Input; flows with scroll

    def append_user(self, text: str) -> None:
        self._push(Static(f"[black on cyan bold] You [/black on cyan bold]  {text}"))

    def append_assistant(self, text: str) -> None:
        self._push(Markdown(text, classes="agent-msg"))

    def append_markup(self, markup: str, classes: str = "") -> None:
        self._push(Static(markup, classes=classes))

    def append_system(self, text: str) -> None:
        self._push(Static(f"[dim]{text}[/dim]"))

    def _push(self, widget) -> None:
        # Insert before ToolStatus so all messages stay above the
        # turn-status / input cluster at the bottom.
        try:
            tool_status = self.query_one(ToolStatus)
            self.mount(widget, before=tool_status)
        except Exception:
            self.mount(widget)
        self.scroll_end(animate=False)


class ToolStatus(Static):
    """Fixed live tool-row panel above Input. Hidden between turns.

    Main tools get an individual blink-dot row. Sub-tools are folded into a
    rolling group row under their parent. Completed tools are flushed to
    ChatLog via ToolFlushed so they persist in scroll history.
    """

    DEFAULT_CSS = """
    ToolStatus {
        height: auto;
        display: none;
        padding: 0 1;
        color: grey;
    }
    ToolStatus.active {
        display: block;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._tools: dict[str, dict] = {}
        self._tool_order: list[str] = []
        self._groups: dict[str, dict] = {}
        self._frame = 0

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)

    def start_turn(self) -> None:
        self._tools.clear()
        self._tool_order.clear()
        self._groups.clear()
        self._frame = 0
        self.add_class("active")

    def end_turn(self) -> None:
        self._tools.clear()
        self._tool_order.clear()
        self._groups.clear()
        self.update("")
        self.remove_class("active")

    def add_tool(
        self,
        call_id: str,
        name: str,
        args_repr: str,
        prefix: str,
        asst_id: str,
        parent_id: str | None,
    ) -> None:
        if parent_id is None:
            self._tools[call_id] = {
                "name": name,
                "args_repr": args_repr,
                "prefix": prefix,
                "asst_id": asst_id,
                "started_at": time.monotonic(),
            }
            self._tool_order.append(call_id)
        else:
            main_call_id = next(
                (cid for cid, t in self._tools.items() if t["asst_id"] == parent_id),
                None,
            )
            if main_call_id is None:
                return
            if main_call_id not in self._groups:
                self._groups[main_call_id] = {
                    "tool_count": 0,
                    "current_label": f"{name}({args_repr})",
                    "started_at": time.monotonic(),
                }
            g = self._groups[main_call_id]
            g["tool_count"] += 1
            g["current_label"] = f"{name}({args_repr})"

    def complete_tool(self, call_id: str) -> None:
        if call_id not in self._tools:
            return
        t = self._tools.pop(call_id)
        self._tool_order.remove(call_id)
        elapsed = time.monotonic() - t["started_at"]
        if call_id in self._groups:
            g = self._groups.pop(call_id)
            markup = (
                f"{t['prefix']}[green]●[/green] "
                f"[cyan]{t['name']}[/cyan]({t['args_repr']})"
                f" · {g['tool_count']} sub-tools · {elapsed:.1f}s"
            )
        else:
            markup = (
                f"{t['prefix']}[green]●[/green] "
                f"[cyan]{t['name']}[/cyan]({t['args_repr']})"
            )
        self.post_message(ToolFlushed(markup))

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_BLINK)
        if not self._tools:
            return
        b = _BLINK[self._frame]
        lines = []
        for cid in self._tool_order:
            t = self._tools[cid]
            lines.append(
                f"{t['prefix']}[grey50]{b}[/grey50] "
                f"[cyan]{t['name']}[/cyan]({t['args_repr']})"
            )
            if cid in self._groups:
                g = self._groups[cid]
                elapsed = int(time.monotonic() - g["started_at"])
                lines.append(
                    f"{t['prefix']}│  [grey50]{b}[/grey50] "
                    f"[cyan]{g['current_label']}[/cyan]"
                    f" · {g['tool_count']} · {elapsed}s"
                )
        self.update("\n".join(lines))


class TurnFooter(Static):
    """Spinner during turn; ※ Worked for Xs after turn ends."""

    DEFAULT_CSS = """
    TurnFooter {
        display: none;
        padding: 0 1;
        color: $text-muted;
    }
    TurnFooter.active {
        display: block;
        height: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._started_at: float | None = None
        self._input_base = 0
        self._output_base = 0
        self._frame = 0
        self._verb_short = _CI_PHRASES[0][0]
        self._verb_full = _CI_PHRASES[0][1]

    def on_mount(self) -> None:
        self.set_interval(0.1, self._tick)

    def start_turn(self) -> None:
        self._started_at = time.monotonic()
        self._input_base = usage._tracker.input_tokens_used()
        self._output_base = usage._tracker.output_tokens_used()
        self._frame = 0
        self._verb_short, self._verb_full = random.choice(_CI_PHRASES)
        self.add_class("active")

    def stop_turn(self) -> str | None:
        """Stop the spinner and return the summary markup (caller appends to ChatLog)."""
        elapsed = int(time.monotonic() - self._started_at) if self._started_at else 0
        self._started_at = None
        self.update("")
        self.remove_class("active")
        if elapsed > 0:
            return f"[dim]※ {self._verb_full} · {elapsed}s[/dim]"
        return None

    def _tick(self) -> None:
        if self._started_at is None:
            return
        self._frame = (self._frame + 1) % len(_FRAMES)
        f = _FRAMES[self._frame]
        elapsed = int(time.monotonic() - self._started_at)
        inp = usage._tracker.input_tokens_used() - self._input_base
        out = usage._tracker.output_tokens_used() - self._output_base
        self.update(
            f"[bold cyan]{f} {self._verb_short}…[/bold cyan] "
            f"[cyan]({elapsed}s · ↓ {_fmt_tokens(inp)} · ↑ {_fmt_tokens(out)})[/cyan]"
        )


class StatusBar(Static):
    """Session-level status line. Sits below Input as ChatLog's last child;
    scrolls with content (not docked to the terminal bottom)."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._model = ""
        self._cwd = ""

    def set_session(self, model: str, cwd: str) -> None:
        self._model = model
        self._cwd = cwd
        self.refresh_status()

    def refresh_status(self) -> None:
        tracker = usage._tracker
        ctx_max = tracker.context_limit or 0

        # Ask the engine for current occupancy — it combines record-accurate
        # baseline with a char-based estimate so new tool_results landing
        # between LLM calls are reflected immediately. `~` prefix signals
        # the number is fully inferred (no real API record yet).
        from mini_cc.engine.query_engine import get_engine
        try:
            engine = get_engine()
            ctx_used = engine.current_context_tokens(parent_id=None)
        except Exception:  # noqa: BLE001
            ctx_used = tracker.context_tokens_used()
        estimate_prefix = "~" if not tracker._records and ctx_used > 0 else ""

        pct = (ctx_used / ctx_max * 100) if ctx_max else 0
        if pct >= 80:
            ctx_color = "red"
        elif pct >= 50:
            ctx_color = "yellow"
        else:
            ctx_color = "dim"

        ctx_max_s = _fmt_tokens(ctx_max) if ctx_max else "?"
        out_total = tracker.output_tokens_used()
        cwd_display = _shorten_cwd(self._cwd)

        self.update(
            f"[dim]▸ {self._model or 'unknown'}[/dim]  ·  "
            f"[{ctx_color}]ctx {estimate_prefix}{_fmt_tokens(ctx_used)} / {ctx_max_s} "
            f"({estimate_prefix}{pct:.0f}%)[/{ctx_color}]  ·  "
            f"[dim]{cwd_display}[/dim]  ·  "
            f"[dim]↑ {_fmt_tokens(out_total)}[/dim]"
        )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class MiniCCApp(App):
    TITLE = "mini-cc"
    # priority=True: fires before any focused widget processes the key,
    # bypassing the normal event-bubbling chain.
    BINDINGS = [Binding("ctrl+c", "interrupt", show=False, priority=True)]
    CSS = """
    Screen { background: $surface; }

    Input {
        height: 3;
        border: solid $accent;
        margin: 0;
    }
    Input:disabled {
        border: solid grey;
        color: grey;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._engine = None
        self._ctx = None
        self._current_worker = None
        self._interrupt_count = 0

    def compose(self) -> ComposeResult:
        yield ChatLog()

    async def on_mount(self) -> None:
        from mini_cc import commands, config, llm as agent
        from mini_cc.commands import CommandContext
        from mini_cc.consumers import persistence
        from mini_cc.consumers.persistence import PersistenceConsumer
        from mini_cc.engine.query_engine import QueryEngine, set_engine
        from mini_cc.tools.skills import _skill_manager

        engine = QueryEngine(
            llm_base=agent._llm_base,
            main_tools=agent.MAIN_TOOLS,
            sub_tools=agent.SUB_TOOLS,
            model_name=agent._MODEL_NAME,
            system_prompt_builder=agent._build_system_prompt,
        )
        set_engine(engine)
        self._engine = engine

        engine.subscribe(
            PersistenceConsumer(),
            name="persistence",
            filter=is_persisted_layer,
            policy="async",
        )
        engine.subscribe(TextualUIConsumer(self), name="tui", policy="async")
        commands.sync_skill_commands(_skill_manager)
        await engine.boot()

        chat_log = self.query_one(ChatLog)
        self._ctx = CommandContext(
            tracker=usage._tracker,
            engine=engine,
            notify=chat_log.append_system,
        )

        chat_log.append_system(f"transcript: {persistence.transcript_path()}")

        self.query_one(StatusBar).set_session(agent._MODEL_NAME, config.CWD)

        self.set_focus(self.query_one(Input))

        # Fallback for platforms where Ctrl+C reaches the OS as SIGINT
        # instead of being delivered as a Textual key event (Linux/Mac).
        loop = asyncio.get_running_loop()
        def _sigint(signum, frame):
            try:
                loop.call_soon_threadsafe(self._do_interrupt)
            except Exception:
                os._exit(0)
        signal.signal(signal.SIGINT, _sigint)

    # -- engine message routing ---------------------------------------------

    def on_engine_msg(self, msg: EngineMsg) -> None:
        payload = msg.payload
        chat_log = self.query_one(ChatLog)
        tool_status = self.query_one(ToolStatus)
        turn_footer = self.query_one(TurnFooter)

        if isinstance(payload, StatusMessage):
            if payload.event == "turn_start":
                tool_status.start_turn()
                turn_footer.start_turn()
                chat_log.scroll_end(animate=False)
            elif payload.event == "turn_end":
                tool_status.end_turn()
                summary = turn_footer.stop_turn()
                if summary:
                    chat_log.append_markup(summary, classes="turn-summary")
                inp = self.query_one(Input)
                inp.disabled = False
                inp.placeholder = "Message…"
                self.set_focus(inp)
                self.query_one(StatusBar).refresh_status()
            elif payload.event == "skills_changed":
                data = payload.data or {}
                for name in data.get("added") or []:
                    chat_log.append_markup(f"[green dim]＋ skill: /{name}[/green dim]")
                for name in data.get("removed") or []:
                    chat_log.append_markup(f"[red dim]－ skill: /{name}[/red dim]")

        elif isinstance(payload, AssistantMessage):
            if isinstance(payload.content, TextBlock) and payload.content.text.strip():
                chat_log.append_assistant(payload.content.text)
            elif isinstance(payload.content, ToolUseBlock):
                block = payload.content
                tool_status.add_tool(
                    call_id=block.call_id,
                    name=block.name,
                    args_repr=_fmt_args(block.args or {}),
                    prefix="  " if payload.parent_id else "",
                    asst_id=payload.id,
                    parent_id=payload.parent_id,
                )
            self.query_one(StatusBar).refresh_status()

        elif isinstance(payload, ToolResultMessage):
            tool_status.complete_tool(payload.tool_call_id)

        elif isinstance(payload, CompactBoundaryMessage):
            label = "auto-compact" if payload.auto else "compact"
            core = f" ※ {label} · removed {payload.pre_count} msgs "
            width = max(20, (chat_log.size.width or 80) - 4)
            pad = max(0, width - len(core))
            left = pad // 2
            right = pad - left
            chat_log.append_markup(f"[dim]{'─' * left}{core}{'─' * right}[/dim]")

    def on_tool_flushed(self, msg: ToolFlushed) -> None:
        self.query_one(ChatLog).append_markup(msg.markup)

    # -- interrupt handling -------------------------------------------------

    def _do_interrupt(self) -> None:
        """First Ctrl+C cancels turn; second force-exits."""
        self._interrupt_count += 1
        if self._interrupt_count == 1 and self._current_worker is not None:
            self._current_worker.cancel()
            try:
                self.query_one(ChatLog).append_markup(
                    "[yellow]⚠ interrupted — press Ctrl+C again to force-quit[/yellow]"
                )
                self.query_one(ToolStatus).end_turn()
                summary = self.query_one(TurnFooter).stop_turn()
                if summary:
                    self.query_one(ChatLog).append_markup(summary, classes="turn-summary")
                inp = self.query_one(Input)
                inp.disabled = False
                inp.placeholder = "Message…"
                self.set_focus(inp)
            except Exception:
                pass
            self._current_worker = None
        else:
            if self._current_worker is not None:
                self._current_worker.cancel()
                self._current_worker = None
            self.exit()

    def action_interrupt(self) -> None:
        """Triggered by the priority Ctrl+C binding."""
        self._do_interrupt()

    # -- input handling -----------------------------------------------------

    @on(Input.Changed, "#input-bar")
    def _clean_input_leak(self, event: Input.Changed) -> None:
        """Strip SGR mouse-tracking bytes that Windows Terminal leaks into the Input."""
        cleaned = _MOUSE_LEAK_RE.sub("", event.value)
        if cleaned != event.value:
            event.input.value = cleaned

    @on(Input.Submitted, "#input-bar")
    def handle_input(self, event: Input.Submitted) -> None:
        text = _MOUSE_LEAK_RE.sub("", event.value).strip()
        event.input.clear()
        if not text:
            return
        self._interrupt_count = 0
        event.input.disabled = True
        event.input.placeholder = "Running… (Ctrl+C to cancel)"
        event.input.blur()
        self._current_worker = self._run_turn(text)

    @work(exclusive=True)
    async def _run_turn(self, text: str) -> None:
        from mini_cc import commands
        from mini_cc.tools.skills import _skill_manager

        try:
            await self._engine.refresh_skills()
            commands.sync_skill_commands(_skill_manager)
        except Exception as e:
            self.query_one(ChatLog).append_system(f"[skills refresh failed: {e}]")

        if text.startswith("/"):
            cmd_name, *rest = text[1:].split(maxsplit=1)
            args = rest[0] if rest else ""
            if not await commands.registry.handle(cmd_name, args, self._ctx):
                self.query_one(ChatLog).append_system(f"Unknown command: /{cmd_name}")
            if self._ctx.should_exit:
                self.exit()
                return
        else:
            self.query_one(ChatLog).append_user(text)
            try:
                await self._engine.query(text)
            except Exception as e:
                self.query_one(ChatLog).append_markup(
                    f"[bold red]✗ Error: {e}[/bold red]"
                )

        inp = self.query_one(Input)
        inp.disabled = False
        inp.placeholder = "Message…"
        self.set_focus(inp)

    async def on_unmount(self) -> None:
        if self._engine:
            try:
                await asyncio.wait_for(self._engine.shutdown(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------


class TextualUIConsumer:
    """Routes engine messages to MiniCCApp via Textual's message bus.

    Queue and lifecycle live on the ``Subscription``; this class only
    declares *what to do* with each message.
    """

    def __init__(self, app: MiniCCApp) -> None:
        self._app = app

    async def on_message(self, msg: Message) -> None:
        self._app.post_message(EngineMsg(msg))
