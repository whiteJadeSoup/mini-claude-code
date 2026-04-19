from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, SkipValidation

from mini_cc.state.usage import UsageTracker
from mini_cc.tools import skills


class CommandContext(BaseModel):
    """Shared state passed to all command handlers."""
    model_config = {"arbitrary_types_allowed": True}

    tracker: UsageTracker
    # SkipValidation + Any: we just need a runtime attribute. Typing it as
    # "QueryEngine" would force a forward-ref rebuild because commands.py
    # is imported before query_engine finishes loading.
    engine: SkipValidation[Any]
    should_exit: bool = False
    # notify: output channel for command text. In Textual mode this posts to
    # ChatLog; callers set it at context-creation time.
    notify: SkipValidation[Callable[[str], None]] = print


class CommandRegistry:
    """Registry for slash commands. Handlers are async, receive (args, ctx)."""

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, name: str, handler: Callable = None):
        if handler is not None:
            self._handlers[name] = handler
            return handler
        def decorator(func):
            self._handlers[name] = func
            return func
        return decorator

    async def handle(self, name: str, args: str, ctx: CommandContext) -> bool:
        handler = self._handlers.get(name)
        if not handler:
            return False
        await handler(args, ctx)
        return True

    def unregister(self, name: str):
        self._handlers.pop(name, None)

    def names(self) -> list[str]:
        return list(self._handlers)


registry = CommandRegistry()


# --- Built-in commands ---

@registry.register("context")
async def cmd_context(args, ctx):
    from io import StringIO
    from rich.console import Console
    buf = StringIO()
    c = Console(file=buf, width=100, no_color=True, highlight=False)
    ctx.tracker.summary(len(ctx.engine.store.all()), console=c)
    ctx.notify(buf.getvalue().strip())


@registry.register("limit")
async def cmd_limit(args, ctx):
    try:
        ctx.tracker.set_limit(int(args))
        ctx.notify(f"Context limit set to {ctx.tracker.context_limit:,}")
    except (ValueError, IndexError):
        ctx.notify("Usage: /limit <number>")


@registry.register("exit")
async def cmd_exit(args, ctx):
    ctx.should_exit = True


@registry.register("compact")
async def cmd_compact(args, ctx):
    ctx.notify("[Compacting...]")
    try:
        n = await ctx.engine.compact(custom_instructions=args)
        ctx.notify(f"[Compacted: {n} messages removed]")
    except Exception as e:
        ctx.notify(f"[Compact failed: {e}]")


@registry.register("tasks")
async def cmd_tasks(args, ctx):
    from mini_cc.state import tasks as t
    if args.strip() == "clear":
        ctx.notify(str(t._tasks.clear()))
    else:
        ctx.notify(t._tasks.render())


@registry.register("skills")
async def cmd_skills(args, ctx):
    names = skills._skill_manager.names()
    if not names:
        ctx.notify("No skills found. Add skills to skills/<name>/SKILL.md")
        return
    lines = [f"  /{name} — {skills._skill_manager.description(name)}" for name in names]
    ctx.notify("\n".join(lines))


_BUILTIN_CMDS = frozenset(registry.names())


# --- Dynamic skill commands ---

def _make_skill_handler(skill_name):
    async def handler(args, ctx):
        # The skill handler's job is to prompt the agent to invoke run_skill.
        # We drive the main branch by iterating engine.query; consumers do
        # the actual rendering.
        msg = (f'Use the \'{skill_name}\' skill: call run_skill("{skill_name}", request=...). '
               f'Include context param only if the request needs conversation history.')
        if args:
            msg += f"\n\nUser request: {args}"
        await ctx.engine.query(msg)
    return handler


def sync_skill_commands(skill_manager) -> bool:
    """Sync slash commands with current skills. Returns True if anything changed."""
    desired = set(skill_manager.names())
    registered = set(registry._handlers) - _BUILTIN_CMDS
    to_remove = registered - desired
    to_add = desired - registered
    for name in to_remove:
        registry.unregister(name)
    for name in to_add:
        registry.register(name, _make_skill_handler(name))
    return bool(to_remove or to_add)
