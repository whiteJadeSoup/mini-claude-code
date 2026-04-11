from collections.abc import Callable

from pydantic import BaseModel, SkipValidation

import skills
import usage
from usage import UsageTracker


class CommandContext(BaseModel):
    """Shared state passed to all command handlers.

    Using a model instead of a dict so field typos fail loudly at init
    rather than silently producing KeyError at runtime.
    """
    model_config = {"arbitrary_types_allowed": True}

    history: list
    tracker: UsageTracker
    run_agent: SkipValidation[Callable]
    compact: SkipValidation[Callable]
    system_prompt_builder: SkipValidation[Callable]
    should_exit: bool = False  # set by /exit handler, checked by main loop to break


class CommandRegistry:
    """Registry for slash commands. Handlers receive (args: str, ctx: CommandContext)."""

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, name: str, handler: Callable = None):
        """Register a command handler. Use as decorator or direct call.

        As decorator: @registry.register("name")
        Direct call:  registry.register("name", handler_fn)
        """
        if handler is not None:
            self._handlers[name] = handler
            return handler
        def decorator(func):
            self._handlers[name] = func
            return func
        return decorator

    def handle(self, name: str, args: str, ctx: CommandContext) -> bool:
        """Dispatch command. Returns True if handled, False if unknown."""
        handler = self._handlers.get(name)
        if not handler:
            return False
        handler(args, ctx)
        return True

    def unregister(self, name: str):
        self._handlers.pop(name, None)

    def names(self) -> list[str]:
        return list(self._handlers)


registry = CommandRegistry()


# --- Built-in commands ---

@registry.register("context")
def cmd_context(args, ctx):
    ctx.tracker.summary(len(ctx.history))


@registry.register("limit")
def cmd_limit(args, ctx):
    try:
        ctx.tracker.set_limit(int(args))
        print(f"Context limit set to {ctx.tracker.context_limit:,}")
    except (ValueError, IndexError):
        print("Usage: /limit <number>")


@registry.register("exit")
def cmd_exit(args, ctx):
    ctx.should_exit = True


@registry.register("compact")
def cmd_compact(args, ctx):
    print("[Compacting...]", flush=True)
    try:
        n = ctx.compact(ctx.history, custom_instructions=args)
        print(f"[Compacted: {n} messages removed]")
    except Exception as e:
        print(f"[Compact failed: {e}]")


@registry.register("tasks")
def cmd_tasks(args, ctx):
    import tasks as t
    if args.strip() == "clear":
        print(t._tasks.clear())
    else:
        print(t._tasks.render())


@registry.register("skills")
def cmd_skills(args, ctx):
    names = skills._skill_manager.names()
    if not names:
        print("No skills found. Add skills to skills/<name>/SKILL.md")
        return
    for name in names:
        desc = skills._skill_manager.description(name)
        print(f"  /{name} — {desc}")


# Snapshot of built-in commands — sync_skill_commands uses this
# to distinguish built-ins from dynamically registered skill commands.
_BUILTIN_CMDS = frozenset(registry.names())


# --- Dynamic skill commands ---

def _make_skill_handler(skill_name):
    """Factory that captures skill_name by value — avoids closure-over-loop-variable bug."""
    def handler(args, ctx):
        msg = (f'Use the \'{skill_name}\' skill: call run_skill("{skill_name}", request=...). '
               f'Include context param only if the request needs conversation history.')
        if args:
            msg += f"\n\nUser request: {args}"
        print("Agent: ", end="", flush=True)
        ctx.run_agent(msg, ctx.history)
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
