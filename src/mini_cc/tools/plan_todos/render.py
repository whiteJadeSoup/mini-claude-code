from mini_cc.tools.base import TodoPlanOutput, ToolOutput


def render_received(args: dict) -> str:
    items = args.get("items", [])
    return f"{len(items)} items"


def render_complete(args: dict, output: ToolOutput | None) -> str:
    if not isinstance(output, TodoPlanOutput):
        return render_received(args)
    return f"{output.count} todos"
