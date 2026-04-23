from mini_cc.tools.base import ToolOutput


def render_received(args: dict) -> str:
    return f"{args.get('task_id', '')} → {args.get('status', '')}"


def render_complete(args: dict, output: ToolOutput | None) -> str:
    return render_received(args)
