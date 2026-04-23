from mini_cc.tools.base import ToolOutput


def render_received(args: dict) -> str:
    desc = args.get("description", "")
    return (desc[:57] + "…") if len(desc) > 60 else desc


def render_complete(args: dict, output: ToolOutput | None) -> str:
    return "done"
