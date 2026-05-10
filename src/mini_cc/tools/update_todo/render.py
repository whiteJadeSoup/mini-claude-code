from mini_cc.tools.base import ToolOutput


def render_received(args: dict) -> str:
    item = args.get("item", "")
    status = args.get("status", "")
    short = (item[:30] + "…") if len(item) > 30 else item
    return f"{short} → {status}"


def render_complete(args: dict, output: ToolOutput | None) -> str:
    return render_received(args)
