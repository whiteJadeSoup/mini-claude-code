from mini_cc.tools.base import ToolOutput


def render_received(args: dict) -> str:
    skill = args.get("name", "")
    req = args.get("request", "")
    short_req = (req[:40] + "…") if len(req) > 40 else req
    return f"{skill}: {short_req}" if short_req else skill


def render_complete(args: dict, output: ToolOutput | None) -> str:
    return f"{args.get('name', '')} · done"
