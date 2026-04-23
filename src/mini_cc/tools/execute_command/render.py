from mini_cc.tools.base import CommandOutput, ToolOutput


def render_received(args: dict) -> str:
    cmd = args.get("command", "")
    return (cmd[:57] + "…") if len(cmd) > 60 else cmd


def render_complete(args: dict, output: ToolOutput | None) -> str:
    base = render_received(args)
    if not isinstance(output, CommandOutput):
        return base
    lines = [ln for ln in output.stdout.splitlines() if ln.strip()]
    if output.returncode != 0:
        return f"{base} · exit {output.returncode}"
    return f"{base} · {len(lines)} lines"
