from mini_cc.tools.base import GlobOutput, ToolErrorOutput, ToolOutput


def render_received(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path")
    return f"{pattern} in {path}" if path else pattern


def render_complete(args: dict, output: ToolOutput | None) -> str:
    base = render_received(args)
    if not isinstance(output, GlobOutput):
        return base
    if output.num_files == 0:
        return f"{base} · no matches"
    word = "file" if output.num_files == 1 else "files"
    suffix = f"{output.num_files} {word}"
    if output.truncated:
        suffix += " (truncated)"
    return f"{base} · {suffix} ({output.duration_ms}ms)"


def render_error(args: dict, output: ToolOutput) -> str:
    base = render_received(args)
    msg = output.message if isinstance(output, ToolErrorOutput) else "error"
    short = (msg[:40] + "…") if len(msg) > 40 else msg
    return f"{base} · {short}"
