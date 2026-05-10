from mini_cc.tools.base import GrepOutput, ToolErrorOutput, ToolOutput


def render_received(args: dict) -> str:
    pattern = args.get("pattern", "")
    path = args.get("path")
    type_ = args.get("type")
    mode = args.get("output_mode", "files_with_matches")

    parts = [f'"{pattern}"']
    if path:
        parts.append(f"in {path}")
    if type_:
        parts.append(f"(.{type_})")
    if mode == "content":
        parts.append("→ content")
    elif mode == "count":
        parts.append("→ count")
    return " ".join(parts)


def render_complete(args: dict, output: ToolOutput | None) -> str:
    base = render_received(args)
    if not isinstance(output, GrepOutput):
        return base

    if output.mode == "files_with_matches":
        if output.num_files == 0:
            return f"{base} · no matches"
        word = "file" if output.num_files == 1 else "files"
        suffix = f"{output.num_files} {word}"
        if output.applied_limit is not None:
            suffix += f" (limit {output.applied_limit})"
        return f"{base} · {suffix}"

    if output.mode == "count":
        if output.num_matches == 0:
            return f"{base} · no matches"
        return f"{base} · {output.num_matches} matches in {output.num_files} files"

    # content
    if not output.content:
        return f"{base} · no matches"
    lines = output.content.count("\n") + 1
    suffix = f"{lines} lines"
    if output.applied_limit is not None:
        suffix += f" (limit {output.applied_limit})"
    return f"{base} · {suffix}"


def render_error(args: dict, output: ToolOutput) -> str:
    base = render_received(args)
    msg = output.message if isinstance(output, ToolErrorOutput) else "error"
    short = (msg[:40] + "…") if len(msg) > 40 else msg
    return f"{base} · {short}"
