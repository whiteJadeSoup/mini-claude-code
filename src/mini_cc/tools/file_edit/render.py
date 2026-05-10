from mini_cc.tools.base import FileEditOutput, ToolOutput


def render_received(args: dict) -> str:
    return args.get("path", "")


def render_complete(args: dict, output: ToolOutput | None) -> str:
    path = args.get("path", "")
    if isinstance(output, FileEditOutput) and output.replaced:
        if output.replace_count > 1:
            return f"{path} · edited ({output.replace_count})"
        return f"{path} · edited"
    return path
