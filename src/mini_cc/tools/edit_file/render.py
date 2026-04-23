from mini_cc.tools.base import FileEditOutput, ToolOutput


def render_received(args: dict) -> str:
    return args.get("path", "")


def render_complete(args: dict, output: ToolOutput | None) -> str:
    path = args.get("path", "")
    if not isinstance(output, FileEditOutput):
        return path
    return f"{path} · edited" if output.replaced else f"{path} · not found"
