from mini_cc.tools.base import FileWriteOutput, ToolOutput


def _size_str(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.1f} KB"


def render_received(args: dict) -> str:
    return args.get("path", "")


def render_complete(args: dict, output: ToolOutput | None) -> str:
    path = args.get("path", "")
    if not isinstance(output, FileWriteOutput):
        return path
    return f"{path} · {_size_str(output.bytes_written)}"
