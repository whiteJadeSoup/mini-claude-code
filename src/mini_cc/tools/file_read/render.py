from mini_cc.tools.base import FileReadOutput, ToolOutput


def render_received(args: dict) -> str:
    path = args.get("path", "")
    offset = args.get("offset")
    limit = args.get("limit")
    if offset is not None and offset != 1:
        return f"{path} @{offset}"
    if limit is not None and limit != 2000:
        return f"{path} (limit={limit})"
    return path


def render_complete(args: dict, output: ToolOutput | None) -> str:
    path = args.get("path", "")
    if not isinstance(output, FileReadOutput):
        return path
    if output.unchanged:
        return f"{path} · unchanged"
    if output.total_lines == 0:
        return f"{path} · empty"
    if output.returned_lines == 0:
        return f"{path} · offset beyond end"
    if output.truncated_by_limit:
        end = output.start_line + output.returned_lines - 1
        return f"{path} · {output.start_line}-{end} of {output.total_lines}"
    return f"{path} · {output.returned_lines} lines"
