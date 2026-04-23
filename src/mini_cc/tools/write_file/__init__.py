import os

from mini_cc import config
from mini_cc.tools.base import FileWriteOutput, MiniTool, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class WriteFileTool(MiniTool):
    name = "write_file"
    description = "Create or overwrite a file"
    prompt = PROMPT

    async def _run(self, path: str, content: str) -> FileWriteOutput:
        p = config.safe_path(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        return FileWriteOutput(path=path, bytes_written=len(content.encode("utf-8")))

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(WriteFileTool())
