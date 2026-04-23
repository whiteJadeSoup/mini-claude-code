from mini_cc import config
from mini_cc.tools.base import FileEditOutput, MiniTool, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class EditFileTool(MiniTool):
    name = "edit_file"
    description = "Edit a file in place"
    prompt = PROMPT

    async def _run(self, path: str, old_string: str, new_string: str) -> FileEditOutput:
        p = config.safe_path(path)
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        if old_string not in content:
            return FileEditOutput(path=path, replaced=False)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content.replace(old_string, new_string, 1))
        return FileEditOutput(path=path, replaced=True)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(EditFileTool())
