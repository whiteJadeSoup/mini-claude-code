from mini_cc.state import todos
from mini_cc.tools.base import MiniTool, TodoUpdateOutput, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class UpdateTodoTool(MiniTool):
    name = "update_todo"
    description = "Update a TODO item status"
    prompt = PROMPT

    async def _run(self, item: str, status: str) -> TodoUpdateOutput:
        rendered = todos._todos.update(item, status)
        return TodoUpdateOutput(item=item, status=status, rendered=rendered)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(UpdateTodoTool())
