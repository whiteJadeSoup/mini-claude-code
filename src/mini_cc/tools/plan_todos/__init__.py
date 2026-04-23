from mini_cc.state import todos
from mini_cc.tools.base import MiniTool, TodoPlanOutput, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class PlanTodosTool(MiniTool):
    name = "plan_todos"
    description = "Set the TODO list"
    prompt = PROMPT

    async def _run(self, items: list[str]) -> TodoPlanOutput:
        rendered = todos._todos.plan(items)
        return TodoPlanOutput(count=len(items), rendered=rendered)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(PlanTodosTool())
