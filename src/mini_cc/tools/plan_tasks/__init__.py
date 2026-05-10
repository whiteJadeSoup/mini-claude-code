from mini_cc.state import tasks
from mini_cc.tools.base import MiniTool, TaskPlanOutput, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class PlanTasksTool(MiniTool):
    name = "plan_tasks"
    description = "Set a dependency-aware task graph"
    prompt = PROMPT

    async def _run(self, tasks_list: list[dict]) -> TaskPlanOutput:
        rendered = tasks._tasks.plan(tasks_list)
        return TaskPlanOutput(count=len(tasks_list), rendered=rendered)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(PlanTasksTool())
