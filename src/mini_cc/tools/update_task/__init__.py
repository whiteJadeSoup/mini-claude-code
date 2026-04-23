from mini_cc.state import tasks
from mini_cc.tools.base import MiniTool, TaskUpdateOutput, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class UpdateTaskTool(MiniTool):
    name = "update_task"
    description = "Update a task status"
    prompt = PROMPT

    async def _run(self, task_id: str, status: str) -> TaskUpdateOutput:
        rendered = tasks._tasks.update(task_id, status)
        return TaskUpdateOutput(task_id=task_id, status=status, rendered=rendered)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(UpdateTaskTool())
