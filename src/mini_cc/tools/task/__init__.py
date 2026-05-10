from mini_cc import prompts
from mini_cc.tools.base import MiniTool, SubTaskOutput, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class TaskTool(MiniTool):
    name = "task"
    description = "Delegate a subtask"
    prompt = PROMPT

    async def _run(self, description: str) -> SubTaskOutput:
        # Lazy imports: same circular-dep rationale as run_skill
        from mini_cc.engine.query_engine import get_engine
        from mini_cc.engine.store import _triggering_asst_id

        parent_id = _triggering_asst_id.get()
        if parent_id is None:
            return SubTaskOutput(result="Error: task called outside a tool-dispatch context")

        result = await get_engine().run_sidechain(
            parent_id=parent_id,
            system_prompt=prompts.SUB_SYSTEM_PROMPT,
            user_prompt=description,
            label="task",
        )
        return SubTaskOutput(result=result)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(TaskTool())
