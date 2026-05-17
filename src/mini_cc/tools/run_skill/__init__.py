from mini_cc import skills
from mini_cc.tools.base import MiniTool, RunSkillOutput, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete


class RunSkillTool(MiniTool):
    name = "run_skill"
    description = "Execute a skill"
    prompt = PROMPT

    async def _run(self, name: str, request: str, context: str = "") -> RunSkillOutput:
        body = skills._skill_manager.body(name)
        if body is None:
            available = ", ".join(skills._skill_manager.names()) or "(none)"
            return RunSkillOutput(
                skill_name=name,
                result=f"Skill '{name}' not found. Available: {available}",
            )

        # Lazy imports: query_engine imports tools indirectly via tool lists;
        # importing at call time avoids circular deps at module load.
        from mini_cc.engine.query_engine import get_engine
        from mini_cc.engine.sandbox import SUB_AGENT_SANDBOX
        from mini_cc.engine.store import _triggering_asst_id

        parent_id = _triggering_asst_id.get()
        if parent_id is None:
            return RunSkillOutput(
                skill_name=name,
                result="Error: run_skill called outside a tool-dispatch context",
            )

        user_content = f"{request}\n\n## Context\n{context}" if context else request
        result = await get_engine().run_sidechain(
            parent_id=parent_id,
            system_prompt=body,
            user_prompt=user_content,
            label=f"skill:{name}",
            sandbox=SUB_AGENT_SANDBOX,
        )
        return RunSkillOutput(skill_name=name, result=result)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(RunSkillTool())
