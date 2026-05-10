from mini_cc.tools.base import TaskPlanOutput, ToolOutput


def render_received(args: dict) -> str:
    tlist = args.get("tasks_list", [])
    return f"{len(tlist)} tasks"


def render_complete(args: dict, output: ToolOutput | None) -> str:
    if not isinstance(output, TaskPlanOutput):
        return render_received(args)
    return f"{output.count} tasks"
