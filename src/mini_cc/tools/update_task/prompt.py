PROMPT = """\
Use when: marking a task in_progress before starting it, or done after completing it.
Don't use for: adding tasks — use plan_tasks to set the full graph upfront.

Note: starting a task whose dependencies aren't done will return an error.

Examples:
    update_task("read", "in_progress")
    update_task("read", "done")
"""
