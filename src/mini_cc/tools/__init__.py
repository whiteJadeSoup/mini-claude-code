# Import all tool packages so their register() calls run on first import.
from mini_cc.tools import (  # noqa: F401
    execute_command,
    file_edit,
    file_read,
    file_write,
    plan_tasks,
    plan_todos,
    run_skill,
    task,
    update_task,
    update_todo,
)
