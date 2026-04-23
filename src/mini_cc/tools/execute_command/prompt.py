PROMPT = """\
Use when: running scripts, installs, git operations, reading/writing/searching files, or any shell task.
Don't use for: targeted edits to existing files — use edit_file instead.

Examples:
    execute_command("cat src/main.py")
    execute_command("grep -rn 'TODO' src/")
    execute_command("python script.py")
    execute_command("pip install requests", timeout=300)
"""
