PROMPT = """\
Use when: running scripts, installs, git operations, content search, or any shell task.
Don't use for: reading text files (use file_read), creating/overwriting files (use file_write), targeted edits to existing files (use file_edit).

Examples:
    execute_command("grep -rn 'TODO' src/")
    execute_command("python script.py")
    execute_command("pip install requests", timeout=300)
    execute_command("git status")
"""
