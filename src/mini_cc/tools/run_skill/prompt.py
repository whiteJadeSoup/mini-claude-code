PROMPT = """\
Use when: the user's request matches a skill listed in the system prompt.
Don't use for: tasks not covered by available skills; don't guess skill names.

Examples:
    run_skill("review", "review the current PR")
    run_skill("init", "initialize CLAUDE.md for this project")
"""
