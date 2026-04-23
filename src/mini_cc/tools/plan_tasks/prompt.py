PROMPT = """\
Use when: complex work with 3+ steps where some steps have prerequisites.
Don't use for: simple checklists with no dependencies — use plan_todos instead.

Examples:
    plan_tasks([
        {"id": "read", "description": "Read existing code"},
        {"id": "impl", "description": "Implement feature", "depends_on": ["read"]},
    ])
"""
