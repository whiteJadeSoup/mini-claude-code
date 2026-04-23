PROMPT = """\
Use when: making a targeted edit to an existing file (fix a bug, rename, update a value).
Don't use for: creating new files or rewriting entire content — use write_file instead.

Examples:
    edit_file("app.py", "def foo():", "def bar():")
    edit_file("config.json", '"debug": false', '"debug": true')
"""
