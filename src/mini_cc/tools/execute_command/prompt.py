"""execute_command tool prompt.

Built at import time so the "Don't use for" reverse list reflects which
dedicated tools are actually available (grep/glob only when ripgrep is on
PATH). Static text would either lie about grep/glob existing in a no-rg
install, or under-sell the priority of grep/glob in a rg-installed install.
"""
from mini_cc import config


def _build_prompt() -> str:
    # "Don't use for" — list grows when dedicated tools are available.
    dont_use_for: list[str] = [
        "reading text files (use file_read)",
        "creating/overwriting files (use file_write)",
        "targeted edits to existing files (use file_edit)",
    ]
    examples: list[str] = []

    if config.RG_PATH:
        dont_use_for += [
            "content search via grep/rg (use grep)",
            "filename search via find/ls -R/git ls-files (use glob)",
        ]
    else:
        # Bundled rg missing → reinstall is the proper fix, but until then
        # `find` / `grep` via the shell are the only search tools the LLM has.
        examples.append('execute_command("grep -rn \'TODO\' src/")')
        examples.append('execute_command("find src -type f -name \'*.py\'")')

    examples += [
        'execute_command("python script.py")',
        'execute_command("pip install requests", timeout=300)',
        'execute_command("git status")',
    ]

    return (
        "Use when: running scripts, installs, git operations, or any shell task "
        "not covered by a dedicated tool.\n"
        f"Don't use for: {', '.join(dont_use_for)}.\n"
        "\n"
        "Examples:\n"
        + "\n".join(f"    {ex}" for ex in examples)
        + "\n"
    )


PROMPT = _build_prompt()
