"""execute_command tool prompt.

Built at import time so the forbidden-use list reflects which dedicated
tools are actually available (grep/glob only when ripgrep was bundled).

Tone: aggressive restriction, not soft "use when". Empirical finding from
deepseek-v4-pro: soft directives ("Use when X. Don't use for Y.") are
routinely ignored — the model reaches for execute_command as a generic
escape hatch even with explicit fallbacks defined. Hard restriction
("RESTRICTED to ... DO NOT use for ...") performs better at the cost of
verbosity.
"""
from mini_cc import config


def _build_prompt() -> str:
    forbidden: list[tuple[str, str]] = [
        ("reading text files",          "DO NOT use cat/head/tail/type → use file_read"),
        ("creating/overwriting files",  "DO NOT use `echo >`, `tee`, `cat <<EOF` → use file_write"),
        ("editing existing files",      "DO NOT use `sed -i` or `awk -i inplace` → use file_edit"),
    ]
    examples: list[str] = []

    if config.RG_PATH:
        forbidden += [
            ("content search",  "DO NOT use grep/rg/awk for searching file contents → use grep"),
            ("filename search", "DO NOT use find/ls -R/git ls-files → use glob"),
        ]
    else:
        # Bundled rg missing → shell tools are the only fallback.
        examples.append('execute_command("grep -rn \'TODO\' src/")')
        examples.append('execute_command("find src -type f -name \'*.py\'")')

    examples += [
        'execute_command("python script.py")',
        'execute_command("pip install requests", timeout=300)',
        'execute_command("git status")',
        'execute_command("npm test")',
    ]

    forbidden_block = "\n".join(f"  - {tag}: {rule}" for tag, rule in forbidden)

    return (
        "Run a shell command. RESTRICTED to: scripts, installs, git, builds, "
        "environment inspection, package managers, and one-off shell utilities.\n"
        "\n"
        "FORBIDDEN uses (the dedicated tool exists for these — use it):\n"
        f"{forbidden_block}\n"
        "\n"
        "Before calling execute_command, ask: is the goal one of the FORBIDDEN "
        "items above? If yes, use the dedicated tool instead — execute_command "
        "will not be your fastest or cheapest path.\n"
        "\n"
        "Examples of legitimate use:\n"
        + "\n".join(f"    {ex}" for ex in examples)
        + "\n"
    )


PROMPT = _build_prompt()
