import asyncio
import locale
import os
import re
import subprocess
import sys

from mini_cc import config
from mini_cc.tools.base import CommandOutput, MiniTool, ToolOutput, register
from .prompt import PROMPT
from .render import render_received, render_complete

_SENSITIVE_PATTERNS = [
    r'\.env\b',
    r'credentials\.json\b',
    r'\.pem\b',
    r'id_rsa\b',
    r'id_ed25519\b',
]
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.I)


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), 9)
        except (ProcessLookupError, OSError):
            pass


class ExecuteCommandTool(MiniTool):
    name = "execute_command"
    description = "Run a shell command"
    prompt = PROMPT

    async def _run(self, command: str, timeout: int = 120) -> CommandOutput:
        if _SENSITIVE_RE.search(command):
            return CommandOutput(
                stdout="Error: command references a sensitive file and was blocked.",
                returncode=1,
            )
        try:
            if config.BASH_PATH:
                proc = await asyncio.create_subprocess_exec(
                    config.BASH_PATH, "--login", "-c", command,
                    cwd=config.CWD,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=config.CWD,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
        except OSError as e:
            return CommandOutput(stdout=f"Error: {e}", returncode=1)

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            _kill_tree(proc.pid)
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                # Process already dead after _kill_tree; ignore to preserve the
                # original TimeoutError context for the returned CommandOutput.
                pass
            return CommandOutput(stdout=f"Error: command timed out after {timeout}s", returncode=1)
        except BaseException:
            # CancelledError or unexpected — kill then re-raise so execute() can catch
            _kill_tree(proc.pid)
            try:
                proc.kill()
            except Exception:
                pass
            raise

        raw = (stdout_b or b"") + (stderr_b or b"")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode(locale.getpreferredencoding(False), errors="replace")
        return CommandOutput(stdout=text.strip() or "(no output)", returncode=proc.returncode)

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)


register(ExecuteCommandTool())
