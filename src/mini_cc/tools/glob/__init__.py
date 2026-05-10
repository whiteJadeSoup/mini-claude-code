"""glob tool: structured `rg --files --glob <pattern>` wrapper.

Implements §D3 algorithm B from grep-glob-tools.md plan: sandbox + directory
validation → spawn `rg --files --glob <pat>` with VCS exclude → mtime sort →
GLOB_CAP=100 truncation → relativize.
"""
import os
import subprocess
import time

from mini_cc import config
from mini_cc.tools.base import (
    GlobOutput,
    MiniTool,
    ToolErrorOutput,
    ToolOutput,
    register,
)
from mini_cc.tools.grep import (
    DEFAULT_TIMEOUT,
    VCS_DIRS_TO_EXCLUDE,
    _suggest_path,
)

from .prompt import PROMPT
from .render import render_complete, render_error, render_received

GLOB_CAP = 100


class GlobTool(MiniTool):
    name = "glob"
    description = "Find files by name pattern (ripgrep)"
    prompt = PROMPT

    async def _run(
        self,
        pattern: str,
        path: str | None = None,
    ) -> ToolOutput:
        # 1. Sandbox + directory check (OQ4: path must be a directory)
        if path is not None:
            try:
                search_dir = config.safe_path(path)
            except ValueError:
                return ToolErrorOutput(message=(
                    f"Path '{path}' is outside the working directory ({config.CWD}). "
                    f"glob can only search inside this project. "
                    f"To search system paths, use execute_command('rg --files ...') instead."
                ))
            if not os.path.exists(search_dir):
                hint = _suggest_path(search_dir, kind="dir")
                return ToolErrorOutput(message=(
                    f"Directory '{path}' does not exist. There are no files to enumerate.{hint} "
                    f"Use execute_command('ls .') to list available directories."
                ))
            if not os.path.isdir(search_dir):
                return ToolErrorOutput(message=(
                    f"Path '{path}' is a file, not a directory. "
                    f"glob searches inside directories. "
                    f"Pass the parent directory as path, "
                    f"or use file_read('{path}') to read the file directly."
                ))
        else:
            search_dir = config.CWD

        # 2. Build argv: rg --files --glob <pat> (+ VCS excludes, --hidden)
        rg_args: list[str] = ["--files", "--hidden"]
        for vcs in VCS_DIRS_TO_EXCLUDE:
            rg_args += ["--glob", f"!{vcs}"]
        rg_args += ["--glob", pattern, search_dir]

        # 3. Spawn rg
        start = time.monotonic()
        try:
            proc = subprocess.run(
                ["rg", *rg_args],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ToolErrorOutput(message=(
                f"rg timed out after {DEFAULT_TIMEOUT}s enumerating '{path or '.'}'. "
                f"The directory may be too large. Narrow the path or pattern."
            ))
        duration_ms = int((time.monotonic() - start) * 1000)

        # 4. rg --files: 0=ok (possibly empty), 2=error. Treat returncode==1 as
        # "no matches" defensively (some rg versions return 1 here, others 0).
        if proc.returncode > 1:
            stderr = proc.stderr.strip()
            short = stderr[:500] if len(stderr) > 500 else stderr
            return ToolErrorOutput(message=(
                f"Invalid glob pattern '{pattern}': {short}. "
                f"Glob patterns use shell syntax — e.g., '**/*.py' for recursive "
                f"Python files, 'src/**/*.{{ts,tsx}}' for nested TS/TSX. "
                f"For content matching (not filename), use grep with a regex pattern instead."
            ))

        # 5. mtime sort + cap
        raw_files = [ln for ln in proc.stdout.splitlines() if ln]
        files_with_mtime: list[tuple[str, float]] = []
        for f in raw_files:
            try:
                mt = os.stat(f).st_mtime
            except OSError:
                mt = 0.0
            files_with_mtime.append((f, mt))
        files_with_mtime.sort(key=lambda x: (-x[1], x[0]))

        truncated = len(files_with_mtime) > GLOB_CAP
        capped = files_with_mtime[:GLOB_CAP]
        relative = [config.relativize(f) for f, _ in capped]

        return GlobOutput(
            filenames=relative,
            num_files=len(relative),
            truncated=truncated,
            duration_ms=duration_ms,
        )

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)

    def render_error(self, args: dict, output: ToolOutput) -> str:
        return render_error(args, output)


if config.RG_PATH:
    register(GlobTool())
