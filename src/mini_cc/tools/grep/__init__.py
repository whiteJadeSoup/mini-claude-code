"""grep tool: structured ripgrep wrapper with VCS exclusion + head_limit budget.

Implements §D3 algorithm A from grep-glob-tools.md plan: sandbox path → arg
construction (VCS exclude, --max-columns 500, type/glob filters, mode flag) →
subprocess with timeout → parse rg returncode (0=hits, 1=none, 2=error) →
mode-specific output mapping with mtime sort + head_limit + relativize.
"""
import os
import subprocess
from typing import Literal

from mini_cc import config
from mini_cc.tools.base import (
    GrepOutput,
    MiniTool,
    ToolErrorOutput,
    ToolOutput,
    register,
)

from .prompt import PROMPT
from .render import render_complete, render_error, render_received

# Constants (D2 §grep)
DEFAULT_HEAD_LIMIT = 100
DEFAULT_TIMEOUT = 60                                  # OQ2 lock
MAX_COLUMNS = 500                                     # base64/minified guard
VCS_DIRS_TO_EXCLUDE = (".git", ".svn", ".hg",
                       ".bzr", ".jj", ".sl")          # Aligned with CC GrepTool.ts:95-102


class GrepTool(MiniTool):
    name = "grep"
    description = "Search file contents with regex (ripgrep)"
    prompt = PROMPT
    is_read_only = True

    async def _run(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches",
        context: int | None = None,
        show_line_numbers: bool = True,
        case_insensitive: bool = False,
        type: str | None = None,
        head_limit: int = DEFAULT_HEAD_LIMIT,
        offset: int = 0,
    ) -> ToolOutput:
        # 1. Sandbox + path validation
        if path is not None:
            try:
                p = config.safe_path(path)
            except ValueError:
                return ToolErrorOutput(message=(
                    f"Path '{path}' is outside the working directory ({config.CWD}). "
                    f"grep can only search files inside this project. "
                    f"To search system paths, use execute_command('rg ...') instead."
                ))
            if not os.path.exists(p):
                hint = _suggest_path(p, kind="any")
                return ToolErrorOutput(message=(
                    f"Path '{path}' does not exist. There are no files to search there.{hint} "
                    f"Use execute_command('ls .') to list available paths."
                ))
        else:
            p = config.CWD

        # 2. Build rg argv (G3 VCS exclude, MAX_COLUMNS, mode flags)
        rg_args: list[str] = ["--hidden", "--max-columns", str(MAX_COLUMNS)]
        for vcs in VCS_DIRS_TO_EXCLUDE:
            rg_args += ["--glob", f"!{vcs}"]
        if case_insensitive:
            rg_args += ["-i"]
        if output_mode == "files_with_matches":
            rg_args += ["-l"]
        elif output_mode == "count":
            rg_args += ["-c"]
        if show_line_numbers and output_mode == "content":
            rg_args += ["-n"]
        if context is not None and output_mode == "content":
            rg_args += ["-C", str(context)]
        if type is not None:
            rg_args += ["--type", type]
        if glob is not None:
            for pat in _split_glob_patterns(glob):
                rg_args += ["--glob", pat]
        # `-e` guards against patterns starting with `-` being interpreted as flags
        rg_args += ["-e", pattern, p]

        # 3. Spawn rg via the bundled binary (config.RG_PATH absolute path,
        # placed by hatch_build.py at install time). No PATH lookup → PATH
        # hijacking is irrelevant here; CC's bare-'rg' workaround does not
        # apply because we never resolve via $PATH.
        try:
            proc = subprocess.run(
                [config.RG_PATH, *rg_args],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ToolErrorOutput(message=(
                f"rg timed out after {DEFAULT_TIMEOUT}s searching '{path or '.'}'. "
                f"The search range may be too large. Narrow the path or pattern, "
                f"or pass head_limit explicitly to bound result size."
            ))

        # 4. Exit code dispatch (rg: 0=hits, 1=no-matches, 2=error)
        if proc.returncode == 1:
            return _empty_output_for_mode(output_mode, offset)
        if proc.returncode > 1:
            return _parse_rg_error(proc.stderr, type=type, pattern=pattern)

        # 5. Parse stdout per mode
        raw_lines = [ln for ln in proc.stdout.splitlines() if ln]

        if output_mode == "content":
            limited, applied_limit = _apply_head_limit(raw_lines, head_limit, offset)
            relativized = [_relativize_grep_line(ln) for ln in limited]
            return GrepOutput(
                mode="content",
                num_files=0,
                content="\n".join(relativized),
                applied_limit=applied_limit,
                applied_offset=offset,
            )

        if output_mode == "count":
            limited, applied_limit = _apply_head_limit(raw_lines, head_limit, offset)
            relativized = [_relativize_count_line(ln) for ln in limited]
            file_counts = [_parse_count(ln) for ln in relativized]
            return GrepOutput(
                mode="count",
                num_files=sum(1 for c in file_counts if c > 0),
                content="\n".join(relativized),
                num_matches=sum(file_counts),
                applied_limit=applied_limit,
                applied_offset=offset,
            )

        # files_with_matches: stat each, sort by mtime desc, then head_limit
        files_with_mtime: list[tuple[str, float]] = []
        for f in raw_lines:
            try:
                mt = os.stat(f).st_mtime
            except OSError:
                # B4: file deleted between rg scan and stat — sort to bottom
                mt = 0.0
            files_with_mtime.append((f, mt))
        files_with_mtime.sort(key=lambda x: (-x[1], x[0]))

        sorted_files = [f for f, _ in files_with_mtime]
        limited, applied_limit = _apply_head_limit(sorted_files, head_limit, offset)
        relative = [config.relativize(f) for f in limited]

        return GrepOutput(
            mode="files_with_matches",
            num_files=len(relative),
            filenames=relative,
            applied_limit=applied_limit,
            applied_offset=offset,
        )

    def render_received(self, args: dict) -> str:
        return render_received(args)

    def render_complete(self, args: dict, output: ToolOutput | None) -> str:
        return render_complete(args, output)

    def render_error(self, args: dict, output: ToolOutput) -> str:
        return render_error(args, output)


# ---------------------------------------------------------------------------
# Helpers (kept module-private; reused by glob via re-export)
# ---------------------------------------------------------------------------

def _split_glob_patterns(s: str) -> list[str]:
    """Accept either a single pattern or comma/whitespace-separated patterns.

    Brace groups (e.g. `*.{ts,tsx}`) are kept intact — comma inside `{...}` is
    a glob alternation, not a separator.
    """
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
            buf.append(ch)
        elif ch == "}":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif depth == 0 and ch in (",", " ", "\t"):
            if buf:
                out.append("".join(buf).strip())
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return [p for p in out if p]


def _apply_head_limit(
    items: list[str], limit: int, offset: int = 0,
) -> tuple[list[str], int | None]:
    """Apply head_limit + offset; return (sliced, applied_limit_if_truncated)."""
    if limit == 0:
        # Explicit 0 means unlimited; offset still applies.
        return items[offset:], None
    sliced = items[offset:offset + limit]
    truncated = len(items) - offset > limit
    return sliced, (limit if truncated else None)


def _relativize_grep_line(line: str) -> str:
    """Content mode line: `path:[lineno:]content` → relativize the leading path."""
    idx = line.find(":")
    if idx < 0:
        return line
    return config.relativize(line[:idx]) + line[idx:]


def _relativize_count_line(line: str) -> str:
    """Count mode line: `path:N` (path may itself contain ':' on Windows drives)
    so split on the LAST ':' rather than the first.
    """
    idx = line.rfind(":")
    if idx < 0:
        return line
    return config.relativize(line[:idx]) + line[idx:]


def _parse_count(line: str) -> int:
    idx = line.rfind(":")
    if idx < 0:
        return 0
    try:
        return int(line[idx + 1:])
    except ValueError:
        return 0


def _empty_output_for_mode(
    mode: Literal["content", "files_with_matches", "count"],
    offset: int,
) -> GrepOutput:
    return GrepOutput(
        mode=mode,
        num_files=0,
        applied_offset=offset,
    )


def _parse_rg_error(
    stderr: str,
    *,
    type: str | None,
    pattern: str,
) -> ToolErrorOutput:
    """Map rg returncode>1 stderr into actionable error messages.

    Three known classes are detected by stderr substring; everything else
    falls into a generic catch-all that still includes a recovery path.
    """
    msg = stderr.strip()
    short = msg[:500] if len(msg) > 500 else msg

    # 1. Unknown --type value (G8 specific)
    if type is not None and ("unrecognized file type" in msg or "unknown type" in msg.lower()):
        return ToolErrorOutput(message=(
            f"Unknown file type '{type}'. ripgrep does not recognize this type name. "
            f"Run execute_command('rg --type-list') to see all valid types, "
            f"or use the 'glob' parameter for arbitrary extensions."
        ))

    # 2. Invalid regex (B3 dialect class)
    if "regex parse error" in msg.lower() or "error parsing regex" in msg.lower():
        return ToolErrorOutput(message=(
            f"Invalid regex pattern '{pattern}': {short}. "
            f"Note: grep uses Rust regex syntax (not PCRE/Python). "
            f"For literal braces use \\{{ \\}}. "
            f"To search across lines, use execute_command('rg -U ...') instead."
        ))

    # 3. Generic fallback
    return ToolErrorOutput(message=(
        f"rg failed unexpectedly: {short}. "
        f"This is usually an invalid argument or rg internal error. "
        f"Two recovery paths: (1) verify your pattern syntax (Rust regex) "
        f"and re-issue grep; (2) if the cause is unclear, run the equivalent "
        f"execute_command('rg ...') to inspect the raw stderr directly."
    ))


def _suggest_path(missing: str, kind: Literal["any", "dir"] = "any") -> str:
    """Return ' Did you mean "X"?' suffix (with leading space) or empty string.

    Same-prefix similarity only (OQ1 lock) — Levenshtein not implemented.
    """
    parent = os.path.dirname(missing) or config.CWD
    base = os.path.basename(missing)
    if not base or not os.path.isdir(parent):
        return ""

    candidates: list[tuple[int, str]] = []
    try:
        entries = os.listdir(parent)
    except OSError:
        return ""

    for entry in entries:
        if kind == "dir" and not os.path.isdir(os.path.join(parent, entry)):
            continue
        common = _common_prefix_len(base.lower(), entry.lower())
        if common >= max(2, len(base) // 2):
            candidates.append((common, entry))

    if not candidates:
        return ""
    candidates.sort(key=lambda x: -x[0])
    best = candidates[0][1]
    suggested = os.path.join(os.path.dirname(missing), best)
    return f' Did you mean "{config.relativize(suggested)}"?'


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


# Conditional registration: rg is required at registration time, not at
# runtime, because the LangChain bridge captures the tool list once when
# llm.py loads. tools/__init__.py also gates the import on config.RG_PATH,
# so this `if` is belt-and-suspenders for direct imports.
if config.RG_PATH:
    register(GrepTool())
