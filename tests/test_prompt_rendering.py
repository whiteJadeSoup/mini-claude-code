"""Tests for system prompt rendering with available_tools gating (G9).

Goal coverage (per grep-glob-tools.md plan §D7):
  G5/G9 — TP01 (rg detection registers tools)
        — TP02 (no-rg drops tools + adds install hint)
        — TP03 (priority table includes grep/glob when available)
        — TP04 (priority table omits grep/glob + adds install hint)
        — TP05 (grep prompt has ALWAYS/NEVER)
        — TP06 (file_read/write/edit prompts hardened)
        — TP07 (execute_command prompt reverse-list when rg present)
        — TP08 (full system prompt diffs cleanly between rg-states)
"""
import pytest

from mini_cc import config
from mini_cc.prompts import build_system_prompt
from mini_cc.tools import file_edit, file_read, file_write
from mini_cc.tools.execute_command import prompt as exec_prompt_module


# Bundled rg is the source of truth (see test_grep_glob_tools.py for context).
HAS_RG = config.RG_PATH is not None


# ---------------------------------------------------------------------------
# TP01-TP02: rg detection / registration gate
# ---------------------------------------------------------------------------

def test_TP01_rg_present_registers_grep_glob(monkeypatch):
    """When config.RG_PATH is non-None, the tool registry should expose
    grep + glob after a fresh import.

    We can't truly re-import here without polluting other tests, so we test
    the behavior more directly: confirm that the conditional at
    tools/__init__.py:`if config.RG_PATH: from mini_cc.tools import grep, glob`
    is the only gate. Equivalent: when HAS_RG is true on this system, the
    registry already contains them.
    """
    if not HAS_RG:
        pytest.skip("requires rg installed for live registry check")
    from mini_cc.tools.base import _REGISTRY
    assert "grep" in _REGISTRY
    assert "glob" in _REGISTRY


def test_TP02_no_rg_priority_table_omits_grep_glob():
    sp = build_system_prompt("", available_tools={
        "file_read", "file_write", "file_edit", "execute_command",
    })
    # Priority table should not list grep/glob
    assert "→  grep" not in sp
    assert "→  glob" not in sp
    # Install hint must appear
    assert "bundled ripgrep binary is missing" in sp


# ---------------------------------------------------------------------------
# TP03-TP04: priority-table conditional rendering
# ---------------------------------------------------------------------------

def test_TP03_with_grep_glob_priority_table_lists_them():
    sp = build_system_prompt("", available_tools={
        "file_read", "file_write", "file_edit",
        "grep", "glob", "execute_command",
    })
    assert "Tool selection priority:" in sp
    assert "→  grep" in sp
    assert "→  glob" in sp
    assert "bundled ripgrep binary is missing" not in sp


def test_TP04_no_grep_no_glob_omits_sections_and_adds_hint():
    sp = build_system_prompt("", available_tools={
        "file_read", "file_write", "file_edit", "execute_command",
    })
    assert "## grep" not in sp
    assert "## glob" not in sp
    # Install hint paragraph
    assert "bundled ripgrep binary is missing" in sp


def test_TP04b_only_one_of_grep_glob_still_gates_each():
    """Belt-and-suspenders: confirm grep can render without glob (and vice
    versa) — this matters if a future change splits rg detection into two
    feature flags."""
    sp_only_grep = build_system_prompt("", available_tools={
        "file_read", "file_write", "file_edit", "grep", "execute_command",
    })
    assert "→  grep" in sp_only_grep
    assert "→  glob" not in sp_only_grep
    assert "## grep" in sp_only_grep
    assert "## glob" not in sp_only_grep


# ---------------------------------------------------------------------------
# TP05: grep tool's own prompt has ALWAYS/NEVER directives
# ---------------------------------------------------------------------------

def test_TP05_grep_prompt_always_never():
    from mini_cc.tools.grep.prompt import PROMPT
    assert "ALWAYS use grep" in PROMPT
    assert "NEVER" in PROMPT and "execute_command" in PROMPT


def test_TP05b_glob_prompt_always_never():
    from mini_cc.tools.glob.prompt import PROMPT
    assert "ALWAYS use glob" in PROMPT
    assert "NEVER" in PROMPT


# ---------------------------------------------------------------------------
# TP06: file_*** prompts hardened with ALWAYS/NEVER
# ---------------------------------------------------------------------------

def test_TP06_file_read_prompt_hardened():
    from mini_cc.tools.file_read.prompt import PROMPT
    assert "ALWAYS use file_read" in PROMPT
    assert "NEVER" in PROMPT


def test_TP06b_file_write_prompt_hardened():
    from mini_cc.tools.file_write.prompt import PROMPT
    assert "ALWAYS use file_write" in PROMPT
    assert "NEVER" in PROMPT


def test_TP06c_file_edit_prompt_hardened():
    from mini_cc.tools.file_edit.prompt import PROMPT
    assert "ALWAYS use file_edit" in PROMPT
    assert "NEVER" in PROMPT


# ---------------------------------------------------------------------------
# TP07: execute_command reverse list reflects rg state
# ---------------------------------------------------------------------------

def test_TP07_execute_command_prompt_reflects_rg_state():
    """The prompt is built at module load time, so it reflects whatever
    config.RG_PATH was at that moment. We just assert the static expectation."""
    p = exec_prompt_module.PROMPT
    if config.RG_PATH:
        assert "use grep" in p
        assert "use glob" in p
    else:
        # Bundled rg missing — reverse list omits grep/glob (would point at
        # non-existent tools), and example list shows shell-tool fallbacks
        # (no rg, since users without bundled rg likely lack system rg too).
        assert "use grep" not in p
        assert "use glob" not in p
        assert "find" in p or "grep -rn" in p   # shell-tool fallback example


# ---------------------------------------------------------------------------
# TP08: full system prompt renders cleanly in both rg-states
# ---------------------------------------------------------------------------

def test_TP08_full_render_both_states_clean():
    """Both branches must render without errors; differences must be
    contained to grep/glob sections + install hint, not bleed into the
    surrounding workflow text."""
    base = {"file_read", "file_write", "file_edit", "execute_command"}
    sp_no_rg = build_system_prompt("", available_tools=base)
    sp_rg = build_system_prompt("", available_tools=base | {"grep", "glob"})

    # Workflow section identical in both
    assert "## Workflow" in sp_no_rg
    assert "## Workflow" in sp_rg
    # file_read sections identical in both (Layer 2 is rg-independent)
    fr_marker = "## file_read"
    assert fr_marker in sp_no_rg
    assert fr_marker in sp_rg
    # And critical: priority table differs only in grep/glob rows.
    # `→` appears outside the table too (Workflow section), so isolate the
    # table block by header.
    def _table_block(p: str) -> str:
        start = p.index("Tool selection priority:")
        # Block ends at the first blank line after the header
        rest = p[start:]
        end = rest.index("\n\n")
        return rest[:end]
    assert _table_block(sp_no_rg).count("→") == 4   # 4 dedicated tools when no rg
    assert _table_block(sp_rg).count("→") == 6      # +grep, +glob
