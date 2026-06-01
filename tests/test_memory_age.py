"""age.py：human-readable age + 固化 freshness header（port CC memoryAge.ts）。"""
import time

from mini_cc.memdir.age import (
    memory_age, memory_age_days, memory_freshness_note, memory_freshness_text,
    memory_header,
)

_DAY = 86_400_000


def _ago(days: int) -> int:
    return int(time.time() * 1000) - days * _DAY


def test_age_words():
    assert memory_age(_ago(0)) == "today"
    assert memory_age(_ago(1)) == "yesterday"
    assert memory_age(_ago(47)) == "47 days ago"


def test_freshness_text_empty_when_fresh():
    assert memory_freshness_text(_ago(0)) == ""
    assert memory_freshness_text(_ago(1)) == ""


def test_freshness_text_present_when_stale():
    t = memory_freshness_text(_ago(47))
    assert "47 days old" in t and "Verify against current code" in t


def test_header_fresh_has_saved_today():
    assert memory_header("user_role.md", _ago(0)) == "Memory (saved today): user_role.md:"


def test_header_stale_has_caveat_and_name():
    h = memory_header("x.md", _ago(47))
    assert h.startswith("This memory is 47 days old.")
    assert h.endswith("Memory: x.md:")


def test_age_days_never_negative():
    assert memory_age_days(int(time.time() * 1000) + 10 * _DAY) == 0


def test_memory_freshness_note_wraps_when_stale():
    note = memory_freshness_note(_ago(47))
    assert note.startswith("<system-reminder>")
    assert note.rstrip().endswith("</system-reminder>")
    assert "47 days old" in note and "Verify against current code" in note


def test_memory_freshness_note_empty_when_fresh():
    # <=1 day → no note（对新 memory 警告是噪声）
    assert memory_freshness_note(_ago(0)) == ""
    assert memory_freshness_note(_ago(1)) == ""
