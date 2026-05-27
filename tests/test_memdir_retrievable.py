"""可取 (retrievable) attribute: 双通道 messages[0] 注入 + 方案C 合并."""
from mini_cc.memdir.truncate import (
    truncate_entrypoint,
    MAX_ENTRYPOINT_LINES,
    MAX_ENTRYPOINT_BYTES,
)


def test_truncate_under_caps_returns_trimmed():
    assert truncate_entrypoint("- a\n- b\n") == "- a\n- b"


def test_truncate_over_line_cap_appends_warning():
    raw = "\n".join(f"- line{i}" for i in range(MAX_ENTRYPOINT_LINES + 50))
    out = truncate_entrypoint(raw)
    # Partition on the full "\n\n> WARNING" separator so the blank line isn't
    # counted as content; `content` is then exactly the truncated index.
    content, sep, warning = out.partition("\n\n> WARNING")
    assert sep  # warning block present, with its blank-line separator
    assert content.count("\n") + 1 <= MAX_ENTRYPOINT_LINES  # N lines = N-1 newlines
    assert str(MAX_ENTRYPOINT_LINES) in warning


def test_truncate_over_byte_cap_under_line_cap():
    # 10 long lines blow the 25KB byte cap while staying under 200 lines.
    raw = "\n".join("x" * 4000 for _ in range(10))
    out = truncate_entrypoint(raw)
    _, sep, warning = out.partition("\n\n> WARNING")
    assert sep
    assert "bytes" in warning  # byte-only branch names the byte cap
    assert len(out.encode("utf-8")) < 26_000


def test_truncate_both_caps_reports_source_size():
    # 250 lines × 120 chars: over BOTH the line cap and the (original) byte cap.
    # CC-faithful: the warning reports the SOURCE size ("lines and bytes") even
    # though line-truncation alone already brings the loaded content under 25KB.
    raw = "\n".join("x" * 120 for _ in range(MAX_ENTRYPOINT_LINES + 50))
    out = truncate_entrypoint(raw)
    content, sep, warning = out.partition("\n\n> WARNING")
    assert sep
    assert "lines and" in warning and "bytes" in warning
    assert content.count("\n") + 1 <= MAX_ENTRYPOINT_LINES


def test_truncate_single_huge_line_no_newline():
    # One line over 25KB with no newline in the window: best-effort byte cut
    # (the docstring's documented exception to "never cut mid-way").
    raw = "y" * 40_000
    out = truncate_entrypoint(raw)
    content, sep, warning = out.partition("\n\n> WARNING")
    assert sep
    assert "bytes" in warning
    assert len(content.encode("utf-8")) <= MAX_ENTRYPOINT_BYTES
