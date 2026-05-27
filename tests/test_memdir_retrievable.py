"""可取 (retrievable) attribute: 双通道 messages[0] 注入 + 方案C 合并."""
from mini_cc.memdir.truncate import truncate_entrypoint, MAX_ENTRYPOINT_LINES


def test_truncate_under_caps_returns_trimmed():
    assert truncate_entrypoint("- a\n- b\n") == "- a\n- b"


def test_truncate_over_line_cap_appends_warning():
    raw = "\n".join(f"- line{i}" for i in range(MAX_ENTRYPOINT_LINES + 50))
    out = truncate_entrypoint(raw)
    body, _, warning = out.partition("> WARNING")
    assert body.count("\n") <= MAX_ENTRYPOINT_LINES
    assert warning  # warning present
    assert str(MAX_ENTRYPOINT_LINES) in warning


def test_truncate_over_byte_cap_under_line_cap():
    # 10 long lines blow the 25KB byte cap while staying under 200 lines.
    raw = "\n".join("x" * 4000 for _ in range(10))
    out = truncate_entrypoint(raw)
    assert "> WARNING" in out
    assert len(out.encode("utf-8")) < 26_000
