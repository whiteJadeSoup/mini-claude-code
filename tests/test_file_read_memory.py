"""file_read 给 memory 文件加冻结的 staleness 头（read-time 抗漂移）。"""
from mini_cc.tools.base import FileReadOutput

_NOTE = "<system-reminder>This memory is 47 days old.</system-reminder>\n"


def _out(**kw) -> FileReadOutput:
    base = dict(path="m.md", content="line1\nline2",
                total_lines=2, start_line=1, returned_lines=2)
    base.update(kw)
    return FileReadOutput(**base)


def test_to_api_str_prepends_staleness_note():
    assert _out(staleness_note=_NOTE).to_api_str() == _NOTE + "line1\nline2"


def test_to_api_str_no_note_is_plain_content():
    # 默认 staleness_note == "" → 非 memory 读行为不变
    assert _out().to_api_str() == "line1\nline2"


def test_to_api_str_note_prepended_on_empty_file():
    o = _out(content="", total_lines=0, returned_lines=0, staleness_note=_NOTE)
    assert o.to_api_str().startswith(_NOTE)
    assert "File is empty" in o.to_api_str()
