"""file_read 给 memory 文件加冻结的 staleness 头（read-time 抗漂移）。"""
import os
import time
from pathlib import Path

from mini_cc import config
from mini_cc.tools.base import FileReadOutput
from mini_cc.tools.file_read import FileReadTool

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


_DAY = 86_400


async def _read(path) -> FileReadOutput:
    return await FileReadTool()._run(path=str(path))


def _point_memdir_at(monkeypatch, memdir: Path) -> None:
    # safe_path 和 file_read 都在调用时 `from mini_cc.memdir import get_auto_mem_path`，
    # 所以 patch 模块属性两边都覆盖。realpath 以匹配 safe_path 的 realpath'd resolved。
    real = Path(os.path.realpath(memdir))
    monkeypatch.setattr("mini_cc.memdir.get_auto_mem_path", lambda: real)


async def test_memory_read_gets_frozen_staleness_note(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CWD", os.path.realpath(tmp_path))
    memdir = tmp_path / "mem"; memdir.mkdir()
    _point_memdir_at(monkeypatch, memdir)
    f = memdir / "user_role.md"
    f.write_text("likes uv\n", encoding="utf-8")
    os.utime(f, (time.time(), time.time() - 47 * _DAY))     # 47 天前
    out = await _read(f)
    assert isinstance(out, FileReadOutput)
    assert "47 days old" in out.staleness_note
    api = out.to_api_str()
    assert api.startswith("<system-reminder>") and "likes uv" in api


async def test_fresh_memory_read_has_no_note(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CWD", os.path.realpath(tmp_path))
    memdir = tmp_path / "mem"; memdir.mkdir()
    _point_memdir_at(monkeypatch, memdir)
    f = memdir / "fresh.md"; f.write_text("x\n", encoding="utf-8")   # mtime ~= now → <=1天
    out = await _read(f)
    assert out.staleness_note == ""


async def test_non_memory_read_has_no_note(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CWD", os.path.realpath(tmp_path))
    memdir = tmp_path / "mem"; memdir.mkdir()
    _point_memdir_at(monkeypatch, memdir)
    f = tmp_path / "code.py"; f.write_text("print(1)\n", encoding="utf-8")  # 在 CWD 下、非 memdir
    os.utime(f, (time.time(), time.time() - 47 * _DAY))
    out = await _read(f)
    assert isinstance(out, FileReadOutput)                  # safe_path 放行(在 CWD 下)
    assert out.staleness_note == ""                         # 但非 memory → 无 note


async def test_cache_hit_read_also_carries_note(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CWD", os.path.realpath(tmp_path))
    memdir = tmp_path / "mem"; memdir.mkdir()
    _point_memdir_at(monkeypatch, memdir)
    f = memdir / "user_role.md"
    f.write_text("likes uv\n", encoding="utf-8")
    os.utime(f, (time.time(), time.time() - 47 * _DAY))
    await _read(f)                                          # 第一次 → 记 dedup entry
    out2 = await _read(f)                                   # 第二次 → cache 命中路径
    assert out2.unchanged is True
    assert "47 days old" in out2.staleness_note             # cache 命中路径也冻结
