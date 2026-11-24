"""Alt+V 从剪贴板贴截图（loongcli/tui/clipboard_image.py + TUI._paste_clipboard + 输入框的 Esc / Alt+V 区分）。"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import time

import pytest
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from loongcli.tui import clipboard_image
from loongcli.tui.app import TUI
from loongcli.tui.clipboard_image import grab_clipboard_images

from tests.tui.test_input_bar import ESC, _Rig

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _fake_grab(monkeypatch, value):
    from PIL import ImageGrab
    if isinstance(value, Exception):
        def boom():
            raise value
        monkeypatch.setattr(ImageGrab, "grabclipboard", boom)
    else:
        monkeypatch.setattr(ImageGrab, "grabclipboard", lambda: value)


def test_screenshot_saved_as_png(monkeypatch, tmp_path):
    _fake_grab(monkeypatch, Image.new("RGB", (4, 3), "red"))
    paths, hint = grab_clipboard_images(tmp_path)
    assert hint == "" and len(paths) == 1
    assert paths[0].endswith(".png") and Image.open(paths[0]).size == (4, 3)


def test_copied_files_keep_only_images(monkeypatch, tmp_path):
    png = tmp_path / "a.png"
    Image.new("RGB", (2, 2)).save(png)
    txt = tmp_path / "b.txt"
    txt.write_text("x")
    _fake_grab(monkeypatch, [str(png), str(txt)])
    assert grab_clipboard_images(tmp_path) == ([str(png)], "")
    _fake_grab(monkeypatch, [str(txt)])
    assert grab_clipboard_images(tmp_path) == ([], "剪贴板里复制的文件不是图片")


def test_empty_clipboard_and_errors_give_hints(monkeypatch, tmp_path):
    _fake_grab(monkeypatch, None)
    paths, hint = grab_clipboard_images(tmp_path)
    assert paths == [] and "没有图片" in hint
    _fake_grab(monkeypatch, OSError("xclip not found"))
    paths, hint = grab_clipboard_images(tmp_path)
    assert paths == [] and "xclip" in hint


def test_missing_pillow_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "PIL", None)            # import PIL 时抛 ImportError
    paths, hint = grab_clipboard_images(tmp_path)
    assert paths == [] and "pip install pillow" in hint


def test_old_clips_cleaned(monkeypatch, tmp_path):
    old = tmp_path / "clip-old.png"
    old.write_bytes(b"x")
    past = time.time() - clipboard_image.KEEP_SECONDS - 10
    os.utime(old, (past, past))
    _fake_grab(monkeypatch, Image.new("RGB", (1, 1)))
    grab_clipboard_images(tmp_path)
    assert not old.exists()


def test_tui_paste_inserts_placeholder_or_prints_hint(monkeypatch, tmp_path):
    from prompt_toolkit.buffer import Buffer
    tui = TUI()
    tui.console = Console(file=io.StringIO(), width=80, force_terminal=False)
    png = tmp_path / "a.png"
    Image.new("RGB", (2, 2)).save(png)
    monkeypatch.setattr("loongcli.tui.app.grab_clipboard_images", lambda: ([str(png)], ""))
    buf = Buffer()
    tui._paste_clipboard(buf)
    tui._paste_clipboard(buf)                                  # 同一张图再贴：复用占位符
    assert buf.text == "[Image #1][Image #1]" and tui._paste_images == {"[Image #1]": str(png)}
    monkeypatch.setattr("loongcli.tui.app.grab_clipboard_images", lambda: ([], "剪贴板里没有图片"))
    tui._paste_clipboard(buf)
    assert "剪贴板里没有图片" in tui.console.file.getvalue()


@pytest.mark.asyncio
async def test_alt_v_not_eaten_by_escape_and_escape_still_interrupts():
    """Alt+V 在终端里是「Esc + v」：不能被当成 Esc 中断；单按 Esc 照旧中断（Esc 曾是 eager，Alt+V 根本到不了）。"""
    hits = []
    kb = KeyBindings()

    @kb.add("escape", "v")
    def _(event):
        hits.append("alt-v")

    async with _Rig(key_bindings=kb) as r:
        await r.send("\x1bv", settle=0.3)
        assert hits == ["alt-v"] and r.interrupts == 0
        assert r.bar.session.default_buffer.text == ""             # v 没有被当成字打进去
        await r.send(ESC, settle=0.3)
        assert r.interrupts == 1
