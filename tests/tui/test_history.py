import pytest
from io import StringIO

from rich.console import Console

from loongcli.tui.app import TUI


def test_render_history_basic():
    tui = TUI()
    tui.console = Console(file=StringIO(), force_terminal=True)

    messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "what is 1+1?"},
        {"role": "assistant", "content": "2"},
    ]
    tui.render_history(messages)

    tui.console.file.seek(0)
    output = tui.console.file.read()
    assert "hello" in output
    assert "hi there" in output
    assert "what is" in output


def test_render_history_with_image_message_no_crash():
    """含图片的多模态 user 消息不能让 render_history 崩（list content 回归）。"""
    tui = TUI()
    tui.console = Console(file=StringIO(), force_terminal=True)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "看这个截图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,X"}},
        ]},
        {"role": "assistant", "content": "这是一个界面"},
    ]
    tui.render_history(messages)  # 不抛异常

    tui.console.file.seek(0)
    output = tui.console.file.read()
    assert "看这个截图" in output
    assert "图片" in output  # 图片计为 [图片] 占位


def test_render_history_skips_system():
    tui = TUI()
    tui.console = Console(file=StringIO(), force_terminal=True)

    messages = [
        {"role": "system", "content": "secret system prompt"},
        {"role": "user", "content": "hello"},
    ]
    tui.render_history(messages)

    tui.console.file.seek(0)
    output = tui.console.file.read()
    assert "secret system prompt" not in output
    assert "hello" in output


def test_render_history_truncates_long():
    tui = TUI()
    tui.console = Console(file=StringIO(), force_terminal=True)

    messages = [{"role": "system", "content": "sys"}]
    for i in range(30):
        messages.append({"role": "user", "content": f"msg{i}"})
        messages.append({"role": "assistant", "content": f"reply{i}"})

    tui.render_history(messages, max_recent=3)

    tui.console.file.seek(0)
    output = tui.console.file.read()
    assert "历史消息" in output
    assert "reply29" in output


def test_render_history_empty():
    tui = TUI()
    tui.console = Console(file=StringIO(), force_terminal=True)
    tui.render_history([])
    tui.console.file.seek(0)
    output = tui.console.file.read()
    assert output.strip() == ""


def test_render_history_skips_tool_messages():
    tui = TUI()
    tui.console = Console(file=StringIO(), force_terminal=True)

    messages = [
        {"role": "user", "content": "read file"},
        {"role": "tool", "content": "file contents here"},
        {"role": "assistant", "content": "done"},
    ]
    tui.render_history(messages)

    tui.console.file.seek(0)
    output = tui.console.file.read()
    assert "file contents" not in output
    assert "read file" in output
    assert "done" in output
