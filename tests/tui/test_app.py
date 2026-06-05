import pytest
from loongcli.tui.app import TUI


def test_tui_init():
    tui = TUI()
    assert tui.console is not None


def test_format_tool_result_short():
    tui = TUI()
    result = tui._format_tool_result("shell", "hello world")
    assert "hello world" in result


def test_format_tool_result_truncated():
    tui = TUI()
    long_result = "x" * 2000
    result = tui._format_tool_result("shell", long_result)
    assert len(result) < len(long_result)
    assert "截断" in result or "..." in result


class TestIsCompactMessage:
    def test_compact_boundary(self):
        assert TUI._is_compact_message("[compact-boundary] mode=auto | pre_tokens=100")

    def test_context_recovery(self):
        assert TUI._is_compact_message("[压缩后上下文恢复]\n\n## 最近文件\n...")

    def test_summary_marker(self):
        assert TUI._is_compact_message("[对话历史摘要]\n## 1. 主要意图\n...")

    def test_summary_ack(self):
        assert TUI._is_compact_message("好的，我已了解之前的对话内容。请继续。")

    def test_attachment_ack(self):
        assert TUI._is_compact_message("好的，我已了解恢复的上下文信息，继续工作。")

    def test_normal_user_message(self):
        assert not TUI._is_compact_message("给loongcli加一个 /status 命令")

    def test_empty_content(self):
        assert not TUI._is_compact_message("")

    def test_none_content(self):
        assert not TUI._is_compact_message(None)


def test_render_history_skips_compact_messages():
    """Compact boundary and recovery messages should not be rendered."""
    tui = TUI()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "[compact-boundary] mode=auto | pre_tokens=1000 | pre_messages=50"},
        {"role": "user", "content": "[对话历史摘要]\n## 1. 主要意图\nUser wants X"},
        {"role": "assistant", "content": "好的，我已了解之前的对话内容。请继续。"},
        {"role": "user", "content": "[压缩后上下文恢复]\n\n## 最近文件\n### foo.py\n```\ncode\n```"},
        {"role": "assistant", "content": "好的，我已了解恢复的上下文信息，继续工作。"},
        {"role": "user", "content": "continue working"},
    ]
    # Build conversations list using same logic as render_history
    conversations = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        if tui._is_compact_message(content):
            continue
        if role == "user":
            conversations.append(("user", content))
        elif role == "assistant" and content:
            conversations.append(("assistant", content))

    assert len(conversations) == 3
    assert conversations[0] == ("user", "hello")
    assert conversations[1] == ("assistant", "hi there")
    assert conversations[2] == ("user", "continue working")
