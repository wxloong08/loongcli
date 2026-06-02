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
