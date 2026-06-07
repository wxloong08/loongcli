import pytest
from loongcli.tui.app import TUI


def test_tui_init():
    tui = TUI()
    assert tui.console is not None


class TestFormatToolResult:
    def setup_method(self):
        self.tui = TUI()

    def test_empty_result(self):
        assert self.tui._format_tool_result("shell", "") == ""

    def test_read_file_shows_line_count(self):
        content = "line1\nline2\nline3\n"
        result = self.tui._format_tool_result("read_file", content)
        assert "(3 行)" in result

    def test_read_file_single_line(self):
        result = self.tui._format_tool_result("read_file", "only one line")
        assert "(1 行)" in result

    def test_shell_pytest_passed(self):
        output = (
            "tests/test_foo.py ...\n"
            "=============== 4 passed in 0.02s ===============\n"
            "[exit code: 0]"
        )
        result = self.tui._format_tool_result("shell", output)
        assert "passed" in result
        assert "4 passed" in result

    def test_shell_pytest_failed(self):
        output = (
            "FAILED tests/test_foo.py::test_bar\n"
            "=============== 1 failed, 3 passed in 0.05s ===============\n"
            "[exit code: 1 — 一般错误]"
        )
        result = self.tui._format_tool_result("shell", output)
        assert "failed" in result

    def test_shell_error_keyword(self):
        output = "SyntaxError: invalid syntax\n[exit code: 1]"
        result = self.tui._format_tool_result("shell", output)
        assert "error" in result.lower()

    def test_shell_fallback_last_line(self):
        output = "building project...\nDone in 2.3s"
        result = self.tui._format_tool_result("shell", output)
        assert "Done in 2.3s" in result

    def test_shell_skips_exit_code_line(self):
        output = "All good\n[exit code: 0]"
        result = self.tui._format_tool_result("shell", output)
        assert "All good" in result
        assert "[exit code" not in result

    def test_shell_long_line_truncated(self):
        output = "x" * 200 + " passed\n[exit code: 0]"
        result = self.tui._format_tool_result("shell", output)
        assert len(result) <= 124
        assert result.endswith("...")

    def test_shell_deprecation_warning_skipped(self):
        output = (
            "tests/test_foo.py ....\n"
            "=============== 4 passed in 0.02s ===============\n"
            "DeprecationWarning: some old API\n"
            "[exit code: 1 — 一般错误]"
        )
        result = self.tui._format_tool_result("shell", output)
        assert "passed" in result
        assert "Deprecation" not in result

    def test_other_tool_short_result(self):
        result = self.tui._format_tool_result("grep", "hello world")
        assert "hello world" in result

    def test_other_tool_long_result_truncated(self):
        long_result = "x" * 2000
        result = self.tui._format_tool_result("grep", long_result)
        assert len(result) < len(long_result)
        assert "截断" in result


class TestIsInternalMessage:
    def test_compact_boundary(self):
        assert TUI._is_internal_message("[compact-boundary] mode=auto | pre_tokens=100")

    def test_context_recovery(self):
        assert TUI._is_internal_message("[压缩后上下文恢复]\n\n## 最近文件\n...")

    def test_summary_marker(self):
        assert TUI._is_internal_message("[对话历史摘要]\n## 1. 主要意图\n...")

    def test_summary_ack(self):
        assert TUI._is_internal_message("好的，我已了解之前的对话内容。请继续。")

    def test_attachment_ack(self):
        assert TUI._is_internal_message("好的，我已了解恢复的上下文信息，继续工作。")

    def test_short_ack(self):
        assert TUI._is_internal_message("好的，已了解。")

    def test_snip_marker(self):
        assert TUI._is_internal_message("[snip] 已删除 10 条远古消息")

    def test_tool_placeholder(self):
        assert TUI._is_internal_message("[工具结果已省略]")

    def test_verify_prompt(self):
        assert TUI._is_internal_message("## 验证\n\n本轮修改了 3 个文件:")

    def test_verify_retry(self):
        assert TUI._is_internal_message("## 验证 — 第 2 轮重试\n\n上一轮验证失败。")

    def test_normal_user_message(self):
        assert not TUI._is_internal_message("给loongcli加一个 /status 命令")

    def test_empty_content(self):
        assert not TUI._is_internal_message("")

    def test_none_content(self):
        assert not TUI._is_internal_message(None)


def test_render_history_skips_internal_messages():
    """Internal messages (compact, verify, tool) should not be rendered."""
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
        {"role": "tool", "content": "file contents here"},
        {"role": "user", "content": "## 验证\n\n本轮修改了 2 个文件:\n- src/foo.py"},
        {"role": "user", "content": "continue working"},
    ]
    conversations = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("system", "tool"):
            continue
        if tui._is_internal_message(content):
            continue
        if role == "user":
            conversations.append(("user", content or ""))
        elif role == "assistant" and content:
            conversations.append(("assistant", content))

    assert len(conversations) == 3
    assert conversations[0] == ("user", "hello")
    assert conversations[1] == ("assistant", "hi there")
    assert conversations[2] == ("user", "continue working")
