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


class TestPasteCollapse:
    def setup_method(self):
        self.tui = TUI()

    def test_large_multiline_paste_collapsed(self):
        data = "\n".join(f"line {i}" for i in range(40))
        out = self.tui._maybe_collapse_paste(data)
        assert out == "[Pasted text #1 +40 lines]"
        assert self.tui._pastes[out] == data  # 完整原文被保存

    def test_long_singleline_paste_collapsed(self):
        data = "x" * 400
        out = self.tui._maybe_collapse_paste(data)
        assert out.startswith("[Pasted text #")
        assert self.tui._pastes[out] == data

    def test_small_paste_not_collapsed(self):
        out = self.tui._maybe_collapse_paste("短文本 a b c")
        assert out == "短文本 a b c"
        assert self.tui._pastes == {}

    def test_image_paste_collapses_to_placeholder(self, tmp_path):
        p = tmp_path / "shot.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\ndata")
        out = self.tui._maybe_collapse_paste(str(p))
        assert out == "[Image #1]"
        assert self.tui._paste_images["[Image #1]"] == str(p)

    def test_quoted_image_paste_collapses(self, tmp_path):
        d = tmp_path / "my shots"
        d.mkdir()
        p = d / "a.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\ndata")
        out = self.tui._maybe_collapse_paste(f'"{p}"')
        assert out == "[Image #1]"
        assert self.tui._paste_images["[Image #1]"] == str(p)


class TestDropManyFiles:
    """一次拖进多个文件：终端粘贴过来的是一串路径（真机：两张图一起拖，变成两段裸路径）。
    各种来源的格式都要认：空格分隔、带引号含空格、每行一个、file:/// URI、/c/ 风格，图片和非图片混着。"""

    def setup_method(self):
        self.tui = TUI()

    @staticmethod
    def _img(d, name):
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG\r\n\x1a\ndata")
        return p

    def test_space_separated_unquoted(self, tmp_path):
        a, b = self._img(tmp_path, "a.png"), self._img(tmp_path, "b.png")
        out = self.tui._maybe_collapse_paste(f"{a} {b}")
        assert out == "[Image #1][Image #2]"
        assert self.tui._paste_images == {"[Image #1]": str(a), "[Image #2]": str(b)}

    def test_quoted_with_spaces_mixed(self, tmp_path):
        a, b = self._img(tmp_path, "my shots/a b.png"), self._img(tmp_path, "c.png")
        out = self.tui._maybe_collapse_paste(f'"{a}" {b}')
        assert out == "[Image #1][Image #2]"
        assert self.tui._paste_images["[Image #1]"] == str(a)

    def test_one_per_line(self, tmp_path):
        a, b = self._img(tmp_path, "a.png"), self._img(tmp_path, "b.jpg")
        out = self.tui._maybe_collapse_paste(f"{a}\r\n{b}\r\n")
        assert out == "[Image #1][Image #2]"

    def test_file_uri_and_percent_encoding(self, tmp_path):
        a = self._img(tmp_path, "截图 1.png")
        from urllib.parse import quote
        uri = "file:///" + quote(str(a).replace("\\", "/"))
        out = self.tui._maybe_collapse_paste(uri)
        assert out == "[Image #1]" and self.tui._paste_images["[Image #1]"] == str(a)

    def test_images_and_other_files_mixed(self, tmp_path):
        a = self._img(tmp_path, "a.png")
        py = tmp_path / "x y.py"
        py.write_text("print(1)")
        out = self.tui._maybe_collapse_paste(f'{a} "{py}"')
        # 图片变占位符；非图片文件保留成路径（含空格的加引号），交给模型自己读
        assert out == f'[Image #1] "{py}"'

    def test_same_image_twice_reuses_placeholder(self, tmp_path):
        a = self._img(tmp_path, "a.png")
        assert self.tui._maybe_collapse_paste(str(a)) == "[Image #1]"
        assert self.tui._maybe_collapse_paste(f'"{a}"') == "[Image #1]"
        assert len(self.tui._paste_images) == 1

    def test_unquoted_path_with_spaces_as_whole(self, tmp_path):
        a = self._img(tmp_path, "my shots/a b.png")
        assert self.tui._maybe_collapse_paste(str(a)) == "[Image #1]"

    def test_text_mentioning_a_path_is_not_a_drop(self, tmp_path):
        a = self._img(tmp_path, "a.png")
        text = f"看看 {a} 这张"
        assert self.tui._maybe_collapse_paste(text) == text      # 不是纯路径就不是拖文件，交给提交时的路径识别

    def test_nonexistent_paths_untouched(self):
        text = r"C:\nope\a.png C:\nope\b.png"
        assert self.tui._maybe_collapse_paste(text) == text


class TestResolveImages:
    def setup_method(self):
        self.tui = TUI()

    def test_placeholder_vision_extracts(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNGd")
        self.tui._paste_images["[Image #1]"] = str(p)
        text, imgs = self.tui._resolve_images("这是什么 [Image #1]", vision=True)
        assert imgs == [str(p)]
        assert text == "这是什么"
        assert self.tui._paste_images == {}  # 已清空

    def test_placeholder_no_vision_restores_path(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNGd")
        self.tui._paste_images["[Image #1]"] = str(p)
        text, imgs = self.tui._resolve_images("看 [Image #1]", vision=False)
        assert imgs == []
        assert str(p) in text  # 占位符还原成路径文本

    def test_typed_path_vision(self, tmp_path):
        p = tmp_path / "y.png"
        p.write_bytes(b"\x89PNGd")
        text, imgs = self.tui._resolve_images(f"描述 {p}", vision=True)
        assert imgs == [str(p)]
        assert text == "描述"

    def test_no_images(self):
        assert self.tui._resolve_images("hello", vision=True) == ("hello", [])

    def test_crlf_normalized(self):
        out = self.tui._maybe_collapse_paste("a\r\nb\r\nc\r\nd\r\ne")
        assert "\r" not in self.tui._pastes[out]

    def test_expand_restores_full_text_and_clears(self):
        data = "\n".join(["广告"] * 40)
        ph = self.tui._maybe_collapse_paste(data)
        expanded = self.tui._expand_pastes(f"我的问题？{ph}")
        assert data in expanded
        assert ph not in expanded
        assert self.tui._pastes == {}  # 提交后清空

    def test_expand_noop_without_pastes(self):
        assert self.tui._expand_pastes("普通输入") == "普通输入"

    def test_multiple_pastes_each_expanded(self):
        a = self.tui._maybe_collapse_paste("\n".join(["A"] * 10))
        b = self.tui._maybe_collapse_paste("\n".join(["B"] * 10))
        assert a != b  # 计数器递增
        out = self.tui._expand_pastes(f"{a} 和 {b}")
        assert "A\nA" in out and "B\nB" in out


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


# ── Shift+Tab 规划模式切换 ──

def test_toggle_plan_mode_roundtrip():
    from io import StringIO
    from unittest.mock import MagicMock
    from rich.console import Console
    from loongcli.tui.app import TUI

    tui = TUI()
    buf = StringIO()
    tui.console = Console(file=buf, force_terminal=True, width=80)

    tui._toggle_plan_mode()  # 无 agent → 静默 no-op
    assert buf.getvalue() == ""

    agent = MagicMock()
    agent.plan_mode = False
    agent.llm = None                    # 状态行右半边（模型、上下文）不取，只看左边的说明
    tui._agent = agent
    tui._toggle_plan_mode()
    agent.enter_plan_mode.assert_called_once()
    assert buf.getvalue() == ""

    agent.plan_mode = True
    assert "规划模式：先调研再出计划" in tui._status_line()[0]   # 含义写在状态行
    tui._toggle_plan_mode()
    assert agent._plan_mode is False
    agent._rebuild_system_prompt.assert_called_once()  # 退出必须重建系统提示
    assert buf.getvalue() == ""                         # 切换不往滚动区打字（真机：切三回打出 6 行）


class TestCountImagesInInput:
    """vision 关闭时用来决定要不要提示「图没发出去」（只数不改）。"""

    def setup_method(self):
        self.tui = TUI()

    def test_counts_typed_path_and_placeholder(self, tmp_path):
        a = tmp_path / "a.png"
        a.write_bytes(b"\x89PNGd")
        b = tmp_path / "b.png"
        b.write_bytes(b"\x89PNGd")
        self.tui._paste_images["[Image #1]"] = str(a)
        assert self.tui._count_images_in_input(f"看 [Image #1] 和 {b}") == 2
        # 只数不改：占位符表原样，后续 _resolve_images 仍能还原路径
        assert self.tui._paste_images == {"[Image #1]": str(a)}

    def test_zero_for_plain_text_or_missing_file(self, tmp_path):
        assert self.tui._count_images_in_input("hello") == 0
        assert self.tui._count_images_in_input(f"看 {tmp_path / 'nope.png'}") == 0
