"""tool_display 摘要器单测：参数核心值化、结果统计化、edit 真 diff、错误分支。"""
from pathlib import Path

from loongcli.tui.tool_display import (
    arg_summary, result_lines, relpath, DIFF_MAX_LINES,
)


def _plain(lines) -> list[str]:
    return [t.plain for t in lines]


# ── arg_summary ─────────────────────────────────────────────────────

class TestArgSummary:
    def test_read_file_relative_path(self):
        p = str(Path.cwd() / "loongcli" / "core" / "config.py")
        s = arg_summary("read_file", {"path": p})
        assert s == str(Path("loongcli") / "core" / "config.py")

    def test_read_file_with_range(self):
        s = arg_summary("read_file", {"path": "x.py", "offset": 10, "limit": 20})
        assert s.endswith(":10-29")

    def test_outside_cwd_path_kept(self):
        assert relpath("C:/elsewhere/a.py") == "C:/elsewhere/a.py"

    def test_grep_pattern_and_path(self):
        # relpath 会把尾斜杠归一掉，显示无妨
        assert arg_summary("grep", {"pattern": "load_config", "path": "loongcli/"}) == \
            "load_config, loongcli"

    def test_shell_command_clipped(self):
        s = arg_summary("shell", {"command": "x" * 100})
        assert len(s) <= 61 and s.endswith("…")

    def test_unknown_tool_values_only(self):
        s = arg_summary("searxng__web_search", {"query": "缓存机制", "pageno": 1})
        assert "query=" not in s and "缓存机制" in s and "1" in s

    def test_edit_file_no_string_dump(self):
        s = arg_summary("edit_file", {"path": "a.py", "old_string": "x" * 200, "new_string": "y" * 200})
        assert "xxx" not in s and s == "a.py"


# ── result_lines ────────────────────────────────────────────────────

class TestResultLines:
    def test_read_file_line_count(self):
        ok, lines = result_lines("read_file", {}, "1\ta\n2\tb\n3\tc")
        assert ok and _plain(lines) == ["3 行"]

    def test_read_file_image_placeholder_passthrough(self):
        ok, lines = result_lines("read_file", {}, "[图片已读取（1 张），内容见下一条消息]")
        assert ok and "图片已读取" in _plain(lines)[0]

    def test_glob_file_count(self):
        ok, lines = result_lines("glob", {}, "a.py\nb.py\nc.py")
        assert ok and _plain(lines) == ["3 个文件"]

    def test_grep_counts_hits_and_files_windows_paths(self):
        result = (
            "loongcli/core/config.py:65: def load_config():\n"
            "loongcli/core/config.py:88: cfg = load_config()\n"
            "loongcli/main.py:12: from x import load_config\n"
        )
        ok, lines = result_lines("grep", {}, result)
        assert ok and _plain(lines) == ["3 处匹配 · 2 个文件"]

    def test_grep_not_found_passthrough(self):
        ok, lines = result_lines("grep", {}, "未找到匹配")
        assert ok and _plain(lines) == ["未找到匹配"]

    def test_edit_diff_lines_styled(self):
        args = {"old_string": "a\nb\nc", "new_string": "a\nB\nc"}
        ok, lines = result_lines("edit_file", args, "成功编辑 x.py（添加 1 行，删除 1 行）")
        assert ok
        plain = _plain(lines)
        assert any(l.startswith("-b") for l in plain)
        assert any(l.startswith("+B") for l in plain)
        minus = next(t for t in lines if t.plain.startswith("-"))
        plus = next(t for t in lines if t.plain.startswith("+"))
        # 显式十六进制纯白字 + 深色底——调色板名经主题映射后亮度不可控（真机两轮发灰）
        assert "#ffffff" in str(minus.style) and "on #6e1e1e" in str(minus.style)
        assert "#ffffff" in str(plus.style) and "on #1d5f2d" in str(plus.style)

    def test_edit_diff_capped(self):
        old = "\n".join(f"l{i}" for i in range(40))
        new = "\n".join(f"L{i}" for i in range(40))
        ok, lines = result_lines("edit_file", {"old_string": old, "new_string": new}, "成功编辑 x.py")
        assert ok
        assert len(lines) == DIFF_MAX_LINES + 1
        assert "还有" in lines[-1].plain

    def test_edit_fuzzy_note(self):
        args = {"old_string": "a", "new_string": "b"}
        ok, lines = result_lines("edit_file", args, "成功编辑 x.py（fuzzy 92%，添加 1 行）")
        assert ok and "fuzzy 92%" in _plain(lines)[0]

    def test_write_file_line_count_from_args(self):
        ok, lines = result_lines("write_file", {"content": "a\nb\nc\nd"}, "成功写入 x.py")
        assert ok and _plain(lines) == ["写入 4 行"]

    def test_shell_picks_test_summary_line(self):
        result = "....\n1297 passed, 3 warnings in 100.91s\n"
        ok, lines = result_lines("shell", {}, result)
        assert ok and "1297 passed" in _plain(lines)[0]

    def test_shell_nonzero_exit_marks_failed(self):
        ok, lines = result_lines("shell", {}, "boom\n[exit code: 1]")
        assert not ok
        assert "boom" in _plain(lines)[0]

    def test_error_result_red_lines(self):
        ok, lines = result_lines("read_file", {}, "错误：文件不存在 'x.py'")
        assert not ok
        assert "文件不存在" in _plain(lines)[0]
        assert str(lines[0].style) == "red"

    def test_default_tool_first_line_clipped(self):
        ok, lines = result_lines("searxng__web_search", {}, "第一行结果\n第二行\n第三行")
        assert ok
        assert _plain(lines)[0].startswith("第一行结果")
        assert _plain(lines)[0].endswith("…")

    def test_empty_result_no_lines(self):
        ok, lines = result_lines("shell", {}, "")
        assert ok and lines == []


class TestReadFileRangeHeader:
    def test_range_header_used_as_summary_no_content_leak(self):
        """带 [第 a-b 行] 头的多行结果：只显示头行，正文绝不漏进折叠块（真机回归）。"""
        result = "[第 1-1 行 / 共 2+ 行]\n\n1\tfrom __future__ import annotations"
        ok, lines = result_lines("read_file", {}, result)
        assert ok
        assert _plain(lines) == ["[第 1-1 行 / 共 2+ 行]"]

    def test_image_placeholder_still_passthrough_single_line(self):
        ok, lines = result_lines("read_file", {}, "[图片已读取（1 张），内容见下一条消息]")
        assert ok and "图片已读取" in _plain(lines)[0]

    def test_plain_content_still_counts_lines(self):
        ok, lines = result_lines("read_file", {}, "1\ta\n2\tb\n3\tc")
        assert ok and _plain(lines) == ["3 行"]
