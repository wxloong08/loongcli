"""执行中输入：键处理纯函数（_apply_key）+ 回显渲染（_fit_cells_tail/_StatusGroup/_typed_echo_line）。"""
from io import StringIO

import pytest
from rich.console import Console
from rich.cells import cell_len
from rich.text import Text

from loongcli.tui.app import TUI, _apply_key, _fit_cells_tail, _INPUT_QUEUE_MAX
from loongcli.tui.mdstream import _StatusGroup


class TestApplyKey:
    @pytest.mark.parametrize("ch,buffer,expected", [
        # (按键, 现有缓冲, (新缓冲, 提交, 中断))
        ("a", "", ("a", None, False)),
        ("中", "a", ("a中", None, False)),
        (" ", "a", ("a ", None, False)),          # 空格是可见输入
        ("\x1b", "half", ("half", None, True)),   # Esc 永远中断，缓冲不动
        ("\x1b", "", ("", None, True)),
        ("\r", "hello ", ("", "hello", False)),   # Enter 提交并 strip
        ("\n", "hi", ("", "hi", False)),
        ("\r", "   ", ("", None, False)),         # 纯空白不提交
        ("\r", "", ("", None, False)),            # 空缓冲回车不提交
        ("\x08", "ab", ("a", None, False)),       # Backspace 删尾
        ("\x7f", "ab", ("a", None, False)),
        ("\x08", "", ("", None, False)),          # 空缓冲退格安全
        ("\x03", "ab", ("ab", None, False)),      # 其余控制字符忽略
    ])
    def test_table(self, ch, buffer, expected):
        assert _apply_key(ch, buffer) == expected

    def test_cjk_accumulates(self):
        buf = ""
        for ch in "修一下bug":
            buf, submitted, cancel = _apply_key(ch, buf)
            assert submitted is None and cancel is False
        buf, submitted, _ = _apply_key("\r", buf)
        assert submitted == "修一下bug"
        assert buf == ""

    def test_surrogate_pair_repaired_on_submit(self):
        # 非 BMP 字符（emoji）经 getwch 以代理对分两次到达，提交时收口成真字符
        buf = ""
        for ch in ("\ud83d", "\ude00"):  # 😀 的代理对
            buf, _, _ = _apply_key(ch, buf)
        _, submitted, _ = _apply_key("\r", buf)
        assert submitted == "😀"

    def test_queue_max_positive(self):
        assert _INPUT_QUEUE_MAX > 0


class TestFitCellsTail:
    def test_no_truncation_when_fits(self):
        assert _fit_cells_tail("hi", 10) == "hi"

    def test_keeps_tail_not_head(self):
        out = _fit_cells_tail("hello world", 6)
        assert cell_len(out) <= 6
        assert out.startswith("…")
        assert out.endswith("world"[-1])

    def test_chinese_double_width(self):
        out = _fit_cells_tail("你好世界", 4)
        assert cell_len(out) <= 4
        assert out == "…界"

    def test_zero_budget(self):
        assert _fit_cells_tail("x", 0) == ""


def _render(renderable, width=40) -> str:
    console = Console(file=StringIO(), width=width, force_terminal=False, legacy_windows=False)
    console.print(renderable)
    return console.file.getvalue()


class TestStatusGroup:
    def test_provider_none_renders_base_only(self):
        out = _render(_StatusGroup(Text("base"), lambda: None))
        assert "base" in out
        assert len(out.strip().splitlines()) == 1

    def test_provider_line_renders_below_base(self):
        out = _render(_StatusGroup(Text("base"), lambda: Text("echo-line")))
        lines = out.strip().splitlines()
        assert "base" in lines[0]
        assert "echo-line" in lines[1]

    def test_provider_exception_treated_as_none(self):
        def boom():
            raise RuntimeError("x")
        out = _render(_StatusGroup(Text("base"), boom))
        assert "base" in out
        assert len(out.strip().splitlines()) == 1


class _StubAgent:
    def __init__(self, queued=None):
        self._queued_user_inputs = queued or []


class _StubTUI:
    """只提供 _typed_echo_line 依赖的属性面，绕过 TUI 完整构造。"""
    PADDING = TUI.PADDING
    console = Console(file=StringIO(), width=60)
    _prompt_active = False
    _typed_buffer = ""


class TestTypedEchoLine:
    def test_empty_state_shows_placeholder(self):
        # 空态常驻占位行（可发现性）：单行、含操作提示
        tui = _StubTUI()
        line = TUI._typed_echo_line(tui, _StubAgent())
        assert line is not None
        out = _render(line, width=60)
        assert "打字可插话" in out
        assert len(out.strip().splitlines()) == 1

    def test_prompt_active_returns_none(self):
        tui = _StubTUI()
        tui._typed_buffer = "half"
        tui._prompt_active = True
        assert TUI._typed_echo_line(tui, _StubAgent()) is None

    def test_buffer_rendered_single_line(self):
        tui = _StubTUI()
        tui._typed_buffer = "修一下 bug"
        line = TUI._typed_echo_line(tui, _StubAgent())
        assert line is not None
        out = _render(line, width=60)
        assert "修一下 bug" in out
        assert len(out.strip().splitlines()) == 1

    def test_queued_indicator_shown(self):
        tui = _StubTUI()
        line = TUI._typed_echo_line(tui, _StubAgent(queued=["a", "b"]))
        out = _render(line, width=60)
        assert "+2 排队" in out

    def test_long_buffer_keeps_tail_one_line(self):
        tui = _StubTUI()
        tui._typed_buffer = "长" * 100  # 200 列，远超 60 列终端
        line = TUI._typed_echo_line(tui, _StubAgent())
        out = _render(line, width=60)
        assert len(out.strip().splitlines()) == 1
        assert "…" in out


class TestPackStatusLines:
    """banner 状态条目按条目边界打包——条目内断行会产生悬空 ⚠（2026-07-19 截图实锤）。"""

    def test_all_fit_one_line(self):
        from loongcli.tui.app import _pack_status_lines
        assert _pack_status_lines(["a: 1", "b: 2"], 40) == ["a: 1 | b: 2"]

    def test_breaks_at_item_boundary(self):
        from loongcli.tui.app import _pack_status_lines
        items = ["model: deepseek-v4-pro", "thinking: max", "⚠ permissions: skip"]
        lines = _pack_status_lines(items, 42)
        # 每个条目完整出现在某一行里，绝不跨行
        joined = "\n".join(lines)
        for item in items:
            assert item in joined
            assert not any(item.split()[0] == line.split(" | ")[-1] and len(item.split()) > 1
                           for line in lines[:-1] if not line.endswith(item))
        # ⚠ 条目与其说明在同一行
        warn_lines = [l for l in lines if "⚠" in l]
        assert len(warn_lines) == 1
        assert "permissions: skip" in warn_lines[0]

    def test_single_oversized_item_own_line_not_split(self):
        from loongcli.tui.app import _pack_status_lines
        lines = _pack_status_lines(["short", "x" * 100], 40)
        assert lines == ["short", "x" * 100]

    def test_cjk_width_counted(self):
        from loongcli.tui.app import _pack_status_lines
        # 4 个中文 = 8 cell；宽度 12 放不下两个条目（8+3+8=19）
        lines = _pack_status_lines(["中文条目", "中文条目"], 12)
        assert len(lines) == 2

    def test_empty(self):
        from loongcli.tui.app import _pack_status_lines
        assert _pack_status_lines([], 40) == []
