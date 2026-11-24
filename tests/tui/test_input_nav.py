"""空闲输入框上下键按显示行移动（loongcli/tui/input_nav.py）。

纯函数测折行与列选择；端到端用真实 PromptSession（管道输入 + 定宽假输出）按键，
并把纯函数的折行结果和 prompt_toolkit 渲染器自己记录的字符坐标逐个比对。
"""
from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.data_structures import Size
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import DummyOutput

from loongcli.tui.input_nav import add_display_row_bindings, layout_rows, move_by_display_row

UP, DOWN, ENTER = "\x1b[A", "\x1b[B", "\r"


def _move(text, cursor, direction, width=20, pref=None):
    return move_by_display_row(text, cursor, direction, width, 2, 2, pref)


# ---------- 纯函数 ----------

class TestLayout:
    def test_pure_english_wraps_at_width(self):
        # 宽 20、前缀 2 → 每行 18 格
        rows = layout_rows("a" * 40, 20, 2, 2)
        assert [len(r) for r in rows] == [18, 18, 5]  # 末行 4 个字符 + 行尾光标格
        assert rows[1][0] == (18, 2)

    def test_exactly_full_line_puts_end_cursor_on_next_row(self):
        # 18 个字符正好填满：行尾光标格放不下，折到第二显示行
        rows = layout_rows("a" * 18, 20, 2, 2)
        assert len(rows) == 2
        assert rows[1] == [(18, 2)]

    def test_wide_char_that_does_not_fit_leaves_blank_cell(self):
        # "a" 占 x=2，8 个中文占 x=3..18，第 9 个从 x=19 起要 2 格放不下 → 折行
        rows = layout_rows("a" + "中" * 10, 20, 2, 2)
        assert rows[0][-1] == (8, 17)
        assert rows[1][0] == (9, 2)

    def test_manual_newline_starts_new_row_with_continuation_prefix(self):
        rows = layout_rows("ab\ncd", 20, 2, 2)
        assert rows == [[(0, 2), (1, 3), (2, 4)], [(3, 2), (4, 3), (5, 4)]]


class TestMove:
    def test_up_keeps_column_across_wrapped_rows(self):
        text = "a" * 40
        assert _move(text, 40, -1) == (22, 6)
        assert _move(text, 22, -1) == (4, 6)

    def test_first_row_up_and_last_row_down_return_none(self):
        text = "a" * 40
        assert _move(text, 4, -1) is None
        assert _move(text, 40, 1) is None
        assert _move("", 0, -1) is None  # 空输入：上键照常翻历史

    def test_down_is_symmetric(self):
        assert _move("a" * 40, 4, 1) == (22, 6)

    def test_chinese_column_counts_two_cells(self):
        # 末行：2 个中文 + 行尾格，行尾在 x=6；上一行 x=2,4,6… → 第 3 个字
        assert _move("中" * 20, 20, -1) == (11, 6)

    def test_goal_inside_wide_char_lands_before_it(self):
        # 第二行 "b" 在 x=3（奇数列），上一行全中文：x=3 落在第一个字中间 → 停在它前面
        text = "中" * 9 + "ab"
        assert _move(text, 10, -1) == (0, 3)

    def test_mixed_chinese_english(self):
        text = "a" + "中" * 10
        # 末行 [9@2, 10@4, 行尾 11@6] → 上一行 x<=6 的最右位置是第 2 个中文（下标 2，x=5）
        assert _move(text, 11, -1) == (2, 6)

    def test_manual_newline_moves_to_end_of_shorter_line(self):
        assert _move("ab\ncdef", 7, -1) == (2, 6)

    def test_exactly_full_line(self):
        text = "a" * 18
        assert _move(text, 18, -1) == (0, 2)
        assert _move(text, 0, 1) == (18, 2)

    def test_preferred_column_survives_short_row(self):
        text = "a" * 30 + "\nb\n" + "c" * 30
        cursor = 33 + 10  # 第三逻辑行（从下标 33 起）第 11 个字符，x=12
        mid, goal = _move(text, cursor, -1)
        assert mid == 32 and goal == 12  # "b" 行很短，停在它的行尾
        top, _ = _move(text, mid, -1, pref=goal)
        assert top == 18 + 10  # 回到第一逻辑行第二显示行的 x=12，而不是被拉到 x=3

    def test_combining_mark_tie_goes_after_it(self):
        # 第二行 "é"(e + U+0301) 后跟 "x"；U+0301 零宽，和 "x" 同列，取靠后的，不把光标插进组合序列
        text = "a" * 18 + "éx"
        rows = layout_rows(text, 20, 2, 2)
        assert rows[1][:3] == [(18, 2), (19, 3), (20, 3)]
        # 从上一行 x=3 下移：e@2、U+0301@3、x@3 → 取 x 前（下标 5），不取组合符前（下标 4）
        assert _move("aa\néx", 1, 1) == (5, 3)

    def test_too_narrow_returns_none(self):
        assert _move("abc", 3, -1, width=3) is None


# ---------- 端到端：真实 PromptSession ----------

class _FixedOutput(DummyOutput):
    def __init__(self, columns: int):
        super().__init__()
        self._columns = columns

    def get_size(self) -> Size:
        return Size(rows=40, columns=self._columns)


def _run(default: str, keys: str, history=(), width: int = 20, prompt="❯ "):
    kb = KeyBindings()
    add_display_row_bindings(kb, prompt, "  ")
    hist = InMemoryHistory()
    for h in history:
        hist.append_string(h)
    with create_pipe_input() as inp:
        session = PromptSession(
            prompt() if callable(prompt) else prompt, history=hist, key_bindings=kb, prompt_continuation="  ",
            input=inp, output=_FixedOutput(width),
        )

        # 历史是异步加载的：按键先于加载完成到达时，原生上键也翻不到历史（不加本绑定同样复现）。
        # 应用启动后稍等再送键，模拟真人按键。
        def _send_later():
            asyncio.get_running_loop().call_later(0.2, inp.send_text, keys)

        result = session.prompt(default=default, pre_run=_send_later)
    return result, session


class TestEndToEnd:
    def test_up_moves_between_wrapped_rows_then_history(self):
        text = "中" * 20  # 折成三显示行
        assert _run(text, UP + "X" + ENTER)[0] == text[:11] + "X" + text[11:]
        assert _run(text, UP + UP + "X" + ENTER)[0] == text[:2] + "X" + text[2:]
        # 第三次上键才翻到上一条历史
        assert _run(text, UP * 3 + ENTER, history=["上一条"])[0] == "上一条"

    def test_prompt_width_follows_callable_prompt(self):
        """提示符会变（规划模式「规划 ❯ 」宽 7 格）：首行前缀按当前提示符现算，而不是绑定时写死的 2 格。"""
        text = "a" * 15                  # 窄提示符一行放得下；宽提示符折成两行，上键回到首行行首
        assert _run(text, UP + "X" + ENTER, prompt=lambda: "规划 ❯ ")[0] == "X" + text
        assert _run(text, UP + "X" + ENTER)[0] == text + "X"

    def test_down_moves_back(self):
        text = "a" * 40
        assert _run(text, UP + UP + DOWN + "X" + ENTER)[0] == text[:22] + "X" + text[22:]

    def test_exactly_full_and_blank_cell_cases_match_renderer(self):
        assert _run("a" * 18, UP + "X" + ENTER)[0] == "X" + "a" * 18
        text = "a" + "中" * 10
        assert _run(text, UP + "X" + ENTER)[0] == text[:2] + "X" + text[2:]

    def test_logical_lines_still_work(self):
        # Ctrl+J 换出的两行：原有行为（上一行同列）保持
        assert _run("ab\ncd", UP + "X" + ENTER)[0] == "abX\ncd"

    @pytest.mark.parametrize("text", [
        "a" * 40,
        "中" * 20,
        "a" + "中" * 10,
        "a" * 18,
        "混排 mixed 中英文 text 测试一下折行 wrap 的位置是否一致",
        "第一行\n" + "第二行很长" * 5 + "\n\nend",
    ])
    def test_layout_matches_prompt_toolkit_renderer(self, text):
        """纯函数的折行规则必须和 prompt_toolkit 实际渲染逐字符一致。"""
        _, session = _run(text, ENTER)
        info = session.app.layout.current_window.render_info
        actual = info._rowcol_to_yx  # (逻辑行, 列) → (屏幕 y, 屏幕 x)
        lines = text.split("\n")
        ours = {}
        for r, row in enumerate(layout_rows(text, info.window_width, 2, 2)):
            for idx, x in row:
                lineno = text.count("\n", 0, idx)
                col = idx - (sum(len(l) + 1 for l in lines[:lineno]))
                ours[lineno, col] = (r, x)
        y0 = min(y for y, _ in actual.values())
        assert {k: (y - y0, x) for k, (y, x) in actual.items()} == ours
