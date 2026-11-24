"""空闲输入框的上下键：按屏幕上的显示行移动，而不是按逻辑行。

prompt_toolkit 默认的上下键（`Buffer.auto_up/auto_down`）只看逻辑行：一行长文本折成多行显示时，
光标不在第一个逻辑行才上移，否则直接翻历史——用户整段长输入时上键一按就丢掉正在编辑的内容。
这里按显示行移动：不在第一显示行就上移一行并尽量保持屏幕列，已在第一显示行才交给 auto_up 翻历史。

折行规则照抄 prompt_toolkit `Window._write_to_screen` 的 `copy_line`：
- 每个显示行先画前缀（第一逻辑行首行是提示符，其余都是续行前缀），再逐字符放；
- 放不下（`x + 字符宽 > 宽度`）就换到下一显示行；
- 每个逻辑行末尾补一格空格（`BufferControl` 的做法），那是光标停在行尾的位置。
端到端测试用真实渲染核对这份规则，prompt_toolkit 升级改了折行方式时会失败。
"""
from __future__ import annotations

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.utils import get_cwidth

# 一个显示行：[(文本下标, 该字符起始屏幕列)]
_Row = list[tuple[int, int]]


def layout_rows(text: str, width: int, first_prefix: int, cont_prefix: int) -> list[_Row]:
    """把输入切成显示行。每个逻辑行的行尾光标格（下标 = 行尾）算在该逻辑行最后一个显示行里。"""
    rows: list[_Row] = []
    start = 0
    for lineno, line in enumerate(text.split("\n")):
        x = first_prefix if lineno == 0 else cont_prefix
        row: _Row = []
        for offset, ch in enumerate(line + " "):  # 末尾空格 = 行尾光标格
            w = get_cwidth(ch)
            if x + w > width:
                rows.append(row)
                row, x = [], cont_prefix
            row.append((start + offset, x))
            x += w
        rows.append(row)
        start += len(line) + 1  # +1 跳过 "\n"
    return rows


def move_by_display_row(
    text: str,
    cursor: int,
    direction: int,
    width: int,
    first_prefix: int,
    cont_prefix: int,
    preferred_x: int | None = None,
) -> tuple[int, int] | None:
    """光标上移（direction=-1）或下移（+1）一个显示行。

    返回 (新光标下标, 本次用的目标列)；已在第一/最后一个显示行时返回 None，由调用方翻历史。
    preferred_x 是连续上下移动时记住的列：途经短行不会把光标永久拉到左边。
    """
    # 宽度容不下前缀加一个宽字符时折行规则退化（每行都空），交回默认行为
    if width - max(first_prefix, cont_prefix) < 2:
        return None
    rows = layout_rows(text, width, first_prefix, cont_prefix)
    for r, row in enumerate(rows):
        for idx, x in row:
            if idx == cursor:
                break
        else:
            continue
        break
    else:
        return None  # 光标下标越界，不该发生；交回默认行为

    target = r + direction
    if not 0 <= target < len(rows):
        return None
    goal_x = x if preferred_x is None else preferred_x
    # 目标行里起始列不超过 goal_x 的最右一个位置；同列并列（零宽组合字符）取靠后的，免得把光标插进组合序列中间
    best = rows[target][0][0]
    for idx, px in rows[target]:
        if px <= goal_x:
            best = idx
    return best, goal_x


def add_display_row_bindings(kb: KeyBindings, prompt, continuation: str) -> None:
    """给空闲输入框注册按显示行移动的上下键。补全菜单打开时保持原行为（选补全项）。

    prompt 可以是字符串或返回字符串的函数——提示符会变（规划模式下是「规划 ❯ 」），首行前缀宽度每次按键现算。"""
    cont_prefix = get_cwidth(continuation)
    # 连续上下移动时记住目标列：(文本, 移动后的光标, 列)，文本或光标变了就作废
    sticky: list[tuple[str, int, int]] = []

    def _move(event, direction: int) -> None:
        buf = event.current_buffer
        info = event.app.layout.current_window.render_info
        if buf.complete_state is None and info is not None:
            text = buf.text
            pref = sticky[0][2] if sticky and sticky[0][:2] == (text, buf.cursor_position) else None
            first_prefix = get_cwidth(prompt() if callable(prompt) else prompt)
            # 宽度取渲染器实际用的值：Windows 上 prompt_toolkit 会比终端列数少用一列，自己量会错位
            moved = move_by_display_row(
                text, buf.cursor_position, direction,
                info.window_width, first_prefix, cont_prefix, pref,
            )
            if moved is not None:
                buf.cursor_position, goal_x = moved
                sticky[:] = [(text, buf.cursor_position, goal_x)]
                return
        sticky.clear()
        if direction < 0:
            buf.auto_up(count=event.arg)
        else:
            buf.auto_down(count=event.arg)

    @kb.add("up")
    def _(event):
        _move(event, -1)

    @kb.add("down")
    def _(event):
        _move(event, 1)
