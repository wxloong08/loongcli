"""滚动区里的「框」：标题 + 缩进内容的分节（不画边框），外加欢迎页 Logo、无背景代码块。

为什么不用 Rich 的 Panel：打出去的内容进了终端历史、程序没法再重画；窗口拉窄时终端把超宽的行折断，
满宽框的右边框、上下边线都被折到下一行，碎成一片（真机）。规律是：行尾是可见字符（或带背景色的空格）的行才会碎，
纯文字行尾的空格终端会裁掉，折行后照样能读。所以分节只用加粗标题 + 缩进区分，不画任何线（左竖线版观感差，用户否了）。
"""
from __future__ import annotations

import textwrap

from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text


class Section:
    """一节输出：加粗标题（缩进 indent）+ 内容（再缩进 2）+ 可选暗色脚注。内容可以是带标记的字符串或任意 Rich 对象。"""

    def __init__(self, body, title: str | None = None, *, footer: str | None = None, indent: int = 2):
        # 字符串按 Rich 标记解析、不走默认高亮（同 Panel）：否则 /help 这类路径会被高亮器拆成几段上色。
        # 先去掉公共缩进：各命令拼的行自带两格缩进（Panel 时代的习惯），再叠上这里的缩进就缩进过头了
        self.body = Text.from_markup(textwrap.dedent(body)) if isinstance(body, str) else body
        self.title = title
        self.footer = footer
        self.indent = indent

    def __rich_console__(self, console, options):
        if self.title:
            yield Padding(Text(self.title, style="bold cyan"), (0, 0, 0, self.indent))
        yield Padding(self.body, (0, 0, 0, self.indent + 2))
        if self.footer:
            yield Padding(Text(self.footer, style="dim"), (0, 0, 0, self.indent))


def section(body, title: str | None = None, **kw) -> Section:
    return Section(body, title, **kw)


def code_block(code: str, lexer: str, theme: str):
    """代码块：只留语法高亮、不铺背景色，比正文多缩进 2 列以示区分（同 Claude Code）。

    背景色怎么都会碎：Rich 默认补满整行，限到代码宽度后窗口窄过代码照样碎、而且碎成参差的色块（真机），
    带背景色的空格终端不裁。用户拍板去掉背景。"""
    return Syntax(code, lexer, theme=theme, word_wrap=True, background_color="default", padding=(0, 0, 0, 2))


# 3 行高的方块字 LOONG（约 26 列宽：拉到很窄也不会被折断）
_LOGO = (
    "█    █▀▀█ █▀▀█ █▄  █ █▀▀▀",
    "█    █  █ █  █ █ ▀▄█ █ ▀█",
    "█▄▄▄ █▄▄█ █▄▄█ █   █ █▄▄█",
)
_FROM, _TO = (0x22, 0xD3, 0xEE), (0x3B, 0x82, 0xF6)    # 青 → 蓝


def logo(suffix: Text | None = None) -> Text:
    """欢迎页 Logo：按列从青到蓝渐变；suffix 接在最后一行右边（版本号）。"""
    width = max(len(row) for row in _LOGO)
    out = Text()
    for i, row in enumerate(_LOGO):
        for x, ch in enumerate(row):
            t = x / max(1, width - 1)
            r, g, b = (round(a + (c - a) * t) for a, c in zip(_FROM, _TO))
            out.append(ch, style=f"bold #{r:02x}{g:02x}{b:02x}")
        if i == len(_LOGO) - 1 and suffix is not None:
            out.append("  ")
            out.append_text(suffix)
        if i < len(_LOGO) - 1:
            out.append("\n")
    return out
