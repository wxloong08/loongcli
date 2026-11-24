"""让 Rich 在中日韩文字之间也能断行。

Rich 只在空格处断行（rich._wrap.words 用 `\\s*\\S+\\s*` 切词）：一长串没有空格的中文算一个「词」，
跟在空格后面、当前行剩余放不下时整串挪到下一行，上一行空半截（真机：「都用 plan」后面断行、「记录进度……」另起一行）。
这里把每个中日韩字符切成单独的词，任意两个汉字之间都能断；顺带做最简单的避头尾——
句读、右括号贴在前一个字后面（不出现在行首），左括号贴在后一个字前面（不留在行尾）。

做法是替换 rich._wrap.words：divide_line 每次调用都按名字取模块里的 words，替换后全局生效（所有 Rich 输出）。
依赖 Rich 私有模块，升级 Rich 时 tests/tui/test_cjk_wrap.py 会报。
"""
from __future__ import annotations

import re
from typing import Iterable

import rich._wrap as _rich_wrap

# 中日韩表意文字、假名、谚文、全角符号与标点
_CJK = "⺀-鿿가-힯豈-﫿︰-﹏＀-￯"
_CLOSE = "，。、；：！？）」』】》〉〕”’…—"     # 不放行首
_OPEN = "（「『【《〈〔“‘"                     # 不放行尾
_WORD = re.compile(
    rf"\s*(?:[{_OPEN}]*[{_CJK}][{_CLOSE}]*|[^\s{_CJK}]+[{_CLOSE}]*)\s*"
)


def _words(text: str) -> Iterable[tuple[int, int, str]]:
    """同 rich._wrap.words 的约定：逐个给出 (起点, 终点, 词含右侧空白)。"""
    position = 0
    match = _WORD.match(text, position)
    while match is not None:
        start, end = match.span()
        yield start, end, match.group(0)
        match = _WORD.match(text, end)


def install() -> None:
    _rich_wrap.words = _words
