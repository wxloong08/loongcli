"""修复 UTF-16 代理码点（surrogate），避免发请求/存盘时 UTF-8 编码崩溃。

触发场景：在 Windows 控制台里粘贴含 emoji（码点 > U+FFFF）的文本，prompt_toolkit
经 Win32 控制台按 UTF-16 把它们当**代理对**（high U+D800–DBFF + low U+DC00–DFFF）交给
应用，未合并成真字符。代理码点按 Unicode 标准不是合法标量值，UTF-8 无法编码：

    'utf-8' codec can't encode characters ...: surrogates not allowed

它会一路带进对话历史、存盘与 LLM 请求体，在 UTF-8 编码处炸（recall 与主请求都中招）。
这是 prompt_toolkit 在 Windows 的已知问题（issue #2061），无法在读取层干净修复，故在输入
离开 prompt_toolkit 的边界修复。

关键：粘进来的代理对**本身就是该 emoji 的合法 UTF-16 编码**，只是没被解码完。所以正确做法
是把它当 UTF-16 重新解码（救回真 emoji），而非丢弃——忠实保留用户输入。
"""
from __future__ import annotations

import re

_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def repair_surrogates(s: str) -> str:
    """把字符串里的 UTF-16 代理码点修好，确保结果可 UTF-8 编码。

    - 相邻 high+low 代理对 → 经 UTF-16 往返重组回原本的 emoji（原样救回）。
    - 无法配对的孤立代理 → 由 ``errors='ignore'`` 丢弃。

    一步同时完成「救回成对 + 清掉孤立」。无代理码点时原样返回（快路径，零开销）。
    """
    if not isinstance(s, str) or not _SURROGATE_RE.search(s):
        return s
    return s.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "ignore")
