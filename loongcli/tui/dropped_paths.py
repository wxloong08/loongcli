"""认出「拖进终端的文件」：粘贴内容整段都是存在的文件路径，就当成拖文件。

终端把拖入的文件当一次粘贴送来，但格式各家不一（真机：两张图一起拖，变成两段裸路径）：
- 单个路径，带或不带引号；多个路径空格分隔（Windows Terminal 只给含空格的路径加引号）
- 每行一个路径（部分文件管理器 / 终端）
- file:///C:/... URI，可能有 %20 之类的编码（从浏览器、部分应用拖出）
- Git Bash 风格的 /c/Users/...
- 含空格却没加引号的单个路径：整串当一个路径再试一次
只有「每一段都是存在的文件」才算拖文件；夹着别的字（「看看 C:\\a.png 这张」）就是普通粘贴，交给提交时的路径识别。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import unquote

_TOKEN = re.compile(r'"[^"]+"|\'[^\']+\'|\S+')
_MAX_TOKENS = 20          # 拖文件不会一次几十个；段数多了多半是普通文字，不逐个查盘
_MAX_CHARS = 8000


def _normalize(token: str) -> str:
    t = token.strip().strip('"').strip("'")
    if t.lower().startswith("file://"):
        t = unquote(t[len("file://"):])
        if re.match(r"^/[A-Za-z]:", t):            # file:///C:/x → C:/x
            t = t[1:]
    elif os.name == "nt":
        m = re.match(r"^/([A-Za-z])/(.*)", t)      # Git Bash：/c/Users/x → C:/Users/x
        if m:
            t = f"{m.group(1).upper()}:/{m.group(2)}"
    return str(Path(t)) if t else t


def split_dropped_paths(data: str) -> list[str] | None:
    """粘贴内容是不是一组拖入的文件：是就返回各文件路径（按出现顺序、规范成本机格式），不是返回 None。"""
    text = data.strip()
    if not text or len(text) > _MAX_CHARS:
        return None
    tokens = _TOKEN.findall(text)
    if 0 < len(tokens) <= _MAX_TOKENS:
        paths = [_normalize(t) for t in tokens]
        if all(p and Path(p).is_file() for p in paths):
            return paths
    whole = _normalize(text)
    if "\n" not in whole and whole and Path(whole).is_file():
        return [whole]
    return None
