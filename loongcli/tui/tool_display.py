"""工具调用的 TUI 展示摘要器（方案 A：● 工具行 + ⎿ 折叠结果）。

纯函数层，与渲染解耦便于测试。原则：
- 参数只留核心值（相对路径、pattern、命令），丢 k=v 键名与大段字符串倾倒；
- 结果统计化（grep→N 处 M 文件、read→N 行、edit→真 diff），原文倾倒只留给错误；
- edit_file 的 diff 由 args 里的 old_string/new_string 本地 difflib 计算，
  不改工具协议、不多喂 LLM 一个字。
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

from rich.text import Text

# 展示上限
DIFF_MAX_LINES = 12
ERROR_MAX_LINES = 5
DEFAULT_MAX_CHARS = 100
_ERROR_PREFIXES = ("错误：", "⚠", "MCP tool error")

# grep 输出行型：rel:line: content（rel 可能带 Windows 盘符，用 :数字: 锚定）
_GREP_LINE_RE = re.compile(r"^(.+?):(\d+): ")


def relpath(path: str) -> str:
    """cwd 内的路径相对化展示，库外原样返回。"""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except (ValueError, OSError):
        return path


def _clip(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[:limit] + "…"


def arg_summary(tool_name: str, args: dict) -> str:
    """工具参数的一行摘要：只留核心值。"""
    if tool_name in ("read_file", "write_file", "edit_file"):
        path = relpath(str(args.get("path", "")))
        if tool_name == "read_file" and (args.get("offset") or args.get("limit")):
            start = int(args.get("offset") or 1)
            limit = args.get("limit")
            span = f":{start}-{start + int(limit) - 1}" if limit else f":{start}-"
            return f"{path}{span}"
        return path
    if tool_name == "grep":
        pattern = _clip(str(args.get("pattern", "")), 40)
        path = relpath(str(args.get("path", ""))) if args.get("path") else ""
        return f"{pattern}, {path}" if path else pattern
    if tool_name == "glob":
        return _clip(str(args.get("pattern", "")), 60)
    if tool_name == "shell":
        return _clip(str(args.get("command", "")), 60)
    # 未知 / MCP 工具：值序列，各截 40，丢键名
    return ", ".join(_clip(str(v), 40) for v in args.values() if v is not None)


def _is_error(result: str) -> bool:
    return result.startswith(_ERROR_PREFIXES)


def _error_lines(result: str) -> list[Text]:
    lines = [l for l in result.splitlines() if l.strip()][:ERROR_MAX_LINES]
    return [Text(l, style="red") for l in lines]


def _edit_diff_lines(args: dict) -> list[Text]:
    """old_string → new_string 的 unified diff（去头），+ 绿 - 红上下文暗。"""
    old = str(args.get("old_string", "")).splitlines()
    new = str(args.get("new_string", "")).splitlines()
    raw = [
        l for l in difflib.unified_diff(old, new, lineterm="", n=1)
        if not l.startswith(("---", "+++", "@@"))
    ]
    out: list[Text] = []
    for l in raw[:DIFF_MAX_LINES]:
        # 增删行用背景色高亮（Claude Code 式）——diff 是折叠输出里唯一需要用户
        # 认真审阅的内容，纯前景色在一屏工具块里没有视觉权重
        # 显式十六进制：调色板名（white/bright_white/dark_*）经终端主题映射后
        # 亮度不可控（真机反馈两轮都发灰）。truecolor 直接钉死纯白字 + 深色底。
        if l.startswith("+"):
            out.append(Text(l, style="bold #ffffff on #1d5f2d"))
        elif l.startswith("-"):
            out.append(Text(l, style="bold #ffffff on #6e1e1e"))
        else:
            out.append(Text(l, style="dim"))
    if len(raw) > DIFF_MAX_LINES:
        out.append(Text(f"… 还有 {len(raw) - DIFF_MAX_LINES} 行", style="dim"))
    return out


def _shell_key_line(result: str) -> str:
    """挑 shell 输出的关键行：测试统计行优先，其次最后一行非空。"""
    lines = [l for l in result.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        stripped = line.strip().strip("=").strip("-").strip()
        if any(kw in stripped for kw in ("passed", "failed", "error")):
            return _clip(stripped, 120)
    last = lines[-1].strip()
    if last.startswith("[exit code:") and len(lines) >= 2:
        last = f"{lines[-2].strip()}  {last}"
    return _clip(last, 120)


def result_lines(tool_name: str, args: dict, result: str) -> tuple[bool, list[Text]]:
    """工具结果 → (是否成功, ⎿ 块展示行)。"""
    if not result:
        return True, []
    if _is_error(result):
        return False, _error_lines(result)

    if tool_name == "read_file":
        first = result.splitlines()[0] if result.splitlines() else ""
        # 短占位文本（如图片提示）原样透传——必须限单行，否则 read_file 带
        # [第 a-b 行] 范围头的多行结果会整段漏进折叠块（真机：正文顶格泄露）
        if result.startswith("[") and "\n" not in result and len(result) <= DEFAULT_MAX_CHARS:
            return True, [Text(result, style="dim")]
        # 带范围头的读取直接用范围头当摘要（比行数统计更准，不把头行算进行数）
        if first.startswith("[第") or first.startswith("[编码"):
            return True, [Text(_clip(first, DEFAULT_MAX_CHARS), style="dim")]
        return True, [Text(f"{len(result.splitlines())} 行", style="dim")]

    if tool_name == "glob":
        if result.startswith("未找到"):
            return True, [Text(result.splitlines()[0], style="dim")]
        return True, [Text(f"{len(result.splitlines())} 个文件", style="dim")]

    if tool_name == "grep":
        lines_ = result.splitlines()
        files: set[str] = set()
        hits = 0
        for line in lines_:
            m = _GREP_LINE_RE.match(line)
            if m:
                hits += 1
                files.add(m.group(1))
        if hits:
            return True, [Text(f"{hits} 处匹配 · {len(files)} 个文件", style="dim")]
        # "未找到"行可能在 [搜索根: …] 头之后，不一定是首行
        nf = next((l for l in lines_ if l.startswith("未找到")), None)
        if nf:
            return True, [Text(_clip(nf, DEFAULT_MAX_CHARS), style="dim")]
        return True, [Text(_clip(lines_[0], DEFAULT_MAX_CHARS), style="dim")]

    if tool_name == "edit_file":
        lines: list[Text] = []
        m = re.search(r"（(fuzzy \d+%)", result)
        if m:
            lines.append(Text(m.group(1), style="yellow"))
        lines.extend(_edit_diff_lines(args))
        return True, lines

    if tool_name == "write_file":
        n = len(str(args.get("content", "")).splitlines())
        return True, [Text(f"写入 {n} 行", style="dim")]

    if tool_name == "shell":
        failed = "[exit code:" in result
        key = _shell_key_line(result)
        return not failed, [Text(key, style="red" if failed else "dim")]

    # 默认 / MCP 工具：首行截断
    first = result.splitlines()[0].strip() if result.strip() else ""
    text = _clip(first, DEFAULT_MAX_CHARS)
    if "\n" in result.strip() and not text.endswith("…"):
        text += " …"
    return True, [Text(text, style="dim")]
