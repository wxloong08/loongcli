"""验证提醒：改了代码、收尾前没有通过的测试，就提醒模型验证一次。

只提醒、不重试。2026-11-15 复盘全部会话里的 5 次验证轮：「改了不测就收尾」真实出现过 1 次（提醒后模型补跑测试
并加了回归断言）；「测试失败还报通过」0 次——5 次真实测试失败都是模型自己看到、自己修好的。原先「判定失败 →
自动重试最多 3 轮」的机制没有一次真实收益，却靠启发式判失败，误判一次就多跑一轮并注入一条假的失败消息，故删除。
"""
from __future__ import annotations

import re
from pathlib import Path

_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".kt", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".scala", ".sh", ".bash",
    ".sql", ".r", ".lua", ".pl", ".ex", ".exs", ".zig", ".v",
})


def filter_code_files(files: list[str]) -> list[str]:
    return [f for f in files if Path(f).suffix.lower() in _CODE_EXTENSIONS]


_EXIT_CODE_RE = re.compile(r"\[exit code: (\d+)")
# 测试命令识别。解释器部分不做要求：真机上常是带路径的 venv python.exe，甚至是 shell 变量
# （`$VENV -m unittest discover`），所以按「-m pytest / -m unittest」认，不按 `python` 这个词认。
_TEST_CMD_RE = re.compile(
    r"(?:^|&&|\|\||[;|])\s*pytest\b"
    r"|(?:^|\s)-m\s+(?:pytest|unittest)\b"
    r"|(?:npm|go|cargo|make)\s+test\b",
    re.IGNORECASE,
)

# 测试框架自己打印的总结行。测试命令常接 `| tail`，管道的退出码是 tail 的（恒为 0），只看退出码会把
# 失败当成通过——2026-11-15 dogfooding 三轮里 33 次 unittest 调用有 26 次接了管道。
_UNITTEST_FAILED_RE = re.compile(r"^FAILED \(", re.MULTILINE)
_UNITTEST_RAN_RE = re.compile(r"^Ran \d+ tests? in ", re.MULTILINE)
_UNITTEST_OK_RE = re.compile(r"^OK\b", re.MULTILINE)
_NO_TESTS_RAN_RE = re.compile(r"\bno tests ran\b", re.IGNORECASE)
# pytest 计数：`3 failed, 977 passed in 90.70s`、`1 error in 0.1s`；xfailed/xpassed 因前缀 x 不会命中
_PYTEST_COUNT_RE = re.compile(r"(\d+) (passed|failed|errors?)\b")


def is_test_command(command: str) -> bool:
    """命令是否在跑测试。认不出的自定义测试脚本（如 `python test_matching.py`）最多多换来一次提醒。"""
    return bool(_TEST_CMD_RE.search(command))


def _pytest_counts(output: str) -> dict[str, int] | None:
    """取最后一行 pytest 计数总结；没有则返回 None。"""
    for line in reversed(output.splitlines()):
        found = _PYTEST_COUNT_RE.findall(line)
        if found:
            counts = {"passed": 0, "failed": 0, "error": 0}
            for n, kind in found:
                counts["error" if kind.startswith("error") else kind] += int(n)
            return counts
    return None


def detect_test_failure(shell_output: str) -> bool:
    """判断一次测试命令的输出是否表示失败。

    先看测试框架自己的总结行——它不受管道退出码影响，也覆盖了 PowerShell 的误报（pytest 往 stderr
    打弃用警告时退出码为 1，但总结行全通过）。找不到总结行（输出被 head 截掉、非 Python 测试）才看退出码。
    """
    if _UNITTEST_FAILED_RE.search(shell_output) or _NO_TESTS_RAN_RE.search(shell_output):
        return True
    counts = _pytest_counts(shell_output)
    if counts is not None:
        return counts["failed"] > 0 or counts["error"] > 0
    if _UNITTEST_RAN_RE.search(shell_output) and _UNITTEST_OK_RE.search(shell_output):
        return False
    m = _EXIT_CODE_RE.search(shell_output)
    return bool(m) and int(m.group(1)) != 0


def build_verify_prompt(changed_files: list[str], test_command: str | None) -> str:
    """生成注入对话的验证提醒（`## 验证` 开头：TUI 按这个前缀把它渲染成系统提示，不当用户消息）。"""
    files_list = "\n".join(f"- {f}" for f in changed_files) if changed_files else "(unknown)"
    lines = [
        "## 验证",
        "",
        f"本轮修改了 {len(changed_files)} 个文件，收尾前还没有通过的测试:",
        files_list,
        "",
    ]
    if test_command:
        lines += [
            f"项目测试命令: `{test_command}`",
            "",
            "请用 Shell 工具运行它；失败就读报错、修复后再跑一次，然后报告结果。",
        ]
    else:
        lines += [
            "未检测到项目测试命令。请自行判断如何验证修改是否正确：",
            "- 检查项目文档（LOONG.md）中是否有测试说明",
            "- 用 Shell 工具运行你认为合适的验证命令",
            "验证后报告结果；无法验证就明确说出来，不要假装成功。",
        ]
    return "\n".join(lines)
