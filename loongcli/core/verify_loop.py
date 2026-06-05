from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_VERIFY_ROUNDS = 3

_EXIT_CODE_RE = re.compile(r"\[exit code: (\d+)")
_PYTEST_PASSED_RE = re.compile(r"(\d+) passed")
_PYTEST_FAILED_RE = re.compile(r"(\d+) failed")


def detect_test_failure(shell_output: str) -> bool:
    """Determine if shell output indicates test failure.

    Handles false positives: PowerShell returns exit code 1 when pytest
    prints deprecation warnings to stderr, even when all tests pass.
    """
    m = _EXIT_CODE_RE.search(shell_output)
    if not m:
        return False
    exit_code = int(m.group(1))
    if exit_code == 0:
        return False

    passed = _PYTEST_PASSED_RE.search(shell_output)
    failed = _PYTEST_FAILED_RE.search(shell_output)
    if passed and not failed:
        return False
    if passed and failed and int(failed.group(1)) == 0:
        return False

    return True


@dataclass
class VerifyState:
    """Tracks verification progress within a single agent run."""
    changed_files: list[str] = field(default_factory=list)
    test_command: str | None = None
    round: int = 0
    last_error: str | None = None
    test_failed: bool = False

    @property
    def is_active(self) -> bool:
        return self.round > 0 and self.round <= MAX_VERIFY_ROUNDS

    @property
    def is_exhausted(self) -> bool:
        return self.round > MAX_VERIFY_ROUNDS

    def start(self, changed_files: list[str], test_command: str | None) -> None:
        self.changed_files = list(changed_files)
        self.test_command = test_command
        self.round = 1
        self.last_error = None
        self.test_failed = False

    def reset(self) -> None:
        self.changed_files.clear()
        self.test_command = None
        self.round = 0
        self.last_error = None
        self.test_failed = False


def build_verify_prompt(
    changed_files: list[str],
    test_command: str | None,
    round_number: int,
    last_error: str | None = None,
) -> str:
    """Build the verification prompt injected into the conversation.

    Round 1: initial verification request
    Round 2+: retry with previous error context
    """
    files_list = "\n".join(f"- {f}" for f in changed_files) if changed_files else "(unknown)"

    if round_number == 1:
        lines = [
            "## 验证",
            "",
            f"本轮修改了 {len(changed_files)} 个文件:",
            files_list,
        ]
        if test_command:
            lines.extend([
                "",
                f"项目测试命令: `{test_command}`",
                "",
                f"请用 Shell 工具运行: `{test_command}`",
                "如果测试失败，分析报错并修复代码，然后重新验证。",
                f"最多 {MAX_VERIFY_ROUNDS} 轮，若仍失败则报告具体原因。",
            ])
        else:
            lines.extend([
                "",
                "未检测到项目测试命令。请自行判断如何验证修改是否正确：",
                "- 检查项目文档（LOONG.md）中是否有测试说明",
                "- 用 Shell 工具运行你认为合适的验证命令",
                "- 至少执行一次验证再报告完成",
                f"最多 {MAX_VERIFY_ROUNDS} 轮，若仍失败则报告具体原因。",
            ])
    else:
        lines = [
            f"## 验证 — 第 {round_number} 轮重试",
            "",
            f"上一轮验证{'失败' if last_error else '未通过'}。",
        ]
        if last_error:
            # Truncate long error messages
            truncated = last_error[:3000] + ("..." if len(last_error) > 3000 else "")
            lines.append(f"```\n{truncated}\n```")
        lines.extend([
            "",
            "请分析失败原因，修复代码，然后重新运行验证。",
            f"剩余尝试次数: {MAX_VERIFY_ROUNDS - round_number + 1}",
        ])

    return "\n".join(lines)
