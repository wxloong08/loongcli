from __future__ import annotations
import os
import re
import subprocess
from enum import Enum
from pathlib import Path


class PermissionMode(Enum):
    DEFAULT = "default"
    SKIP = "skip-permissions"


class Decision(Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


CATASTROPHIC_PATTERNS = [
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)[/\\](\s|$)"), "递归删除根目录"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)/[a-zA-Z]+\s*$"), "递归删除顶级目录"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)~(\s|$)"), "递归删除家目录"),
    (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\s+)[A-Z]:\\(\s|$)", re.IGNORECASE), "递归删除磁盘根目录"),
    (re.compile(r"\bformat\s+[A-Z]:", re.IGNORECASE), "格式化磁盘"),
    (re.compile(r"\bdd\s+.*of\s*=\s*/dev/[a-z]", re.IGNORECASE), "覆写磁盘设备"),
    (re.compile(r"\bmkfs\b", re.IGNORECASE), "格式化文件系统"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;"), "fork bomb"),
]

SAFE_SHELL_PREFIXES = [
    "git status", "git log", "git diff", "git branch", "git show",
    "git remote", "git tag", "git stash list", "git rev-parse",
    "ls", "dir", "pwd", "cd ",
    "cat ", "head ", "tail ", "wc ",
    "echo ", "date", "whoami", "hostname",
    "python --version", "python3 --version", "node --version",
    "pip list", "pip show", "npm list", "npm --version",
    "type ", "where ", "which ",
    "env", "set", "printenv",
]

SENSITIVE_PATHS = [
    re.compile(r"(^|[/\\])\.env($|[/\\])"),
    re.compile(r"(^|[/\\])\.ssh[/\\]"),
    re.compile(r"(^|[/\\])\.git[/\\]config$"),
    re.compile(r"(^|[/\\])credentials", re.IGNORECASE),
    re.compile(r"(^|[/\\])\.aws[/\\]"),
    re.compile(r"(^|[/\\])\.kube[/\\]"),
]

UNSAFE_PROJECT_ROOTS = frozenset({
    "/", os.path.expanduser("~"),
    "C:\\", "D:\\", "E:\\",
})


def _detect_project_root() -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            root = Path(result.stdout.strip()).resolve()
            if str(root) not in UNSAFE_PROJECT_ROOTS:
                return root
    except Exception:
        pass

    cwd = Path.cwd().resolve()
    if str(cwd) not in UNSAFE_PROJECT_ROOTS:
        return cwd
    return None


class PermissionChecker:
    def __init__(self, mode: PermissionMode = PermissionMode.DEFAULT):
        self.mode = mode
        self.project_root = _detect_project_root()

    def check_tool(self, tool_name: str, args: dict, is_mcp: bool = False) -> tuple[Decision, str]:
        if tool_name == "shell":
            return self._check_shell(args.get("command", ""))

        if tool_name in ("write_file", "edit_file"):
            return self._check_file_write(args.get("path", ""))

        if is_mcp:
            if self.mode == PermissionMode.SKIP:
                return Decision.ALLOW, ""
            return Decision.CONFIRM, "MCP 外部工具"

        return Decision.ALLOW, ""

    def _check_shell(self, command: str) -> tuple[Decision, str]:
        for pattern, reason in CATASTROPHIC_PATTERNS:
            if pattern.search(command):
                return Decision.DENY, reason

        if self.mode == PermissionMode.SKIP:
            return Decision.ALLOW, ""

        cmd_stripped = command.strip()
        for prefix in SAFE_SHELL_PREFIXES:
            if cmd_stripped == prefix.strip() or cmd_stripped.startswith(prefix):
                return Decision.ALLOW, ""

        return Decision.CONFIRM, "shell 命令需要确认"

    def _check_file_write(self, path: str) -> tuple[Decision, str]:
        if not path:
            return Decision.CONFIRM, "路径为空"

        for pattern in SENSITIVE_PATHS:
            if pattern.search(path):
                if self.mode == PermissionMode.SKIP:
                    return Decision.CONFIRM, "敏感文件"
                return Decision.CONFIRM, "敏感文件"

        if self.project_root:
            try:
                resolved = Path(path).resolve()
                resolved.relative_to(self.project_root)
                return Decision.ALLOW, ""
            except ValueError:
                pass

        if self.mode == PermissionMode.SKIP:
            return Decision.ALLOW, ""
        return Decision.CONFIRM, "项目目录外写文件"
