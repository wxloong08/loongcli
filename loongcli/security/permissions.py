from __future__ import annotations
import os
import re
import shlex
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

ALWAYS_CONFIRM_COMMANDS = frozenset({
    "rm", "rmdir", "del", "rd",
    "kill", "pkill", "taskkill",
    "chmod", "chown", "icacls",
    "shutdown", "reboot",
    "mkfs", "dd", "format",
})

ALWAYS_CONFIRM_GIT_SUBS = frozenset({
    "push", "reset", "clean", "branch -D", "branch -d",
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


def _shell_pattern(command: str) -> tuple[str, bool]:
    """Extract a session-allowlist key from a shell command.

    Returns (pattern_key, always_confirm).
    always_confirm=True means the command must always ask, never session-allowed.
    """
    cmd = command.strip()
    try:
        tokens = shlex.split(cmd, posix=False)
    except ValueError:
        tokens = cmd.split()

    if not tokens:
        return "", True

    base = tokens[0].lower()

    # Strip path prefix: /usr/bin/python → python
    base = base.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    if base in ALWAYS_CONFIRM_COMMANDS:
        return "", True

    if base == "sudo" and len(tokens) > 1:
        actual = tokens[1].lower().rsplit("/", 1)[-1]
        if actual in ALWAYS_CONFIRM_COMMANDS:
            return "", True
        return f"shell:sudo {actual}", False

    if base == "git" and len(tokens) > 1:
        sub = tokens[1].lower()
        rest = " ".join(t.lower() for t in tokens[1:3])
        if sub in ALWAYS_CONFIRM_GIT_SUBS or rest in ALWAYS_CONFIRM_GIT_SUBS:
            return "", True
        return f"shell:git {sub}", False

    return f"shell:{base}", False


def _write_pattern(path: str) -> str:
    """Extract a session-allowlist key from a file write path."""
    try:
        parent = str(Path(path).resolve().parent)
    except (ValueError, OSError):
        parent = str(Path(path).parent)
    return f"write:{parent}"


class PermissionChecker:
    def __init__(
        self,
        mode: PermissionMode = PermissionMode.DEFAULT,
        extra_safe_prefixes: list[str] | None = None,
    ):
        self.mode = mode
        self.project_root = _detect_project_root()
        self._session_allowed: set[str] = set()
        if extra_safe_prefixes:
            SAFE_SHELL_PREFIXES.extend(extra_safe_prefixes)

    def record_approval(self, tool_name: str, args: dict) -> None:
        """Record a user approval so the same pattern won't ask again this session."""
        key = self._make_key(tool_name, args)
        if key:
            self._session_allowed.add(key)

    def check_tool(self, tool_name: str, args: dict, is_mcp: bool = False) -> tuple[Decision, str]:
        if tool_name == "shell":
            return self._check_shell(args.get("command", ""))

        if tool_name in ("write_file", "edit_file"):
            return self._check_file_write(args.get("path", ""))

        if is_mcp:
            if self.mode == PermissionMode.SKIP:
                return Decision.ALLOW, ""
            key = f"mcp:{tool_name}"
            if key in self._session_allowed:
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

        key, always_confirm = _shell_pattern(command)
        if not always_confirm and key in self._session_allowed:
            return Decision.ALLOW, ""

        return Decision.CONFIRM, "shell 命令需要确认"

    def _check_file_write(self, path: str) -> tuple[Decision, str]:
        if not path:
            return Decision.CONFIRM, "路径为空"

        for pattern in SENSITIVE_PATHS:
            if pattern.search(path):
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

        key = _write_pattern(path)
        if key in self._session_allowed:
            return Decision.ALLOW, ""

        return Decision.CONFIRM, "项目目录外写文件"

    def _make_key(self, tool_name: str, args: dict) -> str | None:
        """Build allowlist key for a tool call. Returns None if always-confirm."""
        if tool_name == "shell":
            key, always = _shell_pattern(args.get("command", ""))
            return key if not always else None
        if tool_name in ("write_file", "edit_file"):
            path = args.get("path", "")
            for pattern in SENSITIVE_PATHS:
                if pattern.search(path):
                    return None
            return _write_pattern(path)
        if tool_name.startswith("mcp_"):
            return f"mcp:{tool_name}"
        return None
