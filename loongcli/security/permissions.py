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

# 能在首 token 之后追加或重定向出「第二条命令」的操作符——它们可绕过对首 token
# 的前缀/白名单匹配。命令替换（$(...)、`...`、<(...)）与重定向（>、<）一律不放行：
# 像 `echo ` 这种「安全前缀」照样能写文件（echo x > f）或执行代码（echo $(rm -rf x)）。
_SUBSTITUTION_RE = re.compile(r"\$\(|`|<\(|>|<")
# 用于把一条命令按链式操作符拆成可逐段独立校验的子命令。
_CHAIN_SPLIT_RE = re.compile(r"&&|\|\||;|\|")


def _split_commands(command: str) -> list[str]:
    """按链式操作符（&&、||、;、|）把 shell 命令拆成若干子命令。

    朴素拆分——带引号的操作符（如 echo "a;b"）也会被拆开，但这只会多要一次确认，
    绝不会漏掉一次。对安全闸而言，偏向「多确认」正是正确方向。
    """
    return [p.strip() for p in _CHAIN_SPLIT_RE.split(command) if p.strip()]


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
        # 计划审批选「批准并自动接受编辑」后置位：项目外非敏感写免确认（敏感路径底线不动）。
        # 注意 loongcli 项目内写本就 ALLOW，此标志的增量只是项目外路径。
        self.auto_accept_edits: bool = False
        if extra_safe_prefixes:
            SAFE_SHELL_PREFIXES.extend(extra_safe_prefixes)

    def record_approval(self, tool_name: str, args: dict, is_mcp: bool = False) -> None:
        """记录一次用户批准，使同类调用本会话内不再重复询问。

        shell 按子命令片段逐个记 key——check 侧要求链式命令每个片段都在白名单，
        只记整条首 token（如 `cd x && python y` 只记 shell:cd）会导致同一条命令
        下次仍要确认。always_confirm 类片段（rm/kill 等）永不入白名单。
        """
        if tool_name == "shell":
            for sc in _split_commands(args.get("command", "")):
                key, always = _shell_pattern(sc)
                if key and not always:
                    self._session_allowed.add(key)
            return
        key = self._make_key(tool_name, args, is_mcp=is_mcp)
        if key:
            self._session_allowed.add(key)

    def check_tool(self, tool_name: str, args: dict, is_mcp: bool = False) -> tuple[Decision, str]:
        if tool_name == "shell":
            return self._check_shell(args.get("command", ""))

        if tool_name in ("write_file", "edit_file"):
            return self._check_file_write(args.get("path", ""))

        if tool_name == "read_file":
            return self._check_file_read(args.get("path", ""))

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

        # 前缀/白名单命中只证明「首 token」安全。命令替换和重定向能把第二条命令
        # 藏进参数里，或把输出写到任意文件，所以它们一律不走快路径。
        if _SUBSTITUTION_RE.search(command):
            return Decision.CONFIRM, "shell 命令含命令替换或重定向，需要确认"

        # 链式（a && b）、管道（a | b）、顺序（a; b）都能追加一条未经校验的第二命令。
        # 仅当「每个」子命令都独立安全时才自动放行。
        subcommands = _split_commands(command)
        if subcommands and all(self._shell_fragment_safe(sc) for sc in subcommands):
            return Decision.ALLOW, ""

        return Decision.CONFIRM, "shell 命令需要确认"

    def _shell_fragment_safe(self, command: str) -> bool:
        """单条（未链式）命令命中安全前缀或会话白名单则返回 True。"""
        cmd_stripped = command.strip()
        for prefix in SAFE_SHELL_PREFIXES:
            if cmd_stripped == prefix.strip() or cmd_stripped.startswith(prefix):
                return True
        key, always_confirm = _shell_pattern(command)
        return (not always_confirm) and key in self._session_allowed

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

        # 「批准并自动接受编辑」生效范围：项目外的非敏感写（敏感路径已在上方拦下）
        if self.auto_accept_edits:
            return Decision.ALLOW, ""

        key = _write_pattern(path)
        if key in self._session_allowed:
            return Decision.ALLOW, ""

        return Decision.CONFIRM, "项目目录外写文件"

    def _check_file_read(self, path: str) -> tuple[Decision, str]:
        """敏感路径读取门：.env/.ssh/credentials 等是密钥外泄的主通道，读取一律确认。

        刻意不进会话白名单（_make_key 无 read 分支）——每次读敏感文件都值得人眼
        看一次。普通文件读取放行（与 Claude Code 的读策略一致）。
        """
        if not path or self.mode == PermissionMode.SKIP:
            return Decision.ALLOW, ""
        for pattern in SENSITIVE_PATHS:
            if pattern.search(path):
                return Decision.CONFIRM, "敏感文件读取"
        return Decision.ALLOW, ""

    def _make_key(self, tool_name: str, args: dict, is_mcp: bool = False) -> str | None:
        """构造工具调用的会话白名单 key；始终需确认的调用返回 None。"""
        if tool_name == "shell":
            key, always = _shell_pattern(args.get("command", ""))
            return key if not always else None
        if tool_name in ("write_file", "edit_file"):
            path = args.get("path", "")
            for pattern in SENSITIVE_PATHS:
                if pattern.search(path):
                    return None
            return _write_pattern(path)
        # MCP 由调用方显式告知（工具实例的 is_mcp），不再按名字前缀猜——
        # 真实 MCP 工具名形如 server__tool，永不以 "mcp_" 开头。key 与
        # check_tool 的 is_mcp 分支保持一致，批准才能被正确记住。
        if is_mcp:
            return f"mcp:{tool_name}"
        return None
