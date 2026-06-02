from __future__ import annotations
import re


class SecurityChecker:
    DANGEROUS_PATTERNS = [
        (re.compile(r"\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+|.*-rf\b)", re.IGNORECASE), "递归强制删除"),
        (re.compile(r"\bgit\s+push\s+.*--force\b", re.IGNORECASE), "强制推送"),
        (re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE), "硬重置"),
        (re.compile(r"\bdrop\s+table\b", re.IGNORECASE), "删除数据库表"),
        (re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE), "删除数据库记录"),
        (re.compile(r"\bformat\s+[a-zA-Z]:", re.IGNORECASE), "格式化磁盘"),
        (re.compile(r"\brmdir\s+.*[/\\]s\b", re.IGNORECASE), "递归删除目录"),
    ]

    SENSITIVE_PATHS = [
        (re.compile(r"(^|[/\\])\.env($|[/\\])"), ".env 文件"),
        (re.compile(r"(^|[/\\])\.ssh[/\\]"), ".ssh 目录"),
        (re.compile(r"(^|[/\\])\.git[/\\]config$"), ".git/config"),
        (re.compile(r"^[A-Z]:\\$"), "磁盘根目录"),
        (re.compile(r"^/$"), "系统根目录"),
        (re.compile(r"^~$"), "用户主目录"),
    ]

    def check_command(self, command: str) -> str | None:
        for pattern, reason in self.DANGEROUS_PATTERNS:
            if pattern.search(command):
                return reason
        return None

    def check_path(self, path: str) -> str | None:
        for pattern, reason in self.SENSITIVE_PATHS:
            if pattern.search(path):
                return reason
        return None

    def is_dangerous_command(self, command: str) -> bool:
        return self.check_command(command) is not None

    def is_sensitive_path(self, path: str) -> bool:
        return self.check_path(path) is not None
