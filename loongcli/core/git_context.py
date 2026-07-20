from __future__ import annotations
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GitContext:
    is_repo: bool = False
    branch: str = ""
    main_branch: str = ""
    dirty_files: list[str] = field(default_factory=list)
    recent_commits: list[str] = field(default_factory=list)
    remote_url: str = ""
    ahead_behind: str = ""

    def to_prompt(self) -> str:
        if not self.is_repo:
            return "- Git 仓库: 否"
        lines = [
            f"- Git 仓库: 是",
            f"- 当前分支: {self.branch}",
        ]
        if self.main_branch and self.main_branch != self.branch:
            lines.append(f"- 主分支: {self.main_branch}")
        if self.ahead_behind:
            lines.append(f"- 与远程: {self.ahead_behind}")
        if self.dirty_files:
            n = len(self.dirty_files)
            preview = self.dirty_files[:15]
            files_str = ", ".join(preview)
            if n > 15:
                files_str += f" ... 共 {n} 个文件"
            lines.append(f"- 未提交变更: {files_str}")
        if self.recent_commits:
            lines.append("- 最近提交:")
            for c in self.recent_commits[:8]:
                lines.append(f"  {c}")
        # 快照标签：git 状态只在系统提示构建时采集，不随本会话操作实时更新——
        # 长会话后 dirty 清单几乎必然过时，不标注模型会把旧快照当现实。
        lines.append("- 以上 git 状态是采集时的快照，不随本会话操作实时更新；需要当前状态时用 git status 查询")
        return "\n".join(lines)


def collect_git_context(cwd: Path | None = None) -> GitContext:
    cwd = cwd or Path.cwd()
    ctx = GitContext()

    if not _run("git rev-parse --is-inside-work-tree", cwd):
        return ctx
    ctx.is_repo = True

    ctx.branch = _run("git rev-parse --abbrev-ref HEAD", cwd) or ""
    ctx.main_branch = _detect_main_branch(cwd)
    ctx.remote_url = _run("git remote get-url origin", cwd) or ""
    ctx.ahead_behind = _get_ahead_behind(cwd)
    ctx.dirty_files = _get_dirty_files(cwd)
    ctx.recent_commits = _get_recent_commits(cwd)

    return ctx


def _run(cmd: str, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=str(cwd),
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _detect_main_branch(cwd: Path) -> str:
    for candidate in ("main", "master"):
        if _run(f"git rev-parse --verify {candidate}", cwd):
            return candidate
    return ""


def _get_dirty_files(cwd: Path) -> list[str]:
    output = _run("git status --short", cwd)
    if not output:
        return []
    lines = output.splitlines()
    return [line.strip() for line in lines if line.strip()]


def _get_recent_commits(cwd: Path, n: int = 8) -> list[str]:
    output = _run(f"git log --oneline -{n}", cwd)
    if not output:
        return []
    return output.splitlines()


def _get_ahead_behind(cwd: Path) -> str:
    upstream = _run("git rev-parse --abbrev-ref @{upstream}", cwd)
    if not upstream:
        return ""
    output = _run(f"git rev-list --left-right --count {upstream}...HEAD", cwd)
    if not output:
        return ""
    parts = output.split()
    if len(parts) == 2:
        behind, ahead = int(parts[0]), int(parts[1])
        if ahead == 0 and behind == 0:
            return "与远程同步"
        segments = []
        if ahead > 0:
            segments.append(f"领先 {ahead}")
        if behind > 0:
            segments.append(f"落后 {behind}")
        return ", ".join(segments)
    return ""
