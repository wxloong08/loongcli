from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from loongcli.tools.base import Tool
from loongcli.tools.errors import ToolError

_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB per file
_MAX_TOTAL_FILES = 5000
_SKIP_DIRS = frozenset({
    ".venv", "venv", ".env", "node_modules", "__pycache__",
    ".git", ".hg", ".svn", ".tox", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "dist", "build", ".eggs",
})


def _is_walk_pattern(glob: str) -> bool:
    """'**/名字型' 模式（含默认 '**/*'）可走 os.walk 剪枝快路径。"""
    return glob.startswith("**/") and "/" not in glob[3:] and "**" not in glob[3:]


def _walk_files(base: Path, glob: str):
    """os.walk 遍历，跳过目录原地剪枝——压根不进去。

    rglob 会枚举 .venv/.git 内的几万个条目，虽被逐条跳过却照样消耗扫描预算
    （真机：项目根下真正的项目文件只轮到 16 个就触顶截断）。剪枝后预算只花在
    真实候选文件上。仅处理 _is_walk_pattern 的模式，其余走原 glob 路径。
    """
    suffix = glob[3:]
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if fnmatch.fnmatch(name, suffix):
                yield Path(dirpath) / name


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents using a regex pattern. "
        "Returns matching lines with file paths and line numbers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Default: current working directory",
            },
            "glob": {
                "type": "string",
                "description": "File pattern filter (e.g. '*.py'). Default: all files",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case insensitive search. Default: false",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of matching lines to return. Default: 500",
            },
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "**/*",
        case_insensitive: bool = False,
        limit: int = 500,
    ) -> str:
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ToolError(f"无效的正则表达式 — {e}", retryable=False)

        base = Path(path).resolve()
        single_file = base.is_file()
        if not single_file and not base.is_dir():
            return f"错误：'{path}' 不存在"

        results: list[str] = []
        files_searched = 0
        files_skipped: list[str] = []
        total_files = 0

        if single_file:
            file_iter = [base]
        elif _is_walk_pattern(glob):
            file_iter = _walk_files(base, glob)
        else:
            try:
                file_iter = sorted(base.rglob(glob) if "**" in glob else base.glob(glob))
            except OSError as e:
                return f"错误：扫描目录失败 — {e}"
        for file_path in file_iter:
            total_files += 1
            if total_files > _MAX_TOTAL_FILES:
                break
            if not single_file:
                try:
                    parts = file_path.relative_to(base).parts
                except ValueError:
                    parts = ()
                if any(p in _SKIP_DIRS for p in parts):
                    continue

            # is_file/stat/read 都可能抛 OSError——Windows 上 reparse 点/OneDrive
            # 占位文件 stat 即抛 WinError 1920，且不是 PermissionError。单个坏文件
            # 只记跳过，绝不让它炸掉整次搜索。
            try:
                if not file_path.is_file():
                    continue
                files_searched += 1
                if file_path.stat().st_size > _MAX_FILE_SIZE:
                    continue
                text = file_path.read_text(encoding="utf-8")
            except OSError:
                files_skipped.append(str(file_path))
                continue
            except UnicodeDecodeError:
                files_skipped.append(str(file_path))
                continue

            for line_num, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    if single_file:
                        rel = file_path.name
                    else:
                        try:
                            rel = file_path.relative_to(base)
                        except ValueError:
                            rel = file_path
                    results.append(f"{rel}:{line_num}: {line.strip()}")
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break

        suffix_parts = []
        if len(results) >= limit:
            suffix_parts.append(f"... 结果过多，仅显示前 {limit} 条")
        if files_skipped:
            skipped_list = files_skipped[:5]
            if len(files_skipped) > 5:
                skipped_list.append(f"... 共 {len(files_skipped)} 个")
            suffix_parts.append(
                f"⚠ 跳过 {len(files_skipped)} 个无法读取的文件：\n  "
                + "\n  ".join(skipped_list)
            )
        if total_files > _MAX_TOTAL_FILES:
            suffix_parts.append(f"... 扫描已截断（超过 {_MAX_TOTAL_FILES} 个文件），请缩小搜索范围")
        suffix = "\n".join(suffix_parts)

        # 结果首行标明搜索根：模型容易把"在哪搜的"弄丢，把 cwd 的结果说成
        # 别的项目"里没有"（毒记忆事故的认知源头）。搜索范围是结论的一部分。
        scope = f"[搜索根: {base}]"
        if not results:
            header = (
                f"{scope}\n未找到匹配 '{pattern}' 的内容（已搜索 {files_searched} 个文件）。"
                f"注意：只能说明该范围内没搜到，不代表其他目录/项目里不存在。"
            )
            return f"{header}\n{suffix}" if suffix else header

        out = scope + "\n" + "\n".join(results)
        if suffix:
            out += "\n" + suffix
        return out
