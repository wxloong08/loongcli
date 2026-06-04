from __future__ import annotations

import re
from pathlib import Path

from loongcli.tools.base import Tool

_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB per file
_MAX_RESULTS = 500
_MAX_TOTAL_FILES = 5000


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
        },
        "required": ["pattern"],
    }

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "**/*",
        case_insensitive: bool = False,
    ) -> str:
        try:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"错误：无效的正则表达式 — {e}"

        base = Path(path).resolve()
        if not base.is_dir():
            return f"错误：'{path}' 不是目录"

        results: list[str] = []
        files_searched = 0
        files_skipped: list[str] = []
        total_files = 0

        iterator = base.rglob(glob) if "**" in glob else base.glob(glob)
        for file_path in sorted(iterator):
            total_files += 1
            if total_files > _MAX_TOTAL_FILES:
                break
            if not file_path.is_file():
                continue
            files_searched += 1

            try:
                st = file_path.stat()
                if st.st_size > _MAX_FILE_SIZE:
                    continue
                text = file_path.read_text(encoding="utf-8")
            except PermissionError:
                files_skipped.append(str(file_path))
                continue
            except UnicodeDecodeError:
                files_skipped.append(str(file_path))
                continue

            for line_num, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    try:
                        rel = file_path.relative_to(base)
                    except ValueError:
                        rel = file_path
                    results.append(f"{rel}:{line_num}: {line.strip()}")
                    if len(results) >= _MAX_RESULTS:
                        break
            if len(results) >= _MAX_RESULTS:
                break

        suffix_parts = []
        if len(results) >= _MAX_RESULTS:
            suffix_parts.append(f"... 结果过多，仅显示前 {_MAX_RESULTS} 条")
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

        if not results:
            header = f"未找到匹配 '{pattern}' 的内容（已搜索 {files_searched} 个文件）"
            return f"{header}\n{suffix}" if suffix else header

        out = "\n".join(results)
        if suffix:
            out += "\n" + suffix
        return out
