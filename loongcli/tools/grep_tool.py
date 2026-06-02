from __future__ import annotations

import re
from pathlib import Path

from loongcli.tools.base import Tool


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
        max_results = 500

        for file_path in sorted(base.rglob(glob) if "**" in glob else base.glob(glob)):
            if not file_path.is_file():
                continue
            files_searched += 1
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            for line_num, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    try:
                        rel = file_path.relative_to(base)
                    except ValueError:
                        rel = file_path
                    results.append(f"{rel}:{line_num}: {line.strip()}")
                    if len(results) >= max_results:
                        results.append(
                            f"... 结果过多，仅显示前 {max_results} 条"
                        )
                        return "\n".join(results)

        if not results:
            return f"未找到匹配 '{pattern}' 的内容（已搜索 {files_searched} 个文件）"
        return "\n".join(results)
