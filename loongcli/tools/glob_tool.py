from __future__ import annotations
from pathlib import Path
from loongcli.tools.base import Tool


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern. Returns matching file paths."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.py', 'src/*.ts')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Default: current working directory",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, pattern: str, path: str = ".") -> str:
        try:
            base = Path(path).resolve()
            if not base.is_dir():
                return f"错误：'{path}' 不是目录"

            matches = sorted(base.glob(pattern))
            files = [m for m in matches if m.is_file()]

            if not files:
                return f"未找到匹配 '{pattern}' 的文件"

            lines = []
            for f in files[:200]:
                try:
                    rel = f.relative_to(base)
                except ValueError:
                    rel = f
                lines.append(str(rel))

            result = "\n".join(lines)
            if len(files) > 200:
                result += f"\n... 共 {len(files)} 个文件，仅显示前 200 个"
            return result
        except Exception as e:
            return f"错误：{e}"
