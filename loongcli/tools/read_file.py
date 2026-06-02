from __future__ import annotations
from pathlib import Path
from loongcli.tools.base import Tool


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file's contents. Returns text with line numbers."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "offset": {
                "type": "integer",
                "description": "Start reading from this line number (1-based). Default: 1",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of lines to read. Default: 2000",
            },
        },
        "required": ["path"],
    }

    async def execute(self, path: str, offset: int = 1, limit: int = 2000) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"错误：文件不存在 '{path}'"
            if not p.is_file():
                return f"错误：'{path}' 不是文件"

            text = p.read_text(encoding="utf-8")
            lines = text.splitlines()

            start = max(0, offset - 1)
            end = start + limit
            selected = lines[start:end]

            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i}\t{line}")

            return "\n".join(numbered)
        except UnicodeDecodeError:
            return f"错误：无法以 UTF-8 解码 '{path}'（可能是二进制文件）"
        except Exception as e:
            return f"错误：{e}"
