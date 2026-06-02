from __future__ import annotations
from pathlib import Path
from loongcli.tools.base import Tool


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file. Creates the file and parent directories if they don't exist. Overwrites existing content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["path", "content"],
    }

    async def execute(self, path: str, content: str) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"成功写入 {path}（{len(content)} 字符）"
        except Exception as e:
            return f"错误：{e}"
