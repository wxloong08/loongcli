from __future__ import annotations
from pathlib import Path
from loongcli.tools.base import Tool


class EditFileTool(Tool):
    name = "edit_file"
    description = "Edit a file by replacing an exact string match. The old_string must appear exactly once in the file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "The exact string to find and replace (must be unique in the file)",
            },
            "new_string": {
                "type": "string",
                "description": "The replacement string",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def execute(self, path: str, old_string: str, new_string: str) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"错误：文件不存在 '{path}'"
            if not p.is_file():
                return f"错误：'{path}' 不是文件"

            content = p.read_text(encoding="utf-8")
            count = content.count(old_string)

            if count == 0:
                return "错误：未找到目标字符串"
            if count > 1:
                return (
                    f"错误：目标字符串有 {count} 处匹配（多处匹配），"
                    f"需要唯一匹配。请提供更多上下文使其唯一。"
                )

            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content, encoding="utf-8")
            return f"成功编辑 {path}"
        except Exception as e:
            return f"错误：{e}"
