from __future__ import annotations
from pathlib import Path
from loongcli.tools.base import Tool
from loongcli.tools.errors import ToolError
from loongcli.tools.syntax_check import check_syntax

_MAX_CONTENT_SIZE = 2 * 1024 * 1024  # 2 MB


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
            if len(content) > _MAX_CONTENT_SIZE:
                size_mb = len(content) / (1024 * 1024)
                return (
                    f"错误：内容过大（{size_mb:.1f} MB，超过 {_MAX_CONTENT_SIZE // (1024 * 1024)} MB 上限）。"
                    f"请拆分为多个文件或使用 Shell 工具写入。"
                )
            # 写盘前语法校验（.py/.json/.toml/.yaml），失败即拒绝——
            # 事故复盘：坏配置落盘后服务起不来，拦在写入前而不是靠事后发现。
            err = check_syntax(path, content)
            if err:
                return (
                    f"错误：语法校验未通过，已拒绝写入。{err}\n"
                    f"请修正后重写；如确需写入非法语法内容（如测试夹具），请改用 shell 工具写入。"
                )
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"成功写入 {path}（{len(content)} 字符）"
        except PermissionError:
            raise ToolError(f"权限不足，无法写入 '{path}'", retryable=True, retry_after=1.0)
        except Exception as e:
            return f"错误：{e}"
