from __future__ import annotations
import inspect
import json
from abc import ABC, abstractmethod

from loongcli.tools.routing import AgentRole, filter_tools


class Tool(ABC):
    name: str
    description: str
    parameters: dict
    supports_progress: bool = False
    # 外部 MCP 工具置 True，权限层据此对其强制确认（内置工具默认 False）。
    is_mcp: bool = False

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        ...


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get_tool_schemas(self, role: AgentRole = AgentRole.MAIN) -> list[dict]:
        visible = filter_tools(self._tools, role)
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in visible.values()
        ]

    async def execute(self, tool_call) -> str:
        tool = self._tools[tool_call.function.name]
        args = json.loads(tool_call.function.arguments)
        return await self._call(tool, args)

    async def execute_by_name(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"错误：未知工具 '{name}'"
        return await self._call(self._tools[name], args)

    @staticmethod
    async def _call(tool: Tool, args: dict) -> str:
        unknown = _unknown_args(tool, args)
        if unknown:
            # 模型多传了参数：给出哪个参数不认识、能用哪些，而不是 Python 的 TypeError 原文（14e 真机）
            allowed = ", ".join(tool.parameters.get("properties", {}))
            return f"错误：工具 {tool.name} 不认识参数 {', '.join(unknown)}（未知参数）。可用参数：{allowed}"
        return await tool.execute(**args)


def _unknown_args(tool: Tool, args: dict) -> list[str]:
    """execute 签名接不住的参数名。签名带 **kwargs 的工具（如 MCP 透传）不校验。"""
    params = inspect.signature(tool.execute).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    return [k for k in args if k not in params]
