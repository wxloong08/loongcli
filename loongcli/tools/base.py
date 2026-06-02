from __future__ import annotations
import json
from abc import ABC, abstractmethod

from loongcli.tools.routing import AgentRole, filter_tools


class Tool(ABC):
    name: str
    description: str
    parameters: dict
    supports_progress: bool = False

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
        return await tool.execute(**args)

    async def execute_by_name(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"错误：未知工具 '{name}'"
        tool = self._tools[name]
        return await tool.execute(**args)
