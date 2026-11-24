"""工具分发层的参数校验：模型多传参数时给出「未知参数」的明确报错（14e 真机：plan 工具抛 TypeError 原文）。"""
import pytest

from loongcli.tools.base import Tool, ToolRegistry


class _Strict(Tool):
    name = "strict"
    description = ""
    parameters = {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}

    async def execute(self, path: str, limit: int = 0) -> str:
        return f"ok {path} {limit}"


class _Passthrough(Tool):
    name = "passthrough"          # 如 MCP：签名带 **kwargs，参数由下游校验
    description = ""
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return f"ok {sorted(kwargs)}"


@pytest.fixture
def reg():
    r = ToolRegistry()
    r.register(_Strict())
    r.register(_Passthrough())
    return r


@pytest.mark.asyncio
async def test_unknown_arg_reported_with_allowed_list(reg):
    out = await reg.execute_by_name("strict", {"path": "a", "description": "多余"})
    assert "未知参数" in out and "description" in out
    assert "path" in out and "limit" in out


@pytest.mark.asyncio
async def test_known_args_pass(reg):
    assert await reg.execute_by_name("strict", {"path": "a", "limit": 3}) == "ok a 3"


@pytest.mark.asyncio
async def test_kwargs_tool_not_checked(reg):
    assert await reg.execute_by_name("passthrough", {"anything": 1}) == "ok ['anything']"
