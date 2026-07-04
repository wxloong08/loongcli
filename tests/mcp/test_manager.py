import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.shared.exceptions import McpError, ErrorData
from loongcli.mcp.manager import MCPManager, MCPTool, _MAX_RETRIES
from loongcli.tools.base import ToolRegistry


def test_register_tools():
    mock_session = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.description = "Search the web"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    mcp_tool = MCPTool(server_name="searxng", tool=mock_tool, session=mock_session)
    assert mcp_tool.name == "searxng__search"
    assert mcp_tool.description == "Search the web"

    registry = ToolRegistry()
    registry.register(mcp_tool)
    schemas = registry.get_tool_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "searxng__search"


async def test_mcp_tool_execute_text():
    mock_session = AsyncMock()

    from mcp import types
    mock_session.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="result data")]
    )

    mock_tool = MagicMock()
    mock_tool.name = "echo"
    mock_tool.description = "Echo"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    tool = MCPTool(server_name="test", tool=mock_tool, session=mock_session)
    result = await tool.execute(message="hello")

    mock_session.call_tool.assert_called_once_with("echo", arguments={"message": "hello"})
    assert result == "result data"


async def test_mcp_tool_execute_non_retryable_error():
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = RuntimeError("bad argument")

    mock_tool = MagicMock()
    mock_tool.name = "broken"
    mock_tool.description = ""
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    tool = MCPTool(server_name="srv", tool=mock_tool, session=mock_session)
    result = await tool.execute()
    assert "MCP tool error" in result
    assert "bad argument" in result
    assert mock_session.call_tool.call_count == 1


async def test_mcp_tool_retries_on_network_error():
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = ConnectionError("connection reset")

    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.description = ""
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    tool = MCPTool(server_name="srv", tool=mock_tool, session=mock_session)
    with patch("loongcli.mcp.manager.asyncio.sleep", new_callable=AsyncMock):
        result = await tool.execute(query="test")
    assert "retried" in result
    assert mock_session.call_tool.call_count == _MAX_RETRIES + 1


async def test_mcp_tool_retry_succeeds():
    from mcp import types

    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = [
        OSError("timeout"),
        types.CallToolResult(content=[types.TextContent(type="text", text="ok")]),
    ]

    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.description = ""
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    tool = MCPTool(server_name="srv", tool=mock_tool, session=mock_session)
    with patch("loongcli.mcp.manager.asyncio.sleep", new_callable=AsyncMock):
        result = await tool.execute(query="test")
    assert result == "ok"
    assert mock_session.call_tool.call_count == 2


async def test_mcp_tool_raises_mcp_error():
    mock_session = AsyncMock()
    mock_session.call_tool.side_effect = McpError(ErrorData(code=-1, message="invalid params"))

    mock_tool = MagicMock()
    mock_tool.name = "broken"
    mock_tool.description = ""
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    tool = MCPTool(server_name="srv", tool=mock_tool, session=mock_session)
    with pytest.raises(McpError):
        await tool.execute()
    assert mock_session.call_tool.call_count == 1


def test_get_tool_descriptions():
    mgr = MCPManager()
    assert mgr.get_tool_descriptions() == ""

    mock_session = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "search"
    mock_tool.description = "Web search"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "language": {"type": "string", "description": "Language code"},
        },
        "required": ["query"],
    }

    mgr._tools.append(MCPTool("searxng", mock_tool, mock_session))
    desc = mgr.get_tool_descriptions()
    assert "searxng__search" in desc
    assert "Web search" in desc
    assert "`query` (必需)" in desc
    assert "`language` (可选)" in desc


def test_get_tool_descriptions_no_params():
    mgr = MCPManager()
    mock_session = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "ping"
    mock_tool.description = "Ping"
    mock_tool.inputSchema = {}

    mgr._tools.append(MCPTool("srv", mock_tool, mock_session))
    desc = mgr.get_tool_descriptions()
    assert "srv__ping" in desc
    assert "参数" not in desc


async def test_connect_all_no_servers():
    mgr = MCPManager()
    tools = await mgr.connect_all()
    assert tools == []


async def test_disconnect_all():
    mgr = MCPManager()
    await mgr.disconnect_all()
    assert mgr.server_count == 0
    assert mgr.tool_count == 0


async def test_connect_server_skips_no_command_no_url():
    mgr = MCPManager(servers={"bad": {}})
    tools = await mgr.connect_all()
    assert tools == []
    assert mgr.server_count == 0


async def test_connect_server_uses_stdio_for_command():
    from mcp import types

    mock_session = AsyncMock()
    mock_tool = MagicMock(spec=types.Tool)
    mock_tool.name = "echo"
    mock_tool.description = "Echo tool"
    mock_tool.inputSchema = {"type": "object", "properties": {}}
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])

    with patch("loongcli.mcp.manager.stdio_client") as mock_stdio:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value = mock_ctx

        with patch("loongcli.mcp.manager.ClientSession") as mock_cs:
            mock_cs_ctx = AsyncMock()
            mock_cs_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value = mock_cs_ctx

            mgr = MCPManager(servers={"local": {"command": "echo", "args": ["hello"]}})
            tools = await mgr.connect_all()

            mock_stdio.assert_called_once()
            assert len(tools) == 1
            assert tools[0].name == "local__echo"


async def test_connect_server_uses_http_for_url():
    from mcp import types

    mock_session = AsyncMock()
    mock_tool = MagicMock(spec=types.Tool)
    mock_tool.name = "search"
    mock_tool.description = "Search"
    mock_tool.inputSchema = {"type": "object", "properties": {}}
    mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])

    with patch("loongcli.mcp.manager.streamablehttp_client") as mock_http:
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock(), lambda: None))
        mock_http.return_value = mock_ctx

        with patch("loongcli.mcp.manager.ClientSession") as mock_cs:
            mock_cs_ctx = AsyncMock()
            mock_cs_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value = mock_cs_ctx

            mgr = MCPManager(servers={"remote": {
                "url": "http://example.com/mcp",
                "headers": {"Authorization": "Bearer token"},
            }})
            tools = await mgr.connect_all()

            mock_http.assert_called_once_with(
                url="http://example.com/mcp",
                headers={"Authorization": "Bearer token"},
                timeout=30,
            )
            assert len(tools) == 1
            assert tools[0].name == "remote__search"


# ── ImageContent → sentinel（三期截图闭环：MCP 图片进图片通道） ──

async def test_mcp_tool_execute_image_returns_sentinel():
    import base64
    from mcp import types
    from loongcli.core.messages import parse_image_sentinel, inline_image_refs

    png = b"\x89PNG\r\n\x1a\nscreenshot-bytes"
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="页面已截图"),
            types.ImageContent(
                type="image",
                data=base64.b64encode(png).decode("ascii"),
                mimeType="image/png",
            ),
        ]
    )

    mock_tool = MagicMock()
    mock_tool.name = "browser_take_screenshot"
    mock_tool.description = "Take screenshot"
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    tool = MCPTool(server_name="playwright", tool=mock_tool, session=mock_session)
    result = await tool.execute()

    parsed = parse_image_sentinel(result)
    assert parsed is not None
    images, text = parsed
    assert text == "页面已截图"
    assert len(images) == 1 and images[0][1] == "image/png"

    # 引用可还原出原始截图字节（经内联通道）
    msg = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": images[0][0]}},
    ]}
    inlined = inline_image_refs([msg])
    url = inlined[0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == png


async def test_mcp_tool_bad_image_degrades_to_text():
    from mcp import types

    mock_session = AsyncMock()
    mock_session.call_tool.return_value = types.CallToolResult(
        content=[
            types.ImageContent(type="image", data="bm90LWFuLWltYWdl", mimeType="image/png"),
        ]
    )

    mock_tool = MagicMock()
    mock_tool.name = "shot"
    mock_tool.description = ""
    mock_tool.inputSchema = {"type": "object", "properties": {}}

    tool = MCPTool(server_name="x", tool=mock_tool, session=mock_session)
    result = await tool.execute()
    # 字节不是图片（magic 校验不过）→ 降级为错误文本，不抛异常、不产生 sentinel
    assert "图片处理失败" in result
