from __future__ import annotations
import asyncio
import base64
import binascii
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError

from loongcli.core.messages import make_image_sentinel, store_image_bytes
from loongcli.tools.base import Tool, ToolRegistry

logger = logging.getLogger(__name__)

_RETRYABLE = (OSError, ConnectionError, TimeoutError, asyncio.TimeoutError, EOFError)
_MAX_RETRIES = 2
_BACKOFF_BASE = 1.0


class MCPTool(Tool):
    """Adapts a single MCP server tool into the loongcli Tool interface."""

    is_mcp = True

    def __init__(self, server_name: str, tool: types.Tool, session: ClientSession):
        self.name = f"{server_name}__{tool.name}"
        self.description = tool.description or ""
        self.parameters = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
        self._session = session
        self._remote_name = tool.name

    async def execute(self, **kwargs) -> str:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                result = await self._session.call_tool(self._remote_name, arguments=kwargs)
                parts: list[str] = []
                images: list[tuple[str, str]] = []
                for block in result.content:
                    if isinstance(block, types.TextContent):
                        parts.append(block.text)
                    elif isinstance(block, types.ImageContent):
                        # 图片（如 Playwright 截图）存入本地图片库，经 sentinel 走
                        # agent 的 user 消息注入通道进对话（视觉闭环的回路）。
                        # 单图失败降级为错误文本，不炸整个工具结果。
                        try:
                            raw = base64.b64decode(block.data)
                            images.append((
                                store_image_bytes(raw),
                                block.mimeType or "image",
                            ))
                        except (ValueError, OSError, binascii.Error) as e:
                            parts.append(f"[图片处理失败: {e}]")
                    elif isinstance(block, types.EmbeddedResource):
                        parts.append(f"[resource: {block.resource.uri}]")
                    else:
                        parts.append(str(block))
                if images:
                    return make_image_sentinel(images, text="\n".join(parts))
                return "\n".join(parts) if parts else "(empty result)"
            except McpError:
                raise
            except _RETRYABLE as e:
                last_err = e
                if attempt < _MAX_RETRIES:
                    delay = _BACKOFF_BASE * (2 ** attempt)
                    logger.warning(
                        "MCP tool '%s' failed (attempt %d/%d): %s, retrying in %.1fs",
                        self.name, attempt + 1, _MAX_RETRIES + 1, e, delay,
                    )
                    await asyncio.sleep(delay)
            except Exception as e:
                return f"MCP tool error ({self.name}): {e}"
        return f"MCP tool error ({self.name}): {last_err} (retried {_MAX_RETRIES} times)"


class MCPManager:
    """Manages connections to multiple MCP servers via stdio transport."""

    def __init__(self, servers: dict[str, dict[str, Any]] | None = None):
        self._server_configs = servers or {}
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: list[MCPTool] = []

    async def connect_all(self) -> list[MCPTool]:
        servers = self._server_configs
        if not servers:
            return []

        for name, cfg in servers.items():
            try:
                await self._connect_server(name, cfg)
            except Exception as e:
                logger.warning("Failed to connect MCP server '%s': %s", name, e)

        return list(self._tools)

    async def _connect_server(self, name: str, cfg: dict):
        url = cfg.get("url", "")
        command = cfg.get("command", "")

        if url:
            transport = streamablehttp_client(
                url=url,
                headers=cfg.get("headers"),
                timeout=cfg.get("timeout", 30),
            )
        elif command:
            params = StdioServerParameters(
                command=command,
                args=cfg.get("args", []),
                env=cfg.get("env"),
            )
            transport = stdio_client(params)
        else:
            logger.warning("MCP server '%s' has no command or url, skipping", name)
            return

        streams = await self._exit_stack.enter_async_context(transport)
        # stdio_client yields (read, write); streamablehttp_client yields (read, write, get_session_id)
        read_stream, write_stream = streams[0], streams[1]
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()

        self._sessions[name] = session

        response = await session.list_tools()
        for tool in response.tools:
            mcp_tool = MCPTool(server_name=name, tool=tool, session=session)
            self._tools.append(mcp_tool)

        transport_type = "HTTP" if url else "stdio"
        logger.info(
            "Connected to MCP server '%s' (%s): %d tools",
            name, transport_type, len(response.tools),
        )

    def register_tools(self, registry: ToolRegistry):
        for tool in self._tools:
            registry.register(tool)

    def get_tool_descriptions(self) -> str:
        if not self._tools:
            return ""
        lines = ["MCP 工具："]
        for t in self._tools:
            lines.append(f"- **{t.name}**: {t.description}")
            props = self._tools_params(t)
            if props:
                lines.append(f"  参数: {'; '.join(props)}")
        return "\n".join(lines)

    @staticmethod
    def _tools_params(tool: MCPTool) -> list[str]:
        props = tool.parameters.get("properties", {})
        if not props:
            return []
        required = set(tool.parameters.get("required", []))
        hints = []
        for pname, pdef in props.items():
            marker = "必需" if pname in required else "可选"
            desc = pdef.get("description", "")[:60]
            hints.append(f"`{pname}` ({marker}) {desc}")
        return hints

    @property
    def server_count(self) -> int:
        return len(self._sessions)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    async def disconnect_all(self):
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
