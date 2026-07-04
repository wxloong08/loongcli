"""End-to-end MCP test — connects to real SearXNG MCP server."""
import asyncio
import sys

from loongcli.core.config import Config
from loongcli.core.llm import LLMClient
from loongcli.core.agent import AgentLoop
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.mcp.manager import MCPManager
from loongcli.security.permissions import PermissionChecker, PermissionMode


async def main():
    cfg = Config.load()
    if not cfg.api_key:
        print("ERROR: api_key not set")
        sys.exit(1)
    if not cfg.mcp_servers:
        print("ERROR: no MCP servers configured")
        sys.exit(1)

    print(f"Config: model={cfg.model}, mcp_servers={list(cfg.mcp_servers.keys())}")

    # Step 1: Connect MCP
    print("\n[1] Connecting to MCP servers...")
    mcp = MCPManager(servers=cfg.mcp_servers)
    try:
        tools = await mcp.connect_all()
        print(f"    Connected! {mcp.server_count} server(s), {mcp.tool_count} tool(s)")
        for t in tools:
            print(f"    - {t.name}: {t.description[:60]}")
    except Exception as e:
        print(f"    FAILED: {e}")
        sys.exit(1)

    # Step 2: Register tools and run agent
    print("\n[2] Running agent with MCP tools...")
    registry = ToolRegistry()
    mcp.register_tools(registry)
    security = PermissionChecker(PermissionMode.SKIP)

    llm = LLMClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)

    system_prompt = (
        "你是一个助手。你可以使用 MCP 工具来搜索网页。\n"
        + mcp.get_tool_descriptions()
    )

    agent = AgentLoop(
        llm=llm, tool_registry=registry, permission_checker=security,
        system_prompt=system_prompt,
    )

    prompt = "用 searxng 搜索 'Python asyncio tutorial'，告诉我第一个结果的标题。"
    print(f"    PROMPT: {prompt}")
    print(f"    {'='*50}")

    try:
        async for event in agent.run_stream(prompt):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                print(f"\n    [TOOL] {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolCallResult):
                preview = event.result[:300] if len(event.result) > 300 else event.result
                print(f"    [RESULT] {preview}")
            elif isinstance(event, AgentDone):
                print(f"\n    [DONE]")
        print("    >> PASS")
    except Exception as e:
        print(f"\n    >> FAIL: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await mcp.disconnect_all()


if __name__ == "__main__":
    asyncio.run(main())
