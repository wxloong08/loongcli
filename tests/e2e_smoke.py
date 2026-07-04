"""End-to-end smoke test — hits real DeepSeek API."""
import asyncio
import sys
from pathlib import Path

from loongcli.core.config import Config
from loongcli.core.llm import LLMClient
from loongcli.core.agent import AgentLoop
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.tools.read_file import ReadFileTool
from loongcli.tools.shell import ShellTool
from loongcli.tools.glob_tool import GlobTool
from loongcli.tools.grep_tool import GrepTool
from loongcli.security.permissions import PermissionChecker, PermissionMode


async def run_test(name: str, prompt: str, agent: AgentLoop):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")

    try:
        async for event in agent.run_stream(prompt):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                print(f"\n  [TOOL] {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolCallResult):
                preview = event.result[:200] if len(event.result) > 200 else event.result
                print(f"  [RESULT] {preview}")
            elif isinstance(event, AgentDone):
                print(f"\n  [DONE]")
        print("  >> PASS")
        return True
    except Exception as e:
        print(f"\n  >> FAIL: {e}")
        return False


async def main():
    cfg = Config.load()
    if not cfg.api_key:
        print("ERROR: api_key not set in ~/.loongcli/config.json")
        sys.exit(1)

    print(f"Config: model={cfg.model}, base_url={cfg.base_url}")

    llm = LLMClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ShellTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    security = PermissionChecker(PermissionMode.SKIP)

    results = []

    # Test 1: pure text (no tool call)
    agent1 = AgentLoop(llm=llm, tool_registry=registry, permission_checker=security,
                       system_prompt="你是一个助手。简洁回答。")
    results.append(await run_test("纯文本回复", "1+1等于几？一个字回答。", agent1))

    # Test 2: tool call — read file
    agent2 = AgentLoop(llm=llm, tool_registry=registry, permission_checker=security,
                       system_prompt="你是一个助手。用工具完成任务。")
    results.append(await run_test("工具调用 - read_file",
                                  f"读取文件 {Path.cwd() / 'pyproject.toml'} 的前5行，告诉我项目名称。",
                                  agent2))

    # Test 3: tool call — glob
    agent3 = AgentLoop(llm=llm, tool_registry=registry, permission_checker=security,
                       system_prompt="你是一个助手。用工具完成任务。")
    results.append(await run_test("工具调用 - glob",
                                  f"在 {Path.cwd()} 目录下找所有 .py 文件，告诉我有多少个。",
                                  agent3))

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
