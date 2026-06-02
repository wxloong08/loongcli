"""End-to-end test: run a real conversation with DeepSeek API."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongcli.core.config import Config
from loongcli.core.llm import LLMClient
from loongcli.core.agent import AgentLoop
from loongcli.core.compact import Compactor
from loongcli.core.prompts import get_system_prompt
from loongcli.tools.base import ToolRegistry
from loongcli.tools.read_file import ReadFileTool
from loongcli.tools.write_file import WriteFileTool
from loongcli.tools.edit_file import EditFileTool
from loongcli.tools.shell import ShellTool
from loongcli.tools.glob_tool import GlobTool
from loongcli.tools.grep_tool import GrepTool
from loongcli.security.checker import SecurityChecker
from loongcli.core.events import (
    TextDelta, ToolCallStart, ToolCallResult, AgentDone,
    CompactNotice, ConfirmRequest,
)


async def main():
    cfg = Config.load()
    if not cfg.api_key:
        print("ERROR: No API key. Set DEEPSEEK_API_KEY or ~/.loongcli/config.json")
        sys.exit(1)

    print(f"Model: {cfg.model}")
    print(f"Base URL: {cfg.base_url}")
    print()

    llm = LLMClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    registry.register(ShellTool())
    registry.register(GlobTool())
    registry.register(GrepTool())

    security = SecurityChecker()
    system_prompt = get_system_prompt(model=cfg.model)
    compactor = Compactor(llm=llm, threshold=cfg.compact_threshold)

    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        security=security,
        system_prompt=system_prompt,
        compactor=compactor,
    )

    test_cases = [
        "你好，请用一句话介绍你自己。",
        "当前目录下有哪些 Python 文件？用 glob 工具查看。",
        "读取 pyproject.toml 的内容。",
    ]

    for i, prompt in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"TEST {i}: {prompt}")
        print(f"{'='*60}")

        buffer = ""
        tool_calls = []

        async for event in agent.run_stream(prompt):
            if isinstance(event, TextDelta):
                buffer += event.text
            elif isinstance(event, ToolCallStart):
                tool_calls.append(event.tool_name)
                print(f"  [TOOL] {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolCallResult):
                result_preview = event.result[:200] if len(event.result) > 200 else event.result
                print(f"  [RESULT] {event.tool_name} -> {result_preview}")
            elif isinstance(event, ConfirmRequest):
                print(f"  [CONFIRM] {event.tool_name}: {event.risk_reason}")
                event.future.set_result(True)
            elif isinstance(event, AgentDone):
                pass

        print(f"\nASSISTANT: {buffer[:500]}")
        print(f"Tools used: {tool_calls}")
        print()

    print("ALL TESTS COMPLETED")


if __name__ == "__main__":
    asyncio.run(main())
