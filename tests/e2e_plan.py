"""End-to-end Plan mode test — hits real DeepSeek API.

Tests that disable_tools=True actually prevents tool calls during planning,
and that the plan text is generated correctly.
"""
import asyncio
import sys

from loongcli.core.config import Config
from loongcli.core.llm import LLMClient
from loongcli.core.agent import AgentLoop
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.tools.read_file import ReadFileTool
from loongcli.tools.shell import ShellTool
from loongcli.security.checker import SecurityChecker


async def main():
    cfg = Config.load()
    if not cfg.api_key:
        print("ERROR: api_key not set")
        sys.exit(1)

    print(f"Config: model={cfg.model}")

    llm = LLMClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ShellTool())
    security = SecurityChecker()

    agent = AgentLoop(
        llm=llm, tool_registry=registry, security=security,
        system_prompt="你是一个助手。用工具完成任务。",
    )

    results = []

    # Test 1: Plan phase — disable_tools=True should produce text only, no tool calls
    print("\n" + "=" * 60)
    print("TEST 1: Plan phase (disable_tools=True, should NOT call tools)")
    print("=" * 60)

    plan_prompt = (
        "请先制定一个详细的分步执行计划，不要直接执行任何工具。\n"
        "按步骤编号列出，每步说明要做什么、用什么工具、预期结果。\n\n"
        "任务：读取 pyproject.toml 文件并告诉我项目名称和版本号。"
    )

    plan_text = ""
    tool_called = False
    try:
        async for event in agent.run_stream(plan_prompt, disable_tools=True):
            if isinstance(event, TextDelta):
                plan_text += event.text
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                tool_called = True
                print(f"\n  [TOOL CALLED — BUG!] {event.tool_name}")
            elif isinstance(event, AgentDone):
                print("\n  [DONE]")

        if tool_called:
            print("  >> FAIL: tools were called during plan phase!")
            results.append(False)
        elif not plan_text.strip():
            print("  >> FAIL: no plan text generated")
            results.append(False)
        else:
            print("  >> PASS: plan generated without tool calls")
            results.append(True)
    except Exception as e:
        print(f"\n  >> FAIL: {e}")
        results.append(False)

    # Test 2: Execute phase — should call tools
    print("\n" + "=" * 60)
    print("TEST 2: Execute phase (tools enabled, should call tools)")
    print("=" * 60)

    execute_prompt = f"用户已确认以下计划，请严格按步骤执行：\n\n{plan_text}"
    tool_called = False
    try:
        async for event in agent.run_stream(execute_prompt):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                tool_called = True
                print(f"\n  [TOOL] {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolCallResult):
                preview = event.result[:200] if len(event.result) > 200 else event.result
                print(f"  [RESULT] {preview}")
            elif isinstance(event, AgentDone):
                print(f"\n  [DONE]")

        if tool_called:
            print("  >> PASS: tools were called during execute phase")
            results.append(True)
        else:
            print("  >> FAIL: no tools called during execute phase")
            results.append(False)
    except Exception as e:
        print(f"\n  >> FAIL: {e}")
        results.append(False)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
