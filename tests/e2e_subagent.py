"""End-to-end SubAgent test — hits real DeepSeek API."""
import asyncio
import sys

from loongcli.core.config import Config
from loongcli.core.llm import LLMClient
from loongcli.core.agent import AgentLoop, AgentServices
from loongcli.core.compact import Compactor
from loongcli.core.task import TaskManager
from loongcli.core.events import (
    TextDelta, ToolCallStart, ToolCallResult, AgentDone, TaskNotification,
)
from loongcli.tools.base import ToolRegistry
from loongcli.tools.read_file import ReadFileTool
from loongcli.tools.shell import ShellTool
from loongcli.tools.glob_tool import GlobTool
from loongcli.tools.agent_tool import AgentTool
from loongcli.tools.send_message import SendMessageTool
from loongcli.tools.task_status import TaskStatusTool
from loongcli.security.permissions import PermissionChecker, PermissionMode


async def main():
    cfg = Config.load()
    if not cfg.api_key:
        print("ERROR: api_key not set in ~/.loongcli/config.json")
        sys.exit(1)

    print(f"Config: model={cfg.model}, base_url={cfg.base_url}")

    llm = LLMClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)
    security = PermissionChecker(PermissionMode.SKIP)
    task_manager = TaskManager()

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ShellTool())
    registry.register(GlobTool())
    registry.register(AgentTool(
        task_manager=task_manager,
        llm=llm,
        parent_registry=registry,
        security=security,
    ))
    registry.register(SendMessageTool(task_manager))
    registry.register(TaskStatusTool(task_manager))

    system_prompt = (
        "你是一个助手。你可以使用 delegate 工具将任务委派给后台 SubAgent。\n"
        "SubAgent 完成后你会收到通知。也可以用 task_status 查询进度。\n"
        "可用工具: read_file, shell, glob, delegate, send_message, task_status\n"
        "注意：delegate 后要用 task_status 轮询结果，不要空等。"
    )

    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=security,
        system_prompt=system_prompt,
        max_iterations=20,
        services=AgentServices(task_manager=task_manager),
    )

    prompt = (
        "请用 delegate 工具派一个 SubAgent 去完成这个任务：读取当前目录下的 pyproject.toml 文件并告诉我项目名称。"
        "派出后，用 task_status 查询结果，最终把项目名称告诉我。"
    )

    print(f"\nPROMPT: {prompt}")
    print("=" * 60)

    passed = False
    try:
        async for event in agent.run_stream(prompt):
            if isinstance(event, TextDelta):
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallStart):
                print(f"\n  [TOOL] {event.tool_name}({event.arguments})")
            elif isinstance(event, ToolCallResult):
                preview = event.result[:300] if len(event.result) > 300 else event.result
                print(f"  [RESULT] {preview}")
            elif isinstance(event, TaskNotification):
                print(f"\n  [NOTIFICATION] task={event.task_id} result={event.result[:200]}")
            elif isinstance(event, AgentDone):
                print(f"\n  [DONE]")

        passed = True
        print("  >> PASS")
    except Exception as e:
        print(f"\n  >> FAIL: {e}")
        import traceback
        traceback.print_exc()

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
