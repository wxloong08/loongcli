"""Test skill auto-activation with real DeepSeek API."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from loongcli.core.config import Config
from loongcli.core.llm import LLMClient
from loongcli.core.agent import AgentLoop
from loongcli.tools.base import ToolRegistry
from loongcli.tools.read_file import ReadFileTool
from loongcli.tools.shell import ShellTool
from loongcli.tools.glob_tool import GlobTool
from loongcli.tools.grep_tool import GrepTool
from loongcli.security.permissions import PermissionChecker, PermissionMode
from loongcli.skills.registry import SkillRegistry
from loongcli.core.events import TextDelta, ToolCallStart, ToolCallResult, AgentDone


async def test_skill(prompt: str, expect_skill: str | None):
    cfg = Config.load()
    llm = LLMClient(api_key=cfg.api_key, model=cfg.model, base_url=cfg.base_url)

    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ShellTool())
    registry.register(GlobTool())
    registry.register(GrepTool())

    skill_reg = SkillRegistry(
        project_dir=Path.cwd(),
        personal_dir=Path.home() / ".loongcli" / "skills",
    )

    perm = PermissionChecker(mode=PermissionMode.SKIP)

    agent = AgentLoop(
        llm=llm,
        tool_registry=registry,
        permission_checker=perm,
        system_prompt="你是一个通用 AI Agent。简洁回答，不超过 200 字。",
        skill_registry=skill_reg,
    )

    matched = skill_reg.match(prompt)
    matched_names = [m.name for m in matched]
    print(f"  Trigger matched: {matched_names}")

    if expect_skill:
        assert expect_skill in matched_names, f"Expected {expect_skill} in {matched_names}"

    buffer = ""
    tools_used = []

    async for event in agent.run_stream(prompt):
        if isinstance(event, TextDelta):
            buffer += event.text
        elif isinstance(event, ToolCallStart):
            tools_used.append(event.tool_name)
        elif isinstance(event, AgentDone):
            pass

    user_msg = agent.messages[0]["content"]
    skill_injected = "[自动加载 skill:" in user_msg
    print(f"  Skill injected into prompt: {skill_injected}")
    print(f"  Tools used: {tools_used}")
    print(f"  Response ({len(buffer)} chars): {buffer[:300]}...")
    print()
    return buffer


async def main():
    cfg = Config.load()
    if not cfg.api_key:
        print("ERROR: No API key")
        sys.exit(1)

    print(f"Model: {cfg.model}\n")

    tests = [
        ("我有个想法，做一个 CLI 工具的自动补全功能", "brainstorm"),
        ("帮我重构 loongcli/core/agent.py 的错误处理逻辑", "karpathy-guidelines"),
        ("帮我设计一个好看的个人主页 landing page", "frontend-design"),
        ("当前目录下有哪些文件？", None),  # no skill should match
    ]

    for prompt, expect in tests:
        print(f"{'='*60}")
        print(f"TEST: {prompt}")
        print(f"Expected skill: {expect or '(none)'}")
        print(f"{'='*60}")
        try:
            await test_skill(prompt, expect)
        except Exception as e:
            print(f"  ERROR: {e}\n")

    print("ALL TESTS DONE")


if __name__ == "__main__":
    asyncio.run(main())
