from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

from loongcli.core.agent import AgentLoop
from loongcli.core.llm import LLMClient
from loongcli.tools.base import ToolRegistry
from loongcli.security.permissions import PermissionChecker


def _make_chunk(content=None, tool_calls=None, finish_reason=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    delta = MagicMock(spec=[])
    delta.content = content
    delta.tool_calls = tool_calls
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


def _make_tool_call(id_suffix, name, args_json):
    tc = MagicMock()
    tc.id = f"call_{id_suffix}"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args_json
    tc.type = "function"
    return tc


@pytest.mark.asyncio
async def test_verify_loop_not_triggered_without_file_changes():
    """Pure Q&A without file modifications → no verify loop."""
    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        yield _make_chunk(content="Python is a great language for CLI tools.")
        yield _make_chunk(finish_reason="stop")

    llm.chat_stream = mock_stream

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="You are helpful.",
    )

    from loongcli.core.events import AgentDone
    events = []
    async for event in agent.run_stream("what language is good for CLI?"):
        events.append(event)

    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert len(done_events) == 1
    assert not agent._verify_state.is_active


@pytest.mark.asyncio
async def test_system_prompt_includes_verify_section():
    """The system prompt includes the verify loop section."""
    from loongcli.core.prompts import get_system_prompt

    prompt = get_system_prompt(model="deepseek-chat")
    assert "验证循环" in prompt
    assert "3 轮" in prompt


@pytest.mark.asyncio
async def test_agent_has_checkpoint_and_verify():
    """Agent is created with checkpoint_manager and verify_state wired."""
    from loongcli.core.checkpoint import CheckpointManager

    ckpt = CheckpointManager(cwd=Path.cwd())

    agent = AgentLoop(
        llm=LLMClient(api_key="test"),
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="test",
        checkpoint_manager=ckpt,
    )

    assert agent.checkpoint_manager is ckpt
    assert agent._verify_state is not None
    assert agent._verify_state.round == 0
    assert not agent._verify_state.is_active


@pytest.mark.asyncio
async def test_checkpoint_manager_wired_in_main():
    """Import main module's _async_main to verify wiring doesn't crash."""
    from loongcli.main import _async_main
    # Just verify imports work — don't actually run
    from loongcli.core.checkpoint import CheckpointManager
    from loongcli.core.verify_loop import VerifyState, build_verify_prompt
    from loongcli.core.test_discovery import discover_test_command
    assert CheckpointManager is not None
    assert VerifyState is not None
    assert build_verify_prompt is not None
    assert discover_test_command is not None
