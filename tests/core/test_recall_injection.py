import pytest
from unittest.mock import AsyncMock, MagicMock
from loongcli.core.agent import AgentLoop
from loongcli.core.llm import LLMClient
from loongcli.core.events import TextDelta, AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.security.permissions import PermissionChecker


def _make_chunk(content=None, finish_reason=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    delta = MagicMock(spec=[])
    delta.content = content
    delta.tool_calls = None
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = None
    return chunk


@pytest.mark.asyncio
async def test_recall_injects_memories():
    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        yield _make_chunk(content="Hello!")
        yield _make_chunk(finish_reason="stop")

    llm.chat_stream = mock_stream

    recall_engine = AsyncMock()
    recall_engine.recall = AsyncMock(return_value=[
        {"name": "user-role", "type": "user", "description": "Python dev",
         "content": "Senior Python dev", "created_at": "", "updated_at": ""},
    ])
    recall_engine.format_for_injection = MagicMock(return_value="# 相关记忆\nSenior Python dev")

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="You are helpful.",
        recall_engine=recall_engine,
    )

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    recall_engine.recall.assert_called_once_with("hi")
    system_msgs = [m for m in agent.messages if m.get("role") == "system"]
    assert any("相关记忆" in (m.get("content") or "") for m in system_msgs)


@pytest.mark.asyncio
async def test_recall_failure_does_not_block():
    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        yield _make_chunk(content="Hello!")
        yield _make_chunk(finish_reason="stop")

    llm.chat_stream = mock_stream

    recall_engine = AsyncMock()
    recall_engine.recall = AsyncMock(side_effect=Exception("API down"))

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="You are helpful.",
        recall_engine=recall_engine,
    )

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_no_recall_engine_works():
    llm = LLMClient(api_key="test")

    async def mock_stream(**kwargs):
        yield _make_chunk(content="Hello!")
        yield _make_chunk(finish_reason="stop")

    llm.chat_stream = mock_stream

    agent = AgentLoop(
        llm=llm,
        tool_registry=ToolRegistry(),
        permission_checker=PermissionChecker(),
        system_prompt="You are helpful.",
    )

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert len(done_events) == 1
