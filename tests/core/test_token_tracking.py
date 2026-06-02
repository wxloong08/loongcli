import pytest
from unittest.mock import MagicMock
from loongcli.core.agent import AgentLoop
from loongcli.core.llm import LLMClient
from loongcli.core.events import TextDelta, AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.security.permissions import PermissionChecker


def _make_text_chunks(texts: list[str], prompt_tokens=100, completion_tokens=50):
    chunks = []
    for t in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = t
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = None
        chunk.usage = None
        chunks.append(chunk)
    if chunks:
        chunks[-1].choices[0].finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        usage.total_tokens = prompt_tokens + completion_tokens
        chunks[-1].usage = usage
    return chunks


@pytest.mark.asyncio
async def test_token_usage_accumulates():
    llm = LLMClient(api_key="test")

    call_count = 0

    async def mock_stream(**kwargs):
        nonlocal call_count
        call_count += 1
        for c in _make_text_chunks(["Hello"], prompt_tokens=100, completion_tokens=20):
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    async for _ in agent.run_stream("hi"):
        pass

    assert agent.token_usage["prompt_tokens"] == 100
    assert agent.token_usage["completion_tokens"] == 20
    assert agent.token_usage["total_tokens"] == 120

    async for _ in agent.run_stream("hello again"):
        pass

    assert agent.token_usage["prompt_tokens"] == 200
    assert agent.token_usage["completion_tokens"] == 40
    assert agent.token_usage["total_tokens"] == 240


@pytest.mark.asyncio
async def test_token_usage_starts_at_zero():
    llm = LLMClient(api_key="test")
    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    assert agent.token_usage["prompt_tokens"] == 0
    assert agent.token_usage["completion_tokens"] == 0
    assert agent.token_usage["total_tokens"] == 0
    assert agent.token_usage["prompt_cache_hit_tokens"] == 0
    assert agent.token_usage["reasoning_tokens"] == 0
