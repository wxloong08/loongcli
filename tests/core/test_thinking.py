import pytest
import json
from unittest.mock import MagicMock
from loongcli.core.stream_collector import StreamCollector, CollectedResponse
from loongcli.core.events import TextDelta, ThinkingDelta
from loongcli.core.agent import AgentLoop
from loongcli.core.llm import LLMClient
from loongcli.core.events import AgentDone
from loongcli.tools.base import ToolRegistry
from loongcli.security.permissions import PermissionChecker


def _make_chunk(content=None, reasoning_content=None, tool_calls=None, finish_reason=None, usage=None):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    delta = MagicMock(spec=[])
    delta.content = content
    delta.reasoning_content = reasoning_content
    delta.tool_calls = tool_calls
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = finish_reason
    chunk.usage = usage
    return chunk


@pytest.mark.asyncio
async def test_stream_collector_captures_thinking():
    chunks = [
        _make_chunk(reasoning_content="Let me think..."),
        _make_chunk(reasoning_content=" step by step."),
        _make_chunk(content="The answer is 42."),
        _make_chunk(finish_reason="stop"),
    ]

    async def mock_stream():
        for c in chunks:
            yield c

    collector = StreamCollector()
    events = []
    async for event in collector.collect(mock_stream()):
        events.append(event)

    thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
    text_events = [e for e in events if isinstance(e, TextDelta)]
    assert len(thinking_events) == 2
    assert thinking_events[0].text == "Let me think..."
    assert thinking_events[1].text == " step by step."
    assert len(text_events) == 1
    assert text_events[0].text == "The answer is 42."
    assert collector.response.reasoning_content == "Let me think... step by step."
    assert collector.response.content == "The answer is 42."


@pytest.mark.asyncio
async def test_to_message_includes_reasoning_with_tool_calls():
    resp = CollectedResponse(
        content="",
        reasoning_content="thinking here",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
    )
    msg = resp.to_message()
    assert msg["reasoning_content"] == "thinking here"
    assert "tool_calls" in msg


@pytest.mark.asyncio
async def test_to_message_omits_reasoning_without_tool_calls():
    resp = CollectedResponse(
        content="answer",
        reasoning_content="thinking here",
        tool_calls=[],
    )
    msg = resp.to_message()
    assert "reasoning_content" not in msg


@pytest.mark.asyncio
async def test_agent_thinking_events():
    llm = LLMClient(api_key="test", thinking=True)

    chunks = [
        _make_chunk(reasoning_content="Hmm..."),
        _make_chunk(content="Hello!"),
        _make_chunk(finish_reason="stop"),
    ]

    async def mock_stream(**kwargs):
        for c in chunks:
            yield c

    llm.chat_stream = mock_stream

    registry = ToolRegistry()
    checker = PermissionChecker()
    agent = AgentLoop(llm=llm, tool_registry=registry, permission_checker=checker)

    events = []
    async for event in agent.run_stream("hi"):
        events.append(event)

    thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
    text_events = [e for e in events if isinstance(e, TextDelta)]
    done_events = [e for e in events if isinstance(e, AgentDone)]
    assert len(thinking_events) == 1
    assert thinking_events[0].text == "Hmm..."
    assert len(text_events) == 1
    assert len(done_events) == 1


def test_llm_thinking_params():
    llm = LLMClient(api_key="test", thinking=True, reasoning_effort="max")
    assert llm.thinking is True
    assert llm.reasoning_effort == "max"

    llm2 = LLMClient(api_key="test")
    assert llm2.thinking is True
    assert llm2.reasoning_effort == "max"


def test_config_thinking():
    import json as j
    from pathlib import Path
    import tempfile
    from loongcli.core.config import Config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        j.dump({"api_key": "test", "thinking": True, "reasoning_effort": "max"}, f)
        f.flush()
        cfg = Config.load(Path(f.name))
    assert cfg.thinking is True
    assert cfg.reasoning_effort == "max"
