import pytest
import json
from unittest.mock import MagicMock
from loongcli.core.stream_collector import StreamCollector
from loongcli.core.events import TextDelta


@pytest.mark.asyncio
async def test_collect_text_only():
    chunks = []
    for text in ["Hello", " World"]:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = None
        chunks.append(chunk)

    chunks[-1].choices[0].finish_reason = "stop"

    collector = StreamCollector()
    events = []

    async def fake_stream():
        for c in chunks:
            yield c

    async for event in collector.collect(fake_stream()):
        events.append(event)

    result = collector.response
    assert result.content == "Hello World"
    assert result.tool_calls == []
    assert len(events) == 2
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "Hello"


@pytest.mark.asyncio
async def test_collect_captures_finish_reason():
    """finish_reason 是截断/正常结束/内容过滤的唯一判别依据，必须收进 response。"""
    chunks = []
    for content, fr in [("hi", None), (None, "length")]:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].finish_reason = fr
        chunk.usage = None
        chunks.append(chunk)

    collector = StreamCollector()

    async def fake_stream():
        for c in chunks:
            yield c

    async for _ in collector.collect(fake_stream()):
        pass

    assert collector.response.finish_reason == "length"


@pytest.mark.asyncio
async def test_collect_with_tool_call():
    chunks = []

    # chunk 1: tool call start
    tc_delta_1 = MagicMock()
    tc_delta_1.index = 0
    tc_delta_1.id = "call_123"
    tc_delta_1.type = "function"
    tc_delta_1.function = MagicMock()
    tc_delta_1.function.name = "shell"
    tc_delta_1.function.arguments = '{"comm'

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = None
    chunk1.choices[0].delta.tool_calls = [tc_delta_1]
    chunk1.choices[0].finish_reason = None
    chunks.append(chunk1)

    # chunk 2: tool call args continuation
    tc_delta_2 = MagicMock()
    tc_delta_2.index = 0
    tc_delta_2.id = None
    tc_delta_2.type = None
    tc_delta_2.function = MagicMock()
    tc_delta_2.function.name = None
    tc_delta_2.function.arguments = 'and": "ls"}'

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = None
    chunk2.choices[0].delta.tool_calls = [tc_delta_2]
    chunk2.choices[0].finish_reason = None
    chunks.append(chunk2)

    # chunk 3: finish
    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = None
    chunk3.choices[0].delta.tool_calls = None
    chunk3.choices[0].finish_reason = "tool_calls"
    chunks.append(chunk3)

    collector = StreamCollector()
    events = []

    async def fake_stream():
        for c in chunks:
            yield c

    async for event in collector.collect(fake_stream()):
        events.append(event)

    result = collector.response
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "call_123"
    assert result.tool_calls[0]["function"]["name"] == "shell"
    assert json.loads(result.tool_calls[0]["function"]["arguments"]) == {"command": "ls"}
    assert len(events) == 0


@pytest.mark.asyncio
async def test_collected_response_to_message():
    from loongcli.core.stream_collector import CollectedResponse

    # Text only
    r1 = CollectedResponse(content="hello", tool_calls=[])
    msg1 = r1.to_message()
    assert msg1 == {"role": "assistant", "content": "hello"}

    # With tool calls
    r2 = CollectedResponse(content="", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "test", "arguments": "{}"}}])
    msg2 = r2.to_message()
    assert msg2["role"] == "assistant"
    assert msg2["content"] is None
    assert len(msg2["tool_calls"]) == 1
