import pytest
from unittest.mock import AsyncMock, MagicMock
from loongcli.core.llm import LLMClient


def test_llm_client_init():
    client = LLMClient(api_key="test-key")
    assert client.model == "deepseek-chat"


def test_llm_client_custom_model():
    client = LLMClient(api_key="test-key", model="deepseek-v4-pro")
    assert client.model == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_chat_stream_yields_chunks():
    client = LLMClient(api_key="test-key")

    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock()]
    mock_chunk_1.choices[0].delta.content = "Hello"
    mock_chunk_1.choices[0].delta.tool_calls = None

    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock()]
    mock_chunk_2.choices[0].delta.content = " World"
    mock_chunk_2.choices[0].delta.tool_calls = None

    async def mock_stream():
        yield mock_chunk_1
        yield mock_chunk_2

    mock_response = mock_stream()
    client.client = AsyncMock()
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    chunks = []
    async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "Hello"


@pytest.mark.asyncio
async def test_chat_stream_passes_tools():
    client = LLMClient(api_key="test-key")

    async def mock_stream():
        return
        yield

    client.client = AsyncMock()
    client.client.chat.completions.create = AsyncMock(return_value=mock_stream())

    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    chunks = []
    async for chunk in client.chat_stream(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    ):
        chunks.append(chunk)

    call_kwargs = client.client.chat.completions.create.call_args[1]
    assert call_kwargs["tools"] == tools
    assert call_kwargs["stream"] is True


@pytest.mark.asyncio
async def test_chat_stream_retries_on_error():
    client = LLMClient(api_key="test-key", max_retries=2, retry_delay=0.01)

    async def mock_stream():
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="ok", tool_calls=None))])

    call_count = 0

    async def flaky_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise Exception("network error")
        return mock_stream()

    client.client = AsyncMock()
    client.client.chat.completions.create = flaky_create

    chunks = []
    async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert call_count == 2
    assert len(chunks) == 1
