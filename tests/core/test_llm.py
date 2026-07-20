import pytest
from unittest.mock import AsyncMock, MagicMock
from loongcli.core.llm import LLMClient


def test_llm_client_init():
    client = LLMClient(api_key="test-key")
    assert client.model == "deepseek-v4-flash"


def test_llm_client_custom_model():
    client = LLMClient(api_key="test-key", model="deepseek-v4-pro")
    assert client.model == "deepseek-v4-pro"


# cache_aware 属性测试已删（2026-07-19）：属性随压缩金字塔一起移除（请求路径统一
# 保前缀发完整历史，分流无消费者）。provider 缓存实测结论存档于 tests/e2e_cache.py。


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


def test_client_timeout_configured():
    """SDK 默认 timeout 600s——流挂死要等 10 分钟才报错，体感"卡住"。
    必须显式收紧：read 180s（相邻 chunk 间隔上限）、connect 10s。"""
    c = LLMClient(api_key="test-key")
    assert c.client.timeout.read == 180.0
    assert c.client.timeout.connect == 10.0


@pytest.mark.asyncio
async def test_chat_stream_midstream_break_reraises_without_retry():
    """流已产出部分 chunk 后断开：必须上抛而非从头重试——部分 chunk 已被
    消费方渲染/收集，从头重试会把新流重复叠进同一消费方（正文重复、
    工具参数拼接损坏，"响应一部分然后没了"的元凶之一）。"""
    client = LLMClient(api_key="test-key", max_retries=3, retry_delay=0.01)

    async def broken_stream():
        yield MagicMock(choices=[MagicMock(delta=MagicMock(content="部分", tool_calls=None))])
        raise ConnectionError("connection lost")

    call_count = 0

    async def create(**kwargs):
        nonlocal call_count
        call_count += 1
        return broken_stream()

    client.client = AsyncMock()
    client.client.chat.completions.create = create

    received = []
    with pytest.raises(ConnectionError):
        async for chunk in client.chat_stream(messages=[{"role": "user", "content": "hi"}]):
            received.append(chunk)

    assert len(received) == 1  # 断前的 chunk 已交付给消费方
    assert call_count == 1  # 没有从头重试


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


def test_switch_same_provider_keeps_client():
    c = LLMClient(api_key="k", model="deepseek-v4-flash", base_url="https://api.deepseek.com")
    old_client = c.client
    c.switch(model="deepseek-v4-pro")
    assert c.model == "deepseek-v4-pro"
    assert c.client is old_client  # 端点/密钥没变，连接复用


def test_switch_cross_provider_rebuilds_all():
    c = LLMClient(api_key="dk", model="deepseek-v4-pro",
                  base_url="https://api.deepseek.com", vision=False)
    old_client = c.client
    c.switch(model="qwen3.7-plus", api_key="qk",
             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", vision=True)
    assert c.model == "qwen3.7-plus"
    assert "dashscope" in c.base_url
    assert c.client is not old_client
    assert c.vision is True
    assert c._provider_type == "qwen"  # provider_type 已按新端点重推断


def test_switch_vision_none_keeps():
    c = LLMClient(api_key="k", model="qwen3.7-plus",
                  base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", vision=True)
    c.switch(model="qwen3.7-plus")
    assert c.vision is True
