import pytest
from unittest.mock import AsyncMock, MagicMock
from loongcli.core.llm import LLMClient


def test_llm_client_init():
    client = LLMClient(api_key="test-key")
    assert client.model == "deepseek-v4-flash"


def test_llm_client_custom_model():
    client = LLMClient(api_key="test-key", model="deepseek-v4-pro")
    assert client.model == "deepseek-v4-pro"


def test_cache_aware_deepseek_base_url():
    c = LLMClient(api_key="k", model="deepseek-v4-pro", base_url="https://api.deepseek.com")
    assert c.cache_aware is True


def test_cache_aware_deepseek_model_via_proxy():
    # 走代理：base_url 不含 deepseek（provider_type 误判成 openai），但 model 名兜住，避免静默退化
    c = LLMClient(api_key="k", model="deepseek-v4-pro", base_url="https://my-proxy.example.com/v1")
    assert c.cache_aware is True


def test_cache_aware_false_for_non_deepseek():
    c = LLMClient(api_key="k", model="gpt-4o", base_url="https://api.openai.com/v1")
    assert c.cache_aware is False


def test_cache_aware_qwen_base_url():
    # 2026-07-03 真机实测（tests/e2e_cache.py）：隐式缓存稳态命中 72%、命中 2 折、位置敏感
    c = LLMClient(
        api_key="k", model="qwen3.7-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    assert c.cache_aware is True


def test_cache_aware_qwen_model_via_proxy():
    c = LLMClient(api_key="k", model="qwen3.7-plus", base_url="https://my-proxy.example.com/v1")
    assert c.cache_aware is True


def test_cache_aware_false_for_untested_providers():
    # Kimi/GLM 有隐式缓存但未接入未实测——接入后跑 e2e_cache 再开
    c = LLMClient(api_key="k", model="kimi-k2.6", base_url="https://api.moonshot.cn/v1")
    assert c.cache_aware is False
    c = LLMClient(api_key="k", model="glm-5.2", base_url="https://open.bigmodel.cn/api/paas/v4")
    assert c.cache_aware is False


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
