import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from loongcli.core.llm import LLMClient


@pytest.mark.asyncio
async def test_chat_returns_string():
    llm = LLMClient(api_key="test")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello!"

    llm.client.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await llm.chat("Hi")
    assert result == "Hello!"

    call_kwargs = llm.client.chat.completions.create.call_args[1]
    assert call_kwargs["stream"] is False
    assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_chat_returns_empty_on_none_content():
    llm = LLMClient(api_key="test")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None

    llm.client.chat.completions.create = AsyncMock(return_value=mock_response)
    result = await llm.chat("Hi")
    assert result == ""
