import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from loongcli.core.llm import (
    LLMClient, count_and_validate_images,
    MAX_IMAGES_PER_REQUEST, MAX_IMAGE_DATA_URL_CHARS,
)


def _img_block(url="data:image/png;base64,ABC"):
    return {"type": "image_url", "image_url": {"url": url}}


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


# ── 视觉：图片配额校验 + list content 透传 ──────────────────────────

class TestImageValidation:
    def test_text_only_zero(self):
        assert count_and_validate_images([{"role": "user", "content": "hi"}]) == 0

    def test_counts_images(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "x"}, _img_block(), _img_block(),
        ]}]
        assert count_and_validate_images(msgs) == 2

    def test_too_many_raises(self):
        blocks = [_img_block() for _ in range(MAX_IMAGES_PER_REQUEST + 1)]
        with pytest.raises(ValueError, match="图片过多"):
            count_and_validate_images([{"role": "user", "content": blocks}])

    def test_oversized_raises(self):
        big = "data:image/png;base64," + "A" * (MAX_IMAGE_DATA_URL_CHARS + 1)
        with pytest.raises(ValueError, match="过大"):
            count_and_validate_images([{"role": "user", "content": [_img_block(big)]}])


@pytest.mark.asyncio
async def test_chat_stream_passes_list_content_through():
    llm = LLMClient(api_key="test")
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "ok"
    chunk.choices[0].delta.tool_calls = None
    chunk.usage = None

    async def _stream(**kwargs):
        yield chunk

    llm.client.chat.completions.create = AsyncMock(return_value=_stream())

    img_msg = {"role": "user", "content": [{"type": "text", "text": "看图"}, _img_block()]}
    out = [c async for c in llm.chat_stream(messages=[img_msg])]
    assert len(out) == 1

    sent = llm.client.chat.completions.create.call_args[1]["messages"]
    # image_url 块原样透传，未被 mangle
    assert sent[0]["content"][1]["type"] == "image_url"
    assert sent[0]["content"][1]["image_url"]["url"] == "data:image/png;base64,ABC"


@pytest.mark.asyncio
async def test_chat_stream_inlines_image_refs(tmp_path):
    """loongimg:// 引用在发送前内联成 base64 data URL；调用方消息里仍是引用。"""
    from loongcli.core.messages import store_image, IMAGE_REF_PREFIX

    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\ndata")
    ref = store_image(str(p))

    llm = LLMClient(api_key="test")
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "ok"
    chunk.choices[0].delta.tool_calls = None
    chunk.usage = None

    async def _stream(**kwargs):
        yield chunk

    llm.client.chat.completions.create = AsyncMock(return_value=_stream())

    img_msg = {"role": "user", "content": [{"type": "text", "text": "看图"}, _img_block(ref)]}
    [c async for c in llm.chat_stream(messages=[img_msg])]

    sent = llm.client.chat.completions.create.call_args[1]["messages"]
    assert sent[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    # 调用方的消息历史未被就地改写，仍是引用
    assert img_msg["content"][1]["image_url"]["url"] == ref
    assert ref.startswith(IMAGE_REF_PREFIX)


@pytest.mark.asyncio
async def test_chat_stream_rejects_too_many_images():
    llm = LLMClient(api_key="test")
    llm.client.chat.completions.create = AsyncMock()
    blocks = [_img_block() for _ in range(MAX_IMAGES_PER_REQUEST + 1)]
    with pytest.raises(ValueError, match="图片过多"):
        [c async for c in llm.chat_stream(messages=[{"role": "user", "content": blocks}])]
    llm.client.chat.completions.create.assert_not_called()
