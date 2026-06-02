import pytest
from unittest.mock import MagicMock

from loongcli.core.intent import (
    StopIntent, detect_stop_intent, _parse_intent, TAIL_CHARS,
)


def _mock_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = response_text
    chunk.choices[0].delta.tool_calls = None
    chunk.choices[0].finish_reason = "stop"
    chunk.usage = None

    async def mock_stream(**kwargs):
        yield chunk

    llm.chat_stream = mock_stream
    return llm


class TestParseIntent:
    def test_completed(self):
        assert _parse_intent("COMPLETED") == StopIntent.COMPLETED

    def test_needs_input(self):
        assert _parse_intent("NEEDS_INPUT") == StopIntent.NEEDS_INPUT

    def test_continue(self):
        assert _parse_intent("CONTINUE") == StopIntent.CONTINUE

    def test_stuck(self):
        assert _parse_intent("STUCK") == StopIntent.STUCK

    def test_case_insensitive(self):
        assert _parse_intent("completed") == StopIntent.COMPLETED
        assert _parse_intent("Needs_Input") == StopIntent.NEEDS_INPUT

    def test_with_extra_text(self):
        assert _parse_intent("The intent is CONTINUE.") == StopIntent.CONTINUE

    def test_unknown_defaults_completed(self):
        assert _parse_intent("something random") == StopIntent.COMPLETED

    def test_empty_defaults_completed(self):
        assert _parse_intent("") == StopIntent.COMPLETED


class TestDetectStopIntent:
    async def test_completed(self):
        llm = _mock_llm("COMPLETED")
        result = await detect_stop_intent(llm, "全部完成了，共处理 20 个文件。")
        assert result == StopIntent.COMPLETED

    async def test_needs_input(self):
        llm = _mock_llm("NEEDS_INPUT")
        result = await detect_stop_intent(llm, "你希望用 JSON 还是 CSV 格式？")
        assert result == StopIntent.NEEDS_INPUT

    async def test_continue(self):
        llm = _mock_llm("CONTINUE")
        result = await detect_stop_intent(llm, "前 5 个已完成，继续处理剩余的。")
        assert result == StopIntent.CONTINUE

    async def test_stuck(self):
        llm = _mock_llm("STUCK")
        result = await detect_stop_intent(llm, "无法连接数据库，请检查配置。")
        assert result == StopIntent.STUCK

    async def test_empty_text(self):
        calls = []

        async def track(**kwargs):
            calls.append(1)
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "COMPLETED"
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            chunk.usage = None
            yield chunk

        llm = MagicMock()
        llm.chat_stream = track
        result = await detect_stop_intent(llm, "")
        assert result == StopIntent.COMPLETED
        assert len(calls) == 0

    async def test_truncates_long_text(self):
        calls = []

        async def capture(**kwargs):
            calls.append(kwargs)
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "COMPLETED"
            chunk.choices[0].delta.tool_calls = None
            chunk.choices[0].finish_reason = "stop"
            chunk.usage = None
            yield chunk

        llm = MagicMock()
        llm.chat_stream = capture

        long_text = "x" * 5000
        await detect_stop_intent(llm, long_text)

        sent_content = calls[0]["messages"][0]["content"]
        assert "x" * TAIL_CHARS in sent_content
        assert "x" * 5000 not in sent_content

    async def test_llm_failure_defaults_completed(self):
        llm = MagicMock()

        async def fail_stream(**kwargs):
            raise RuntimeError("API down")
            yield  # noqa: make it a generator

        llm.chat_stream = fail_stream
        result = await detect_stop_intent(llm, "some text")
        assert result == StopIntent.COMPLETED
