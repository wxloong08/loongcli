from __future__ import annotations
import logging
from enum import Enum

from loongcli.core.llm import LLMClient
from loongcli.core.stream_collector import StreamCollector

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """\
判断以下AI助手回复的意图类别。只输出一个词。

类别：
- COMPLETED — 任务已完成，在汇报最终结果
- NEEDS_INPUT — 向用户提问、请求选择或确认
- CONTINUE — 中间进展，还有后续工作要做
- STUCK — 遇到障碍无法继续（错误、权限不足、资源不可用等）

回复内容：
{text}

类别："""

TAIL_CHARS = 800


class StopIntent(Enum):
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    CONTINUE = "continue"
    STUCK = "stuck"


_KEYWORD_MAP = {
    "COMPLETED": StopIntent.COMPLETED,
    "NEEDS_INPUT": StopIntent.NEEDS_INPUT,
    "CONTINUE": StopIntent.CONTINUE,
    "STUCK": StopIntent.STUCK,
}


def _parse_intent(raw: str) -> StopIntent:
    token = raw.strip().upper()
    for key, intent in _KEYWORD_MAP.items():
        if key in token:
            return intent
    return StopIntent.COMPLETED


async def detect_stop_intent(llm: LLMClient, text: str) -> StopIntent:
    if not text or not text.strip():
        return StopIntent.COMPLETED

    tail = text[-TAIL_CHARS:] if len(text) > TAIL_CHARS else text

    messages = [
        {"role": "user", "content": CLASSIFY_PROMPT.format(text=tail)},
    ]

    try:
        collector = StreamCollector()
        async for _ in collector.collect(
            llm.chat_stream(messages=messages, tools=None),
        ):
            pass
        result = _parse_intent(collector.response.content)
        logger.debug("Intent detected: %s (raw: %s)", result.value, collector.response.content[:50])
        return result
    except Exception as e:
        logger.warning("Intent detection failed: %s, defaulting to COMPLETED", e)
        return StopIntent.COMPLETED
