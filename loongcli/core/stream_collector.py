from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

from loongcli.core.events import TextDelta, ThinkingDelta


@dataclass
class CollectedResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0

    def to_message(self) -> dict:
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.reasoning_content and self.tool_calls:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


class StreamCollector:
    def __init__(self):
        self.response: CollectedResponse | None = None

    async def collect(self, stream: AsyncIterator) -> AsyncIterator:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_index: dict[int, dict] = {}
        usage = None

        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

            choice = chunk.choices[0] if chunk.choices else None
            if not choice:
                continue

            delta = choice.delta

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning and isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
                yield ThinkingDelta(text=reasoning)

            if delta.content:
                content_parts.append(delta.content)
                yield TextDelta(text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index

                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }

                    tc = tool_calls_by_index[idx]

                    if tc_delta.id:
                        tc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

                    pass

        sorted_tool_calls = [
            tool_calls_by_index[i]
            for i in sorted(tool_calls_by_index.keys())
        ]

        self.response = CollectedResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
            tool_calls=sorted_tool_calls,
        )
        if usage:
            self.response.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            self.response.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            self.response.total_tokens = getattr(usage, "total_tokens", 0) or 0
            self.response.prompt_cache_hit_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            self.response.prompt_cache_miss_tokens = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                self.response.reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
