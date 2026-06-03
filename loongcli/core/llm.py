from __future__ import annotations
import asyncio
import logging
from typing import AsyncIterator

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        thinking: bool = True,
        reasoning_effort: str = "max",
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort

    async def chat(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        thinking_type = "enabled" if self.thinking else "disabled"
        kwargs["extra_body"] = {"thinking": {"type": thinking_type}}
        if self.thinking:
            kwargs["reasoning_effort"] = self.reasoning_effort

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    yield chunk
                return
            except Exception as e:
                last_error = e
                logger.warning("LLM request failed (attempt %d/%d): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)

        raise last_error
