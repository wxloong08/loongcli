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
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        thinking: bool = True,
        reasoning_effort: str = "max",
        provider_type: str = "",
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self._provider_type = provider_type or _infer_provider(base_url)

    def _build_thinking_params(self, kwargs: dict) -> None:
        """Inject provider-specific thinking/reasoning parameters."""
        if not self.thinking:
            if self._provider_type == "deepseek":
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            return

        pt = self._provider_type
        if pt == "deepseek":
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = self.reasoning_effort
        elif pt == "openai":
            kwargs["reasoning_effort"] = self.reasoning_effort
        elif pt == "anthropic":
            budget = {"low": 2048, "medium": 8192, "high": 16384, "max": 32768}
            tokens = budget.get(self.reasoning_effort, 16384)
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": tokens}}

    async def chat(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        self._build_thinking_params(kwargs)
        response = await self.client.chat.completions.create(**kwargs)
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
        self._build_thinking_params(kwargs)

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


def _infer_provider(base_url: str) -> str:
    url = base_url.lower()
    if "deepseek" in url:
        return "deepseek"
    if "openai" in url:
        return "openai"
    if "anthropic" in url:
        return "anthropic"
    if "localhost" in url or "127.0.0.1" in url:
        return "local"
    return "openai"
