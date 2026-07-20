from __future__ import annotations
import asyncio
import logging
import random
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI

from loongcli.core.messages import inline_image_refs

logger = logging.getLogger(__name__)

# 单次请求的图片配额。图片 token 成本高且不可部分截断，设上限防失控累积/巨图。
MAX_IMAGES_PER_REQUEST = 20
MAX_IMAGE_DATA_URL_CHARS = 15 * 1024 * 1024  # ~15MB base64 data URL（约 11MB 原图）

# openai SDK 默认 timeout 600s：流挂死（服务端停发但 TCP 未断）要等 10 分钟
# 才报错，TUI 体感就是"卡住"。read 超时是相邻 chunk 的间隔上限——thinking
# 阶段 reasoning delta 也在持续流动，180s 收不到任何字节必属挂死；connect
# 收紧到 10s 早失败早重试。
_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


def count_and_validate_images(messages: list[dict]) -> int:
    """扫描 messages 里的 image_url 块，校验单图体积与总数量。返回图片总数。

    纯文本消息（content 非 list）直接跳过，开销可忽略。超限抛 ValueError，
    由调用方转成明确错误——图片是原子的，宁可提前拒绝也别让 API 端糊涂报错。
    """
    n = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            n += 1
            url = (block.get("image_url") or {}).get("url", "")
            if isinstance(url, str) and len(url) > MAX_IMAGE_DATA_URL_CHARS:
                raise ValueError(
                    f"单张图片过大（{len(url) // 1024 // 1024}MB base64，"
                    f"上限 {MAX_IMAGE_DATA_URL_CHARS // 1024 // 1024}MB）"
                )
    if n > MAX_IMAGES_PER_REQUEST:
        raise ValueError(f"单次请求图片过多（{n} 张，上限 {MAX_IMAGES_PER_REQUEST}）")
    return n


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
        vision: bool = False,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=_TIMEOUT)
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self._provider_type = provider_type or _infer_provider(base_url)
        # 该模型是否具备视觉能力（能吃图片）。由 role 配置显式传入，门控图片输入。
        self.vision = vision

    def switch(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        vision: bool | None = None,
    ) -> None:
        """一致性切换模型/供应商：模型名、端点、密钥、provider_type、vision 一起更新。

        防两类半切换事故：换模型名不换 base_url（跨供应商 400）；vision 残留
        （把图喂给纯文本模型，或反向误拒）。base_url/api_key 未变时复用现有连接。
        vision 传 None 表示调用方明确"保持不变"（仅限同模型场景，谨慎使用）。
        """
        new_key = api_key or self._api_key
        new_url = base_url or self.base_url
        if new_url != self.base_url or new_key != self._api_key:
            self.client = AsyncOpenAI(api_key=new_key, base_url=new_url, timeout=_TIMEOUT)
            self.base_url = new_url
            self._api_key = new_key
        self.model = model
        self._provider_type = _infer_provider(self.base_url)
        if vision is not None:
            self.vision = vision

    # cache_aware 属性已删（2026-07-19）：曾用于「压缩金字塔 vs 保前缀」的 provider 分流，
    # 金字塔删除后请求路径统一为保前缀发完整历史，分流失去消费者。provider 缓存实测
    # 结论（DS 1:120、qwen 稳态命中 72%）存档于 tests/e2e_cache.py 与 interview-qa Q43。

    def _build_thinking_params(self, kwargs: dict, stream: bool = False) -> None:
        """Inject provider-specific thinking/reasoning parameters."""
        pt = self._provider_type
        # Qwen(dashscope 兼容)：思考模式用 extra_body.enable_thinking 控制，不认
        # reasoning_effort（它只接受 none/minimal/.../xhigh，发 max 会 400）。且思考
        # 模式要求流式，非流式一律关，避免"thinking mode requires stream"报错。
        if pt == "qwen":
            kwargs.setdefault("extra_body", {})["enable_thinking"] = bool(self.thinking and stream)
            return

        if not self.thinking:
            if pt == "deepseek":
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            return

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
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning("LLM chat failed (attempt %d/%d): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
        raise last_error

    async def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator:
        # 图片引用（loongimg://）在此单点内联成 base64 data URL——所有调用方
        # （agent 主循环、compactor 摘要、子代理）都经这里，消息历史里只留引用。
        # 随后校验图片配额，放在重试循环外——参数错误重试无意义。
        messages = inline_image_refs(messages)
        count_and_validate_images(messages)
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        self._build_thinking_params(kwargs, stream=True)

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            yielded = False
            try:
                stream = await self.client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    yielded = True
                    yield chunk
                return
            except Exception as e:
                if yielded:
                    # 流已开始产出：部分 chunk 已被消费方处理（渲染上屏、灌进
                    # collector），从头重试会把新流重复叠进同一消费方——正文重复、
                    # 工具参数拼接损坏。只能上抛，由 agent 层显式报错给用户。
                    logger.warning("LLM 流中断（已产出部分 chunk，不可重试）: %s", e)
                    raise
                last_error = e
                logger.warning("LLM request failed (attempt %d/%d): %s", attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    await asyncio.sleep(delay)

        raise last_error


def _infer_provider(base_url: str) -> str:
    url = base_url.lower()
    if "deepseek" in url:
        return "deepseek"
    if "dashscope" in url or "aliyuncs" in url:
        return "qwen"
    if "openai" in url:
        return "openai"
    if "anthropic" in url:
        return "anthropic"
    if "localhost" in url or "127.0.0.1" in url:
        return "local"
    return "openai"
